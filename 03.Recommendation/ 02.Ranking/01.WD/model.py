import torch.nn as nn
import torch
from feature_columns import *
import torch.nn.functional as F


class SequencePoolingLayer(nn.Module):
    def __init__(self, mode='mean', support_masking=False, device='cpu'):
        super(SequencePoolingLayer, self).__init__()
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError('parameter mode should in [sum, mean, max]')
        self.supports_masking = support_masking
        self.mode = mode
        self.device = device
        self.eps = torch.FloatTensor([1e-8]).to(device)
        self.to(device)

    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        # Returns a mask tensor representing the first N positions of each cell.
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def forward(self, seq_value_len_list):
        if self.supports_masking:
            uiseq_embed_list, mask = seq_value_len_list    # [B, T, E], [B, 1]
            mask = mask.float()
            user_behavior_length = torch.sum(mask, 1, keepdim=True)
            mask = mask.unsqueeze(2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list    # [B, T, E], [B, 1]
            mask = self._sequence_mask(user_behavior_length, maxlen=uiseq_embed_list.shape[1], dtype=torch.float32)
            mask = torch.transpose(mask, 1, 2)

        embedding_size = uiseq_embed_list.shape[-1]

        mask = torch.repeat_interleave(mask, embedding_size, dim=2)  # [B, maxlen, E]

        if self.mode == 'max':
            hist = uiseq_embed_list - (1 - mask) * 1e9
            hist = torch.max(hist, dim=1, keepdim=True)[0]
            return hist
        hist = uiseq_embed_list * mask.float()
        hist = torch.sum(hist, dim=1, keepdim=False)

        if self.mode == 'mean':
            self.eps = self.eps.to(user_behavior_length.device)
            hist = torch.div(hist, user_behavior_length.type(torch.float32) + self.eps)

        hist = torch.unsqueeze(hist, dim=1)
        return hist

def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


def combine_inputs(sparse_inputs_list, dense_inputs_list):
    if len(sparse_inputs_list) > 0 and len(dense_inputs_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_inputs_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_inputs_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_inputs_list) > 0:
        return torch.flatten(torch.cat(sparse_inputs_list, dim=-1), start_dim=1)
    elif len(dense_inputs_list) > 0:
        return torch.flatten(torch.cat(dense_inputs_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


def get_seq_sparse_embedding_inputs(inputs, embedding_layers, feature_pos, seq_sparse_feature_columns, device='cpu'):
    seq_sparse_embedding_list = []

    for feat in seq_sparse_feature_columns:
        seq_emb = embedding_layers[feat.embedding_name](
            inputs[:, feature_pos[feat.name][0]: feature_pos[feat.name][1]].long()
        )

        if feat.length_name is None:
            seq_mask = inputs[:, feature_pos[feat.name][0]:feature_pos[feat.name][1]].long() != 0
            emb = SequencePoolingLayer(mode=feat.combiner, support_masking=True, device=device)([seq_emb, seq_mask])
        else:
            seq_length = inputs[:, feature_pos[feat.length_name][0]:feature_pos[feat.length_name][1]].long()
            emb = SequencePoolingLayer(mode=feat.combiner, support_masking=False, device=device)([seq_emb, seq_length])

        seq_sparse_embedding_list.append(emb)

    return seq_sparse_embedding_list



class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, use_bn=False,
                 dropout_rate=0., init_std=0.0001, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.use_bn = use_bn
        self.device = device
        hidden_units = [inputs_dim] + hidden_units

        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)]
        )

        for name, tensor in self.linear_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(tensor)

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)]
            )
        self.to(device)

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)

            if self.use_bn:
                x = self.bn[i](x)

            x = torch.relu(x)
            x = self.dropout(x)
        return x


class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)


def build_input_feature_position(feature_columns):
    feature_pos = OrderedDict()

    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in feature_pos:
            continue

        if isinstance(feat, SparseFeat):
            feature_pos[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            feature_pos[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        elif isinstance(feat, SeqSparseFeat):
            feature_pos[feat_name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None and feat.length_name not in feature_pos:
                feature_pos[feat.length_name] = (start, start + 1)
                start += 1
        else:
            raise TypeError("Invalid feature column type,got", type(feat))

    return feature_pos


class WideDeep(nn.Module):
    def __init__(self, feature_columns, hidden_units=[256, 128], use_bn=False, dropout_rate=0.,
                 embeddings_initializer_std=0.0001, device='cpu'):
        super(WideDeep, self).__init__()
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.seq_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SeqSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_layers = nn.ModuleDict({feat.embedding_name: nn.Embedding(num_embeddings=feat.feature_size,
                                                                                 embedding_dim=feat.embedding_dim)
                                               for i, feat in enumerate(
                self.sparse_feature_columns + self.seq_sparse_feature_columns)})

        # initialize the embedding weight
        for tensor in self.embedding_layers.values():
            nn.init.normal_(tensor.weight, mean=0.0, std=embeddings_initializer_std)

        # compute feature position
        self.feature_pos = build_input_feature_position(feature_columns)

        # compute input dimension
        deep_inputs_dim = sum([feat.embedding_dim for i, feat in enumerate(self.sparse_feature_columns +
                                                                           self.seq_sparse_feature_columns)]) + \
                          sum([feat.dimension for i, feat in enumerate(self.dense_feature_columns)])

        # deep part
        self.deep = DNN(inputs_dim=deep_inputs_dim, hidden_units=hidden_units)
        self.deep_linear = nn.Linear(hidden_units[-1], 1)

        # wide part
        wide_inputs_dim = sum([feat.dimension for i, feat in enumerate(self.dense_feature_columns)])
        self.wide = LinearModel(input_dim=wide_inputs_dim)

        self.to(device)

    def forward(self, inputs):
        embedding_inputs_list = [self.embedding_layers[feat.embedding_name](
            inputs[:, self.feature_pos[feat.name][0]: self.feature_pos[feat.name][1]].long()
        ) for feat in self.sparse_feature_columns]

        seq_embedding_inputs_list = get_seq_sparse_embedding_inputs(inputs, self.embedding_layers, self.feature_pos,
                                                                    self.seq_sparse_feature_columns, self.device)

        dense_inputs_list = [inputs[:, self.feature_pos[feat.name][0]: self.feature_pos[feat.name][1]]
                             for feat in self.dense_feature_columns]

        # deep part
        deep_inputs = combine_inputs(embedding_inputs_list + seq_embedding_inputs_list, dense_inputs_list)
        deep_outputs = self.deep(deep_inputs)
        deep_outputs = self.deep_linear(deep_outputs)

        # wide part
        wide_inputs = combine_inputs([], dense_inputs_list)
        wide_outputs = self.wide(wide_inputs)

        output = torch.sigmoid(0.5 * (deep_outputs + wide_outputs))
        return output






