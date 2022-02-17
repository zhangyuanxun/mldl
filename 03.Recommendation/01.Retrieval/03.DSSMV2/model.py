import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from feature_columns import SparseFeat, DenseFeat, SeqSparseFeat
from collections import OrderedDict


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


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)

def hinge_loss(input, target, margin=1.0, reduction='mean', pos_weight=1.0):
    output = target * input + (1 - target) * torch.relu(margin - input)

    # scale by positive weight
    output = output * target * pos_weight + output * (1 - target)
    # if target == 1, output = input
    # if target == 0, output = torch.max(0, margin - input)
    if reduction == 'sum':
        return torch.sum(output)
    else:
        return torch.mean(output)

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


class Similarity(nn.Module):
    def __init__(self, type='cos', dim=-1):
        super(Similarity, self).__init__()
        self.type = type
        self.dim = dim
        self.eps = 1e-8
        assert (self.type == 'cos')

    def forward(self, query, key):
        if self.type == 'cos':
            query_norm = torch.norm(query, dim=self.dim)
            key_norm = torch.norm(key, dim=self.dim)
            score = torch.sum(torch.multiply(query, key), dim=self.dim)
            score = torch.div(score, query_norm * key_norm + self.eps)
            score = torch.clip(score, -1.0, 1.0)

        return score


class PredictionLayer(nn.Module):
    def __init__(self, use_bias=True):
        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, inputs):
        x = inputs
        if self.use_bias:
            x += self.bias
        out = torch.sigmoid(x)
        return out, x, inputs


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

            x = torch.tanh(x)
            x = self.dropout(x)
        return x


def get_regularization_weight(weight_list, l1=0.0, l2=0.0):
    if isinstance(weight_list, torch.nn.parameter.Parameter):
        weight_list = [weight_list]
    else:
        weight_list = list(weight_list)
    return list([(weight_list, l1, l2)])


def get_regularization_loss(regularization_weight, device):
    total_reg_loss = torch.zeros((1,), device=device)

    for weight_list, l1, l2 in regularization_weight:
        for w in weight_list:
            if isinstance(w, tuple):
                parameter = w[1]                  # named_parameters
            else:
                parameter = w

            if l1 > 0:
                total_reg_loss += torch.sum(l1 * torch.abs(parameter))

            if l2 > 0:
                try:
                    total_reg_loss += torch.sum(l2 * torch.square(parameter))
                except AttributeError:
                    total_reg_loss += torch.sum(l2 * parameter * parameter)

    return total_reg_loss


class UserModel(nn.Module):
    def __init__(self, user_feature_columns, hidden_units, embeddings_initializer_std=1e-4, l2_reg_embedding=1e-3,
                 device='cpu'):
        super(UserModel, self).__init__()
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), user_feature_columns)) if len(user_feature_columns) else []

        self.seq_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SeqSparseFeat), user_feature_columns)) if len(user_feature_columns) else []

        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), user_feature_columns)) if len(user_feature_columns) else []

        self.embedding_layers = nn.ModuleDict({feat.embedding_name: nn.Embedding(num_embeddings=feat.feature_size,
                                                                                 embedding_dim=feat.embedding_dim)
                                               for i, feat in enumerate(self.sparse_feature_columns +
                                                                        self.seq_sparse_feature_columns)})
        # initialize the embedding weight
        for tensor in self.embedding_layers.values():
            nn.init.normal_(tensor.weight, mean=0.0, std=embeddings_initializer_std)

        # add regularization_weight
        self.regularization_weight = get_regularization_weight(self.embedding_layers.parameters(),
                                                               l2=l2_reg_embedding)

        # compute input dimension
        inputs_dim = sum([feat.embedding_dim for i, feat in enumerate(self.sparse_feature_columns +
                                                                      self.seq_sparse_feature_columns)]) + \
            sum([feat.dimension for i, feat in enumerate(self.dense_feature_columns)])

        # compute feature position
        self.feature_pos = build_input_feature_position(user_feature_columns)
        self.dnn = DNN(inputs_dim=inputs_dim, hidden_units=hidden_units)
        self.to(device)

    def forward(self, inputs):
        embedding_inputs_list = [self.embedding_layers[feat.embedding_name](
            inputs[:, self.feature_pos[feat.name][0]: self.feature_pos[feat.name][1]].long()
        ) for feat in self.sparse_feature_columns]

        seq_embedding_inputs_list = get_seq_sparse_embedding_inputs(inputs, self.embedding_layers, self.feature_pos,
                                                                    self.seq_sparse_feature_columns, self.device)

        dense_inputs_list = [inputs[:, self.feature_pos[feat.name][0]: self.feature_pos[feat.name][1]]
                             for feat in self.dense_feature_columns]

        dnn_inputs = combine_inputs(embedding_inputs_list + seq_embedding_inputs_list, dense_inputs_list)
        output = self.dnn(dnn_inputs)

        # normalization embedding
        output = F.normalize(output, p=2, dim=1)
        return output


class ItemModel(nn.Module):
    def __init__(self, item_feature_columns, hidden_units, embeddings_initializer_std=1e-4,
                 l2_reg_embedding=1e-3, device='cpu'):
        super(ItemModel, self).__init__()
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), item_feature_columns)) if len(item_feature_columns) else []

        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), item_feature_columns)) if len(item_feature_columns) else []

        self.seq_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SeqSparseFeat), item_feature_columns)) if len(item_feature_columns) else []

        self.embedding_layers = nn.ModuleDict({feat.embedding_name: nn.Embedding(num_embeddings=feat.feature_size,
                                                                                 embedding_dim=feat.embedding_dim)
                                               for i, feat in enumerate(self.sparse_feature_columns + self.seq_sparse_feature_columns)})
        # initialize the embedding weight
        for tensor in self.embedding_layers.values():
            nn.init.normal_(tensor.weight, mean=0.0, std=embeddings_initializer_std)

        # add regularization_weight
        self.regularization_weight = get_regularization_weight(self.embedding_layers.parameters(),
                                                               l2=l2_reg_embedding)

        # compute input dimension
        inputs_dim = sum([feat.embedding_dim for i, feat in enumerate(self.sparse_feature_columns +
                                                                      self.seq_sparse_feature_columns)]) + \
                     sum([feat.dimension for i, feat in enumerate(self.dense_feature_columns)])

        # compute feature position
        self.feature_pos = build_input_feature_position(item_feature_columns)

        self.dnn = DNN(inputs_dim=inputs_dim, hidden_units=hidden_units)
        self.to(device)

    def forward(self, inputs):
        embedding_inputs_list = [self.embedding_layers[feat.embedding_name](
            inputs[:, self.feature_pos[feat.name][0]: self.feature_pos[feat.name][1]].long()
        ) for feat in self.sparse_feature_columns]

        seq_embedding_inputs_list = get_seq_sparse_embedding_inputs(inputs, self.embedding_layers, self.feature_pos,
                                                                    self.seq_sparse_feature_columns, self.device)

        dense_inputs_list = [inputs[:, self.feature_pos[feat.name][0]: self.feature_pos[feat.name][1]]
                             for feat in self.dense_feature_columns]

        dnn_inputs = combine_inputs(embedding_inputs_list + seq_embedding_inputs_list, dense_inputs_list)

        output = self.dnn(dnn_inputs)

        # normalization embedding
        output = F.normalize(output, p=2, dim=1)
        return output


class DSSM(nn.Module):
    def __init__(self, user_feature_columns, item_feature_columns, num_negative, device='cpu'):
        super(DSSM, self).__init__()
        self.device=device
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        self.num_negative = num_negative

        self.user_model = UserModel(self.user_feature_columns, [64, 32])
        self.item_model = ItemModel(self.item_feature_columns, [64, 32])
        self.similarity_layer = Similarity(type='cos', dim=-1)
        self.output_layer = PredictionLayer()
        self.to(device)

    def forward(self, inputs):
        user_last_pos = next(reversed(self.user_model.feature_pos))
        item_last_pos = next(reversed(self.item_model.feature_pos))
        user_inputs = inputs[:, :self.user_model.feature_pos[user_last_pos][1]]
        item_inputs = inputs[:, self.user_model.feature_pos[user_last_pos][1]:]

        user_out = self.user_model(user_inputs)
        item_feature_len = self.item_model.feature_pos[item_last_pos][1]
        cos_sim_scores = list()
        for i in range(self.num_negative + 1):
            item_out = self.item_model(item_inputs[:, i * item_feature_len: (i + 1) * item_feature_len])
            score = self.similarity_layer(user_out, item_out)
            cos_sim_scores.append(score.reshape(-1, 1))
        cos_sim_scores = torch.cat(cos_sim_scores, dim=1)

        return cos_sim_scores

    def generate_user_embedding(self, user_inputs):
        return self.user_model(user_inputs)

    def generate_item_embedding(self, item_inputs):
        return self.item_model(item_inputs)

    def get_embedding_regularization_loss(self):
        user_model_reg_loss = get_regularization_loss(self.user_model.regularization_weight, self.device)
        item_model_reg_loss = get_regularization_loss(self.item_model.regularization_weight, self.device)
        return user_model_reg_loss[0] + item_model_reg_loss[0]

