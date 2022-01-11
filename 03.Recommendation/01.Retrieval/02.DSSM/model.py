import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from feature_columns import SparseFeat, DenseFeat
from collections import OrderedDict


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
        else:
            raise TypeError("Invalid feature column type,got", type(feat))

    return feature_pos


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
        return out


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, use_bn=True,
                 dropout_rate=0., init_std=0.0001):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.use_bn = use_bn
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

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)

            if self.use_bn:
                x = self.bn[i](x)

            x = torch.tanh(x)
            x = self.dropout(x)
        return x


class UserModel(nn.Module):
    def __init__(self, user_feature_columns, hidden_units, embeddings_initializer_std=0.0001):
        super(UserModel, self).__init__()
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), user_feature_columns)) if len(user_feature_columns) else []

        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), user_feature_columns)) if len(user_feature_columns) else []

        self.embedding_layers = nn.ModuleDict({feat.embedding_name: nn.Embedding(num_embeddings=feat.feature_size,
                                                                                 embedding_dim=feat.embedding_dim)
                                               for i, feat in enumerate(self.sparse_feature_columns)})
        # initialize the embedding weight
        for tensor in self.embedding_layers.values():
            nn.init.normal_(tensor.weight, mean=0.0, std=embeddings_initializer_std)

        # compute input dimension
        inputs_dim = sum([feat.embedding_dim for i, feat in enumerate(self.sparse_feature_columns)]) + \
                     sum([feat.dimension for i, feat in enumerate(self.dense_feature_columns)])

        # compute feature position
        self.feature_pos = build_input_feature_position(user_feature_columns)
        self.dnn = DNN(inputs_dim=inputs_dim, hidden_units=hidden_units)

    def forward(self, inputs):
        embedding_inputs_list = [self.embedding_layers[feat.embedding_name](
            inputs[:, self.feature_pos[feat.name][0]: self.feature_pos[feat.name][1]].long()
        ) for feat in self.sparse_feature_columns]

        dense_inputs_list = [inputs[:, self.feature_pos[feat.name][0]: self.feature_pos[feat.name][1]]
                             for feat in self.dense_feature_columns]

        dnn_inputs = combine_inputs(embedding_inputs_list, dense_inputs_list)
        output = self.dnn(dnn_inputs)
        return output


class ItemModel(nn.Module):
    def __init__(self, item_feature_columns, hidden_units, embeddings_initializer_std=0.0001):
        super(ItemModel, self).__init__()
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), item_feature_columns)) if len(item_feature_columns) else []

        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), item_feature_columns)) if len(item_feature_columns) else []

        self.embedding_layers = nn.ModuleDict({feat.embedding_name: nn.Embedding(num_embeddings=feat.feature_size,
                                                                                 embedding_dim=feat.embedding_dim)
                                               for i, feat in enumerate(self.sparse_feature_columns)})
        # initialize the embedding weight
        for tensor in self.embedding_layers.values():
            nn.init.normal_(tensor.weight, mean=0.0, std=embeddings_initializer_std)

        # compute input dimension
        inputs_dim = sum([feat.embedding_dim for i, feat in enumerate(self.sparse_feature_columns)]) + \
                     sum([feat.dimension for i, feat in enumerate(self.dense_feature_columns)])

        # compute feature position
        self.feature_pos = build_input_feature_position(item_feature_columns)

        self.dnn = DNN(inputs_dim=inputs_dim, hidden_units=hidden_units)

    def forward(self, inputs):
        embedding_inputs_list = [self.embedding_layers[feat.embedding_name](
            inputs[:, self.feature_pos[feat.name][0]: self.feature_pos[feat.name][1]].long()
        ) for feat in self.sparse_feature_columns]

        dense_inputs_list = [inputs[:, self.feature_pos[feat.name][0]: self.feature_pos[feat.name][1]]
                             for feat in self.dense_feature_columns]

        dnn_inputs = combine_inputs(embedding_inputs_list, dense_inputs_list)

        output = self.dnn(dnn_inputs)
        return output


class DSSM(nn.Module):
    def __init__(self, user_feature_columns, item_feature_columns):
        super(DSSM, self).__init__()
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns

        self.user_model = UserModel(self.user_feature_columns, [64, 32])
        self.item_model = ItemModel(self.item_feature_columns, [64, 32])
        self.similarity_layer = Similarity(type='cos', dim=-1)
        self.output_layer = PredictionLayer()

    def forward(self, inputs):
        user_inputs = inputs[:, :len(self.user_feature_columns)]
        item_inputs = inputs[:, len(self.user_feature_columns):]

        user_out = self.user_model(user_inputs)
        item_out = self.item_model(item_inputs)
        score = self.similarity_layer(user_out, item_out)
        output = self.output_layer(score)
        return output

    def generate_user_embedding(self, user_inputs):
        user_inputs = user_inputs.long()
        return self.user_model(user_inputs)

    def generate_item_embedding(self, item_inputs):
        item_inputs = item_inputs.long()
        return self.item_model(item_inputs)