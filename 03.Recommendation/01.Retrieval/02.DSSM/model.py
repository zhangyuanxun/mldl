import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F


class Similarity(nn.Module):
    def __init__(self, type='cos', dim=0):
        super(Similarity, self).__init__()
        self.type = type
        self.dim = dim
        assert(self.type == 'cos')

    def forward(self, query, key):
        if self.type == 'cos':
            score = torch.cosine_similarity(query, key, -1)
            score = torch.clip(score, 0, 1.0)

        return score


class PredictionLayer(nn.Module):
    def __init__(self):
        super(PredictionLayer, self).__init__()

    def forward(self, inputs):
        x = torch.sigmoid(inputs)
        return x


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, use_bn=True,
                 dropout_rate=0., device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.use_bn = use_bn
        hidden_units = [inputs_dim] + hidden_units

        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)]
        )

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


class UserModel(nn.Module):
    def __init__(self, user_feature_columns, hidden_units):
        super(UserModel, self).__init__()
        self.user_feature_columns = user_feature_columns
        self.embedding_layers = nn.ModuleDict({
            'user_embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.user_feature_columns)
        })
        inputs_dim = sum([feat['embed_dim'] for i, feat in enumerate(self.user_feature_columns)])
        self.dnn = DNN(inputs_dim=inputs_dim, hidden_units=hidden_units)

    def forward(self, inputs):
        embeddings = [self.embedding_layers['user_embed_'+str(i)](inputs[:, i]) for i in range(inputs.shape[1])]
        embeddings = torch.cat(embeddings, axis=-1)
        output = self.dnn(embeddings)
        return output


class ItemModel(nn.Module):
    def __init__(self, item_feature_columns, hidden_units):
        super(ItemModel, self).__init__()
        self.item_feature_columns = item_feature_columns
        self.embedding_layers = nn.ModuleDict({
            'item_embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.item_feature_columns)
        })
        inputs_dim = sum([feat['embed_dim'] for i, feat in enumerate(self.item_feature_columns)])
        self.dnn = DNN(inputs_dim=inputs_dim, hidden_units=hidden_units)

    def forward(self, inputs):
        embeddings = [self.embedding_layers['item_embed_'+str(i)](inputs[:, i]) for i in range(inputs.shape[1])]
        embeddings = torch.cat(embeddings, axis=-1)
        output = self.dnn(embeddings)
        return output

        
class DSSM(nn.Module):
    def __init__(self, user_feature_columns, item_feature_columns):
        super(DSSM, self).__init__()
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns

        self.user_model = UserModel(self.user_feature_columns, [64, 32])
        self.item_model = ItemModel(self.item_feature_columns, [64, 32])
        self.similarity_layer = Similarity(type='cos', dim=1)
        self.output_layer = PredictionLayer()

    def forward(self, inputs):
        user_inputs = inputs[:, :len(self.user_feature_columns)]
        item_inputs = inputs[:, len(self.user_feature_columns):]
        user_inputs = user_inputs.long()
        item_inputs = item_inputs.long()

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



