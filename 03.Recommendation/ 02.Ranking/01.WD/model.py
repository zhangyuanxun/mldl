import torch.nn as nn
import torch


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, use_bn=False,
                 dropout_rate=0., activation='relu'):
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

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)

            if self.use_bn:
                x = self.bn[i](x)

            x = torch.tanh(x)
            x = self.dropout(x)

        return x


class WideDeep(nn.Module):
    def __init__(self, sparse_feature_columns, dense_feature_columns,
                 hidden_units=(256, 128), use_bn=False, dropout_rate=0.):
        super(WideDeep, self).__init__()