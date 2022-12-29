from torch import nn
from torch.nn.functional import dropout
from torch_geometric.nn import SAGEConv, GINConv, GATConv, GCNConv
from torch_geometric.nn.models import MLP


class GNN_model(nn.Module):

    def __init__(self, in_channels, hidden_channels, n_conv_layers, conv_type, act_func):
        nn.Module.__init__(self)

        # conv layers
        self.n_conv_layers = n_conv_layers
        self.convs = nn.ModuleList()

        for _ in range(self.n_conv_layers):
            conv = self.ConvLayer(conv_type, in_channels, hidden_channels, act_func)
            self.convs.append(conv)

        self.act_func = act_func


    def ConvLayer(self, conv_type, in_channels, hidden_channels, act_func):

        if conv_type == 'SAGEConv':
            return SAGEConv(in_channels= in_channels, out_channels=hidden_channels, normalize=False)
        elif conv_type == 'GCNConv':
            return GCNConv(in_channels= in_channels, out_channels=hidden_channels, normalize=True)
        elif conv_type == 'GINConv':
            return GINConv(MLP(in_channels= in_channels, hidden_channels=hidden_channels, out_channels=hidden_channels,
                               num_layers=3, act=act_func.__name__, dropout=0.4))
        elif conv_type == 'GATConv':
            return GATConv(in_channels= in_channels, out_channels=hidden_channels, dropout=0.1)


    def forward(self, x, msg_pass_edge_index):

        for i in range(self.n_conv_layers):
            # gnn operation
            x = self.convs[i](x, msg_pass_edge_index)
            # dropout
            x = dropout(x, p=0.2, training=self.training)
            # activation func
            if i < self.n_conv_layers - 1:
                x = self.act_func(x)

            return x

    def decode(self, z, edge_index):
        # dot product
        out = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        return out


