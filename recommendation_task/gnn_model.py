import torch
from torch import nn, sigmoid
from torch.nn import Linear
from torch.nn.functional import dropout
from torch_geometric.nn import SAGEConv, GINConv, GATConv, GCNConv
from torch_geometric.nn.models import MLP

class GNN_model(nn.Module):

    def __init__(self, in_channels, hidden_channels, n_conv_layers, conv_type, act_func,
                 edge_attributes_dim=0, decoder_layers=None):
        nn.Module.__init__(self)

        self.use_edge_attributes = edge_attributes_dim > 0

        # conv layers
        self.n_conv_layers = n_conv_layers
        self.convs = nn.ModuleList()

        for _ in range(self.n_conv_layers):
            conv = self.ConvLayer(conv_type, in_channels, hidden_channels, act_func)
            self.convs.append(conv)

        if decoder_layers is not None:
            self.decoder = nn.ModuleList()
            edge_attributes_dim += hidden_channels
            self.decoder.append(MLP(in_channels= edge_attributes_dim,
                                    hidden_channels=edge_attributes_dim,
                                    out_channels=edge_attributes_dim,
                                    num_layers=decoder_layers - 1,
                                    act=act_func.__name__, dropout=0.2))

            self.decoder.append(Linear(in_features=edge_attributes_dim,
                                       out_features=1))
            self.decode = self.multilayered_decoder

        else:
            self.decode = self.dot_prod_decode

        self.act_func = act_func


    def ConvLayer(self, conv_type, in_channels, hidden_channels, act_func):

        if conv_type == 'SAGEConv':
            return SAGEConv(in_channels= in_channels, out_channels=hidden_channels, normalize=False)
        elif conv_type == 'GCNConv':
            return GCNConv(in_channels= in_channels, out_channels=hidden_channels, normalize=True)
        elif conv_type == 'GINConv':
            return GINConv(MLP(in_channels= in_channels, hidden_channels=hidden_channels, out_channels=hidden_channels,
                               num_layers=5, act=act_func.__name__, dropout=0.2))
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


    def dot_prod_decode(self, z, edge_index, edge_attributes=None):
        # dot product
        out = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return out

    def multilayered_decoder(self, z, edge_index, edge_attributes):

        out = z[edge_index[0]] * z[edge_index[1]]
        if self.use_edge_attributes:
            out = torch.cat([out, edge_attributes], dim=-1)

        for i in range(len(self.decoder)):
            out = self.decoder[i](out)
        return out.view(-1)


