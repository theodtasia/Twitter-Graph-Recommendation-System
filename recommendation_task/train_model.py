import torch
from torch import relu, tanh
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import leaky_relu

from recommendation_task.gnn_model import GNN_model
from recommendation_task.graph_dataset import Dataset
from utils import set_seed, device

LR = 0.01
WEIGHT_DECAY = 1e-5
HIDDEN_CHANNELS = 16
N_CONV_LAYERS = 1
CONV_TYPE = 'GINConv'
ACT_FUNC = leaky_relu
DECODER_LAYERS = None
EPOCHS = 100

class TrainClassificationModel:

    def __init__(self):

        set_seed()
        self.device = device()
        self.dataset = Dataset(self.device)
        self.criterion = BCEWithLogitsLoss()
        self.train_model()


    def recommendation_model(self):

        return GNN_model(
            # Πάντα ίδια :
            in_channels = self.dataset.attr_dim,
            hidden_channels = HIDDEN_CHANNELS,

            # Παράμετροι μοντέλου :
            n_conv_layers = N_CONV_LAYERS,
            conv_type= CONV_TYPE,

            act_func=ACT_FUNC,
            decoder_layers=DECODER_LAYERS

        )


# ================================================================================================

    def train_model(self):

        while self.dataset.has_next():
            print('\nDay: ', self.dataset.day)

            self.model = self.recommendation_model()
            self.model.to(self.device)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=LR, weight_decay=WEIGHT_DECAY)

            train_edges, test_edges = self.dataset.get_dataset()

            self.run_day_training(train_edges)


    def run_day_training(self, train_edges):

        self.model.train()

        for epoch in range(EPOCHS):

            self.optimizer.zero_grad()

            z = self.model(self.dataset.graph.x, train_edges)
            supervision_scores = self.model.decode(z, train_edges)
            negative_edges = self.dataset.negative_sampling()
            negative_scores = self.model.decode(z, negative_edges)
            scores = torch.cat([supervision_scores, negative_scores])
            link_labels = torch.tensor([1] * len(supervision_scores) + [0] * len(negative_scores),
                                       device=self.device, dtype=torch.float32)

            loss = self.criterion(scores, link_labels)

            loss.backward()
            self.optimizer.step()
            print('Epoch', epoch, loss.item())



TrainClassificationModel()

