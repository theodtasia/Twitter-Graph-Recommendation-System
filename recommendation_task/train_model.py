import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import relu, tanh
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import leaky_relu
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import RetrievalRecall
from torchmetrics import F1Score

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
at_k = [10, 20]

class TrainClassificationModel:

    def __init__(self):

        set_seed()
        self.device = device()
        self.dataset = Dataset(self.device)
        self.criterion = BCEWithLogitsLoss()
        self.metrics = {k : RetrievalRecall(k=k) for k in at_k}
        self.results = {k : [] for k in at_k}

        self.model = self.recommendation_model()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=LR, weight_decay=WEIGHT_DECAY)
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
            train_edges, test_edges = self.dataset.get_dataset()

            self.run_day_training(train_edges)
            self.test_on_next_day_graph(train_edges, test_edges)
        self.plot_results()


    def run_day_training(self, train_edges):

        self.model.to(self.device)
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
            if epoch % 10 == 0 or epoch == EPOCHS -1:
                print('Epoch', epoch, loss.item())



    def test_on_next_day_graph(self, train_edges, test_edges):
        self.model.eval()
        with torch.no_grad():
            # perform evaluation on cpu to avoid cuda.OutOfMemoryError
            self.model = self.model.to('cpu')
            x = self.dataset.graph.x.to('cpu')
            train_edges = train_edges.to('cpu')

            z = self.model(x, train_edges)
            scores = self.model.decode(z, test_edges.test_edges)

            for k, metric in self.metrics.items():
                result = metric(scores, test_edges.targets, indexes=test_edges.indexes)
                self.results[k].append(result)
                print(f'Recall@{k} = {result}')

                # print(f'F1 Score = {F1Score(scores, test_edges.targets, indexes=test_edges.indexes)}')
                # print(f'RMSE = {MeanSquaredError(scores, test_edges.targets, indexes=test_edges.indexes)}')
                # print(f'MAE = {MeanAbsoluteError(scores, test_edges.targets, indexes=test_edges.indexes)}')

    def plot_results(self):

        for k, results in self.results.items():
            days = range(len(results))
            plt.plot(days, results, label=f'Recall@{k}')
        plt.legend(loc='upper left')
        plt.show()

TrainClassificationModel()

