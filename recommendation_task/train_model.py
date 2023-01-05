import torch
from matplotlib import pyplot as plt
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import leaky_relu
from torchmetrics import RetrievalRecall, RetrievalPrecision

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
        self.metrics = [{k : (metric(k=k), []) for k in at_k}
                        for metric in [RetrievalRecall, RetrievalPrecision]]

        # train the same (continuously) model every day = don't reset daily
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

            train_edges, test_edges = self.dataset.get_dataset()

            print('\nDay: ', self.dataset.day)

            self.run_day_training(train_edges)
            self.test_on_next_day_graph(train_edges, test_edges)
        self.plot_results()


    def run_day_training(self, train_edges):

        self.model.to(self.device)
        self.model.train()

        for epoch in range(EPOCHS):

            self.optimizer.zero_grad()

            # update node embeddings (apply gnn) according to existing edges
            z = self.model(self.dataset.graph.x, train_edges)
            # positive (existing) edges scores
            positives = self.model.decode(z, train_edges)

            # undirected edge_index tensor. sample new negative edges per epoch
            negatives = self.dataset.negative_sampling()
            # negative edges (samples) scores
            negatives = self.model.decode(z, negatives)

            link_labels = torch.tensor([1] * len(positives) + [0] * len(negatives),
                                       device=self.device, dtype=torch.float32)

            loss = self.criterion(torch.cat([positives, negatives]),
                                  link_labels)

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

            # update node embeddings (apply gnn) according to existing edges
            z = self.model(x, train_edges)
            # predict/score next day's edges (positives and all negatives)
            scores = self.model.decode(z, test_edges.test_edges)

            # evaluation metrics
            for metric_at_k in self.metrics:
                for k, (metric_func, results_list) in metric_at_k.items():
                    result = metric_func(scores, test_edges.targets, indexes=test_edges.indexes)
                    results_list.append(result)

                    print(f'{metric_func.__class__.__name__}@{k} = {result}')


    def plot_results(self):

        for metric_at_k in self.metrics:
            for k, (metric_func, results_list) in metric_at_k.items():
                days = range(len(results_list))
                plt.plot(days, results_list, label=f'{metric_func.__class__.__name__}@{k}')

            plt.legend(loc='upper left')
            plt.show()

TrainClassificationModel()

