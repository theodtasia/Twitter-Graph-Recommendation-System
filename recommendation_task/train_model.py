import json
import time
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch.nn import BCEWithLogitsLoss
from torchmetrics import RetrievalRecall, RetrievalPrecision

from other.handle_files import RESULTS_DIR
from other.utils import set_seed
from recommendation_task.gnn_model import GNN_model
from recommendation_task.graph_dataset import Dataset


class TrainClassificationModel:

    def __init__(self, args):

        set_seed()
        self.args = args
        self.dataset = Dataset(self.args)
        self.criterion = BCEWithLogitsLoss()
        self.metrics = [{k : (metric(k=k), []) for k in self.args.at_k}
                        for metric in [RetrievalRecall, RetrievalPrecision]]
        self.runtime_per_day = []

        # train the same (continuously) model every day = don't reset daily
        self.model = self.recommendation_model()
        self.model.to(self.args.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.args.LR, weight_decay=self.args.WEIGHT_DECAY)
        self.train_model()


    def recommendation_model(self):

        return GNN_model(
            # Πάντα ίδια :
            in_channels = self.dataset.attr_dim,
            hidden_channels = self.args.HIDDEN_CHANNELS,

            # Παράμετροι μοντέλου :
            n_conv_layers = self.args.N_CONV_LAYERS,
            conv_type= self.args.CONV_TYPE,

            act_func=self.args.ACT_FUNC,
            decoder_layers=self.args.DECODER_LAYERS,
            edge_attributes_dim = self.args.edge_attrs_dim if self.args.use_edge_attrs else 0
        )


# ================================================================================================

    def train_model(self):

        while self.dataset.has_next():

            train_edges, test_edges = self.dataset.get_dataset()

            print('\nDay: ', self.dataset.day)
            start = time.time()

            self.run_day_training(train_edges)
            self.test_on_next_day_graph(train_edges, test_edges)

            self.runtime_per_day.append(time.time() - start)
        self.save_results()
        self.plot_results()


    def run_day_training(self, train_edges):

        self.model.to(self.args.device)
        self.model.train()

        for epoch in range(self.args.EPOCHS):

            self.optimizer.zero_grad()

            # update node embeddings (apply gnn) according to existing edges
            z = self.model(self.dataset.graph.x, train_edges.edges)
            # positive (existing) edges scores
            positives = self.model.decode(z, train_edges.edges, train_edges.attributes)

            # negatives.edges : undirected edge_index tensor. sample new negative edges per epoch
            # negatives.attributes : tensor (# negatives, edge attributes dim)
            negatives = self.dataset.negative_sampling()
            # negative edges (samples) scores
            negatives = self.model.decode(z, negatives.edges, negatives.attributes)

            link_labels = torch.tensor([1] * len(positives) + [0] * len(negatives),
                                       device=self.args.device, dtype=torch.float32)

            loss = self.criterion(torch.cat([positives, negatives]),
                                  link_labels)

            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0 or epoch == self.args.EPOCHS -1:
                print('Epoch', epoch, loss.item())



    def test_on_next_day_graph(self, train_edges, test_edges):
        self.model.eval()
        with torch.no_grad():
            # perform evaluation on cpu to avoid cuda.OutOfMemoryError
            self.model = self.model.to('cpu')
            x = self.dataset.graph.x.to('cpu')
            train_edges = train_edges.edges.to('cpu')

            # update node embeddings (apply gnn) according to existing edges
            z = self.model(x, train_edges)
            # predict/score next day's edges (positives and all negatives)
            scores = self.model.decode(z, test_edges.edges, test_edges.attributes)

            # evaluation metrics
            for metric_at_k in self.metrics:
                for k, (metric_func, results_list) in metric_at_k.items():
                    result = metric_func(scores, test_edges.targets, indexes=test_edges.indexes)
                    results_list.append(result)

                    print(f'{metric_func.__class__.__name__}@{k} = {result}')


    def save_results(self):
        for metric_at_k in self.metrics:
            for k, (metric_func, results_list) in metric_at_k.items():
                label = f'{metric_func.__class__.__name__}@{k}'
                self.args[label] = [el.item() for el in results_list]
        self.args.runtime_per_day = self.runtime_per_day


        self.args.device, self.args.ACT_FUNC = "", str(self.args.ACT_FUNC)
        with open(f'{self.args.file_name}.json', 'w') as json_file:
            json.dump(self.args, json_file, indent=4)

    def plot_results(self):

        for metric_at_k in self.metrics:
            for k, (metric_func, results_list) in metric_at_k.items():
                days = range(len(results_list))
                label = f'{metric_func.__class__.__name__}@{k}'
                plt.plot(days, results_list, label=label)

            plt.legend(loc='upper left')
            plt.show()
