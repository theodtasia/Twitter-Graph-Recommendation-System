import pickle
from random import shuffle

import torch
from clean_datasets import clean_data_path, Graph_
from features_extraction import FeaturesExtraction
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader


clean_data_path = f'../{clean_data_path}'
device = 'cuda'

class Dataset:

    def __init__(self):
        self.day = 0

        self.graph = Data(x = self.get_node_attributes(),
                          edge_index = self.get_day_graph_edge_index(self.day))
        self.graph.to(device=device)


    def increment_day(self):
        self.day += 1

    def get_node_attributes(self):
        x = pickle.load(open(f'{clean_data_path}node_attributes', 'rb'))
        x = FeaturesExtraction(x, turn_to_numeric=True).attributes   # dataframe
        x = x.values.tolist()
        x = torch.tensor(x, device=device, dtype=torch.float32)
        return x

    def get_day_graph_edge_index(self, day):
        # get undirected edges of graph_day as tensor
        edges = pickle.load(open(f'{clean_data_path}{Graph_}{day}', 'rb')).edges()
        return self.to_directed_edge_index(edges)

    def to_directed_edge_index(self, edges):
        return torch.tensor(list(edges), dtype=torch.long, device=device).T

    def to_undirected(self, edge_index):
        return torch.cat([edge_index, edge_index[[1, 0]]], dim=1)


    def split_edges(self, msg_pass_ratio=.5):
        transform = RandomLinkSplit(is_undirected=False, num_val= 1 - msg_pass_ratio, num_test=0,
                                    add_negative_train_samples=False, neg_sampling_ratio=0)
        msg_pass, supervision, _ = (split.edge_label_index
                                    for split in transform(self.graph))

        neg = self.negative_sampling(supervision)
        print(f'original\n{self.graph.edge_index}\nmsg\n{msg_pass}\nsupv\n{supervision}\nneg\n{neg}')

        edge_label = torch.cat([
            torch.full((edge_set.shape[1], ), edge_label, dtype=torch.int8, device=device)
            for edge_set, edge_label in [(msg_pass, 0), (supervision, 1), (neg, 2)]
        ])
        print(edge_label)
        self.graph.edge_label_index = torch.cat([msg_pass, supervision, neg], dim=1)
        del self.graph['edge_index']
        self.graph.edge_label = edge_label
        print(self.graph)
        print(self.graph.edge_label_index)


    def negative_sampling(self, supervision):
        nodes = list(set(v.item() for v in self.graph.edge_index.reshape(-1)))
        positive_edges = set((v.item(),u.item()) for v, u in zip(self.graph.edge_index[0], self.graph.edge_index[1]))
        negative_samples = set()
        for i in range(2):
            for v in supervision[i]:
                v = v.item()
                shuffle(nodes)
                for u in nodes:
                    if u != v and (u,v) not in negative_samples \
                    and (v,u) not in positive_edges and (u,v) not in positive_edges:
                        negative_samples.add((v, u))
                        break
        return self.to_directed_edge_index(negative_samples)


    def edge_subsets(self, batch):
        edge_sets = [
            torch.stack([
                batch.edge_label_index[i][((batch.edge_label == label).nonzero(as_tuple=True)[0])]
                for i in range(2)])
            for label in range(3)
        ]
        edge_sets = [self.to_undirected(edge_index) for edge_index in edge_sets]
        return edge_sets[0], edge_sets[1], edge_sets[2]


    def dataloader(self, batch_size=4):
        return LinkNeighborLoader(
            self.graph,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[1] * 2,
            # Use a batch size of _ for sampling training nodes
            batch_size=batch_size,
            edge_label_index=self.graph.edge_label_index,
            edge_label=self.graph.edge_label,
            neg_sampling_ratio=0,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            is_sorted=False
        )



ds = Dataset()
ds.split_edges()
loader = ds.dataloader()

print(len(loader))
batch = next(iter(loader))
print("\nBatch: ")
print(batch)
print(batch.edge_label_index)
print(batch.edge_label)
msg_pass, supervision, neg = ds.edge_subsets(batch)
print(f'msg\n{msg_pass}\nsupv\n{supervision}\nneg\n{neg}')
