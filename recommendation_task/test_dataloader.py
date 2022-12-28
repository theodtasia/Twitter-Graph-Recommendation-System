from torch_geometric.datasets import Planetoid
from torch_geometric.loader import LinkNeighborLoader

data = Planetoid('.', name='Cora')[0]
data.to('cuda')

loader = LinkNeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=128,
    edge_label_index=data.edge_index,
)

sampled_data = next(iter(loader))
print(sampled_data)