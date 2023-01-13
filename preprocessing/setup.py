from other.utils import dotdict, device
from preprocessing.edge_handler import EdgeHandler


args = dotdict({})

args.use_stats_based_attr = True

args.topological_attrs_dim = 5
args.use_topological_node_attrs = False
args.rerun_topological_node_attrs = False
args.rerun_topological_node_attrs_day_limit = 3

args.edge_attrs_dim = 3
args.use_edge_attrs = True
args.rerun_edge_attrs = False
args.rerun_edge_attrs_day_limit = 2

args.find_test_edges = False

args.device = device()

test = EdgeHandler(args).loadTestEdges(1)
print(test)
