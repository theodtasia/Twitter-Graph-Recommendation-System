from other.utils import dotdict, device

args = dotdict({})

args.extract_stats_based_attr = True

args.topological_attrs_dim = 5
args.use_topological_node_attrs = False
args.rerun_topological_node_attrs = False
args.rerun_tpological_node_attrs_day_limit = None

args.edge_attrs_dim = 3
args.use_edge_attrs = False
args.rerun_edge_attrs = False
args.rerun_edge_attrs_day_limit = None

args.device = device()

