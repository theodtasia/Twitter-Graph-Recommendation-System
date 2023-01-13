from torch.nn.functional import leaky_relu

from other.handle_files import validate_args
from other.utils import dotdict, device

args = dotdict({})

# node statistical features
args.use_stats_based_attrs = True

# topological node attributes
args.topological_attrs_dim = 5
args.use_topological_node_attrs = False
args.rerun_topological_node_attrs = False
args.rerun_topological_node_attrs_day_limit = 3

# edge attributes
args.edge_attrs_dim = 3
args.use_edge_attrs = True
args.rerun_edge_attrs = False
args.rerun_edge_attrs_day_limit = 2

# necessary
args.find_test_edges = False
args.clean_dataset = False
args.device = device()

# training parameters
args.LR = 0.01
args.WEIGHT_DECAY = 1e-5
args.HIDDEN_CHANNELS = 16
args.N_CONV_LAYERS = 1
args.CONV_TYPE = 'GINConv'
args.ACT_FUNC = leaky_relu
args.DECODER_LAYERS = 2
args.EPOCHS = 100
args.at_k = [10, 20]

args = validate_args(args)





