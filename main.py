from torch.nn.functional import leaky_relu

from other.handle_files import validate_args, make_save_results_dir
from other.utils import *
from preprocessing.clean_datasets import CleanData
from recommendation_task.train_model import TrainClassificationModel

def set_arguments():
    args = dotdict({})

    # node statistical features
    args.use_stats_based_attrs = True

    # topological node attributes
    args.topological_attrs_dim = 5
    args.use_topological_node_attrs = False
    args.rerun_topological_node_attrs = False
    args.rerun_topological_node_attrs_day_limit = 23

    # edge attributes
    args.edge_attrs_dim = 3
    args.use_edge_attrs = False
    args.rerun_edge_attrs = False
    args.rerun_edge_attrs_day_limit = 13

    # other
    # args.find_test_edges -> set by the validator
    # args.clean_dataset   -> set by the validator
    args.device = device()

    # training parameters
    args.LR = 0.01
    args.WEIGHT_DECAY = 1e-5
    args.HIDDEN_CHANNELS = 16
    args.N_CONV_LAYERS = 1
    args.CONV_TYPE = 'GINConv'
    args.ACT_FUNC = leaky_relu
    args.DECODER_LAYERS = None
    args.EPOCHS = 100
    args.at_k = [10, 20]

    args.file_name = "GNN dim 16 and 3 aggr layers scale binaries"

    return validate_args(args)


def main():
    make_save_results_dir()
    args = set_arguments()
    if args.clean_dataset:
        CleanData()

    TrainClassificationModel(args)



if __name__ == '__main__':
    main()