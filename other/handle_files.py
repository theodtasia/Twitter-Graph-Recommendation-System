import glob
from os import mkdir
from os.path import exists

# Clean Dataset
ORIGINAL_DATA_PATH = '../data/original_data/'
CLEAN_DATA_PATH = '../data/clean_data/'
Graph_ = 'day_graphs/Graph_'

# feature extractor
DAY_NODE_ATTRS_PATH = f'{CLEAN_DATA_PATH}node_attrs_per_day/'

# edge handler
TEST_EDGES_PATH = f'{CLEAN_DATA_PATH}test_edges/'
EDGE_ATTRIBUTES_PATH = f'{CLEAN_DATA_PATH}edge_attributes/'

def get_number_of_files(files):
    return len(glob.glob(files))

def numOfGraphs():
    files = f'{CLEAN_DATA_PATH}{Graph_}*'
    return get_number_of_files(files)

def final_day_with_topological_attrs():
    files = f'{DAY_NODE_ATTRS_PATH}nodeAttrsG_*'
    return get_number_of_files(files)
def final_day_with_edge_attrs():
    files = f'{EDGE_ATTRIBUTES_PATH}edge_attrsG_*'
    return get_number_of_files(files)

def final_day_with_test_edges():
    files = f'{TEST_EDGES_PATH}negativeG_*'
    return get_number_of_files(files)

def feature_saved(dir_name):
    saved = exists(dir_name[:-1])
    if not saved:
        mkdir(dir_name[:-1])
    return saved


def validate_args(args):
    if args.use_topological_node_attrs :
        if not feature_saved(DAY_NODE_ATTRS_PATH) \
        or final_day_with_topological_attrs() < args.rerun_topological_node_attrs_day_limit :
            args.rerun_topological_node_attrs = True

    if args.use_edge_attrs :
        if not feature_saved(EDGE_ATTRIBUTES_PATH) \
        or final_day_with_edge_attrs() < args.rerun_edge_attrs_day_limit :
            args.rerun_edge_attrs = True

    args.find_test_edges = not feature_saved(TEST_EDGES_PATH) \
                           or final_day_with_test_edges() != numOfGraphs()

    args.clean_dataset = not feature_saved(CLEAN_DATA_PATH) \
                         or not feature_saved(CLEAN_DATA_PATH + 'day_graphs/')

    return args