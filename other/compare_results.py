import glob
import json

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from preprocessing.clean_datasets import CleanData

from other.handle_files import RESULTS_DIR

class CompareResults:

    def __init__(self):
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        self.results = self.read_results_files()

        self.metrics = [
            element for element in next(iter(self.results.values()))
            if element.startswith('Retrieval') or element.startswith('Precision')
        ]
        axis = [(i, j) for i in range(len(self.metrics) // 2) for j in range(len(self.metrics) // 2)]
        self.metric_axis = {metric : a for metric, a in zip(self.metrics, axis)}

        self.compare_plot_metric_results('RetrievalRecall@20')
        # self.compare_plot_metric_results('RetrievalPrecision@10')
        self.print_describe()
        # self.make_subplots()
        self.plot_dynamic_graph()

    def plot_dynamic_graph(self):
        print(self.results.keys())
        selected_model = self.results['../results//to_compare\\BASE.json']
        graphs = CleanData.loadDayGraphs()
        merged = nx.Graph()
        nodes, edges = [], []
        for graph in graphs:
            merged = nx.compose(merged, graph)
            nodes.append(len(merged.nodes()))
            edges.append(len(merged.edges()))
        x = range(len(nodes))
        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, figsize =(7, 10))
        ax1.plot(x, nodes)
        ax1.set_ylabel('# nodes')

        # ax2.plot(x, edges)

        for metric in self.metrics:
            ax = ax2 if 'Recall' in metric else ax3
            ax.plot(x[:-1], selected_model[metric], label=metric[-3:])
            ax.legend(loc='upper left')
            ax.set_ylabel(metric[9:-3])
        plt.savefig('base.png', dpi=300)
        plt.show()


        """gs_kw = dict(width_ratios=[1, 1.5], height_ratios=[1, 1])
        fig, axd = plt.subplot_mosaic([['left', 'upper right'],
                                       ['left', 'lower right']],
                                      gridspec_kw=gs_kw, figsize=(5.5, 3.5),
                                      layout="constrained")
        axd['left'].plot(x, nodes)
        axd['left'].set_ylabel('# nodes')

        # ax2.plot(x, edges)

        for metric in self.metrics:
            ax = 'upper right' if 'Recall' in metric else 'lower right'
            axd[ax].plot(x[:-1], selected_model[metric], label=metric[-3:])
            axd[ax].legend(loc='upper left')
            axd[ax].set_ylabel(metric[9:-3])
        plt.show()
    """

    def read_results_files(self):
        files = glob.glob(f'../{RESULTS_DIR}/to_compare/*.json')
        results = {}
        for file in files:
            with open(file, 'r') as f:
                results[file] = json.load(f)
        return results

    def compare_plot_metric_results(self, metric):
        plt.figure(figsize=(7, 4))
        models = [2, 3, 1]
        # min_days = min([len(result[self.metrics[0]]) for result in self.results.values()])
        for file, results in zip(models, self.results.values()):
            metric_values = results[metric]
            min_days = len(results[metric])
            plt.plot(range(min_days), metric_values[:min_days], label=file)

        plt.legend(loc='upper right')
        plt.ylabel(metric)
        plt.savefig('encoder_layers.png', dpi=300)
        plt.show()

    def make_subplots(self):
        font = {'size': 6}
        plt.rc('font', **font)
        fig, axs = plt.subplots(2, 2, sharex=True)
        for name in sorted(self.results.keys(), reverse=True):
            model = self.results[name]
            for metric, axis in self.metric_axis.items():
                days = range(len(model[metric]))
                axs[axis[0], axis[1]].plot(days, np.array(model[metric]) * 100)

        for i, (metric, ax) in enumerate(zip(self.metrics, axs.flat)):
            ax.set(title=metric)
            if i == 0:
                ax.legend(['Base', 'Central', 'EdgeCentral'])
        # fig.legend(['name'] * 3)
        plt.savefig('comparison.png', dpi=300)
        plt.show()

    def print_describe(self):

        min_days = min([len(result[self.metrics[0]]) for result in self.results.values()])
        lines = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        output = {line : [] for line in lines}

        for name in sorted(self.results.keys(), reverse=True):
            model = self.results[name]
            df = {
                metric : model[metric][:min_days] for metric in self.metrics
            }
            df = pd.DataFrame(df).describe()
            print(name); print(df)

            for metric in self.metrics:
                for line in lines:
                    output[line].append(str(round(df[metric][line] * 100, 2)))

        for line in lines:
            row = line.replace('%', '\\%') + ' & ' + ' & '.join(output[line]) + ' \\\\'
            print(row, '\n')



# CompareResults()












