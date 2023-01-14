import glob
import json

from matplotlib import pyplot as plt

from other.handle_files import RESULTS_DIR

class CompareResults:

    def __init__(self):
        self.results = self.read_results_files()
        self.compare_plot_metric_results('RetrievalRecall@20')
    def read_results_files(self):
        files = glob.glob(f'../{RESULTS_DIR}/to_compare/*.json')
        results = {}
        for file in files:
            with open(file, 'r') as f:
                results[file] = json.load(f)
        return results

    def compare_plot_metric_results(self, metric):
        for file, results in self.results.items():
            metric_values = results[metric]
            days = range(0, len(metric_values))
            plt.plot(days, metric_values, label=file)

        plt.legend(loc='upper left')
        plt.show()


CompareResults()












