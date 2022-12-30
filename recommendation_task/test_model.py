import torch

from utils import set_seed, device
from recommendation_task.train_model import TrainClassificationModel
from recommendation_task.graph_dataset import Dataset
from sklearn.metrics import roc_auc_score
import tqdm


class TestClassificationModel:

    def __init__(self):

        set_seed()
        self.device = device()
        self.dataset = Dataset()
        self.model = TrainClassificationModel()
        self.model.to(self.device)
        # self.test_model()


    def __getTraining(self):
        sampled_data = next(iter(self.dataset))
        preds = []
        ground_truths = []
        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        auc = roc_auc_score(ground_truth, pred)
        print()
        print(f"Validation AUC: {auc:.4f}")

