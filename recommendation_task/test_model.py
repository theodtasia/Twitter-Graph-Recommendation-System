import torch

from utils import set_seed, device
from recommendation_task.train_model import TrainClassificationModel
from recommendation_task.graph_dataset import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


class TestClassificationModel:

    def __init__(self):
        set_seed()
        self.device = device()
        self.dataset = Dataset()
        self.model = TrainClassificationModel()
        self.model.to(self.device)
        # self.test_model()

    def eval(self):
        with torch.no_grad():
            preds = []
            ground_truths = []
            pred = torch.cat(preds, dim=0).cpu().numpy()
            ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
            auc = roc_auc_score(ground_truth, pred)
            f1 = f1_score(ground_truth, pred)
            mae = mean_absolute_error(ground_truth, pred)
            rmse = mean_squared_error(ground_truth, pred)
            print(f"Validation AUC: {auc:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")

    def test(self):
        print("Evaluation:")
        self.model.eval()
        self.eval(self.model, self.dataset)
