from utils import set_seed, device
from recommendation_task.train_model import TrainClassificationModel
from recommendation_task.graph_dataset import Dataset

class TestClassificationModel:

    def __init__(self):

        set_seed()
        self.device = device()
        self.dataset = Dataset()
        self.model = TrainClassificationModel()
        self.model.to(self.device)
        self.test_model()



