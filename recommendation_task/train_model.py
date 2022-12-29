import torch
from torch import relu, binary_cross_entropy_with_logits, sigmoid

from recommendation_task.gnn_model import GNN_model
from recommendation_task.graph_dataset import Dataset
from utils import set_seed, device

LR = 0.01
WEIGHT_DECAY = 1e-5
HIDDEN_CHANNELS = 6
N_CONV_LAYERS = 1
CONV_TYPE = 'SAGEConv'
ACT_FUNC = relu
EPOCHS = 100

class TrainClassificationModel:

    def __init__(self):

        set_seed()
        self.device = device()

        self.dataset = Dataset()
        self.model = self.recommendation_model()
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        self.train_model()



    def recommendation_model(self):

        return GNN_model(
            # Πάντα ίδια :
            in_channels = self.dataset.attr_dim,
            hidden_channels = HIDDEN_CHANNELS,

            # Παράμετροι μοντέλου :
            n_conv_layers = N_CONV_LAYERS,
            conv_type= CONV_TYPE,

            act_func=ACT_FUNC,

        )


# ================================================================================================

    def train_model(self):

        while self.dataset.has_next():
            print('\nDay: ', self.dataset.day)

            g_train, test_edges = self.dataset.get_dataset_method2()
            if self.dataset.day < 62:
                continue
            self.run_day_training_method2(g_train)


    def run_day_training(self, g_train):

        self.model.train()

        for epoch in range(min(self.dataset.day * 10, EPOCHS)):

            self.optimizer.zero_grad()

            z = self.model(self.dataset.graph.x, g_train.msg_pass)
            supervision_scores = sigmoid(self.model.decode(z, g_train.supervision))
            negative_scores = sigmoid(self.model.decode(z, g_train.negative))

            loss = - torch.log(supervision_scores + 1e-15).mean() - torch.log(1 - negative_scores + 1e-15).mean()

            loss.backward()
            self.optimizer.step()
            print('Epoch', epoch, loss.item())


    def run_day_training_method2(self, g_train):

        self.model.train()

        for epoch in range(min(self.dataset.day * 10, EPOCHS)):

            self.optimizer.zero_grad()

            z = self.model(self.dataset.graph.x, g_train.training)
            supervision_scores = self.model.decode(z, g_train.training)
            negative_scores = self.model.decode(z, g_train.negative)

            link_labels = torch.tensor([1] * len(supervision_scores) + [0] * len(negative_scores), dtype=torch.float32, device=self.device)
            scores = torch.cat([supervision_scores, negative_scores])
            # print(scores.tolist()[0:10], scores.tolist()[-10:-1])

            loss = binary_cross_entropy_with_logits(scores, link_labels).mean()

            loss.backward()
            self.optimizer.step()
            print('Epoch', epoch, loss.item())



TrainClassificationModel()