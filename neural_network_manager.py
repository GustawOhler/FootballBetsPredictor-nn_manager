# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from abc import ABCMeta, abstractmethod, ABC


class NeuralNetworkManager(ABC):
    def __init__(self, train_set, val_set):
        self.model = self.create_model()
        self.x_train = train_set[0]
        self.y_train = train_set[1]
        self.x_val = val_set[0]
        self.y_val = val_set[1]
        self.history = None

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def perform_model_learning(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

