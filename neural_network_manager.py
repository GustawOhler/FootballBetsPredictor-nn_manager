from abc import abstractmethod, ABC
from tensorflow import keras
from constants import saved_model_based_path, saved_model_weights_base_path
from nn_manager.common import load_model


class NeuralNetworkManager(ABC):
    def __init__(self, train_set, val_set, should_hyper_tune):
        keras.backend.clear_session()
        self.x_train = train_set[0]
        self.y_train = train_set[1]
        self.x_val = val_set[0]
        self.y_val = val_set[1]
        self.should_hyper_tune = should_hyper_tune
        if not should_hyper_tune:
            self.model = self.create_model()
        self.history = None


    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def perform_model_learning(self, verbose=True):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    @abstractmethod
    def hyper_tune_model(self):
        pass

    def get_path_for_saving_model(self):
        return saved_model_based_path + self.__class__.__name__

    def get_path_for_saving_weights(self):
        return saved_model_weights_base_path + self.__class__.__name__ + '/checkpoint'