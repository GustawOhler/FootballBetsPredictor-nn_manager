import pprint
import sys
from abc import abstractmethod, ABC
from tensorflow import keras
from constants import saved_model_based_path, saved_model_weights_base_path
from nn_manager.common import eval_model_after_learning_within_threshold, plot_metric, eval_model_after_learning


class NeuralNetworkManager(ABC):
    def __init__(self, train_set, val_set, should_hyper_tune, test_set):
        keras.backend.clear_session()
        self.x_train = train_set[0]
        self.y_train = train_set[1]
        self.x_val = val_set[0]
        self.y_val = val_set[1]
        if test_set is not None:
            self.x_test = test_set[0]
            self.y_test = test_set[1]
        else:
            self.x_test = None
            self.y_test = None
        self.should_hyper_tune = should_hyper_tune
        if not should_hyper_tune:
            self.model = self.create_model()
        self.history = None

    @abstractmethod
    def create_model(self, hp=None):
        pass

    @abstractmethod
    def perform_model_learning(self, verbose=True):
        pass

    def evaluate_model(self, should_plot=True, should_print_train=True, hyperparams=None):
        if should_print_train:
            print("Treningowy zbior: ")
            pprint.pprint(self.model.evaluate(self.x_train, self.y_train, verbose=0, batch_size=16, return_dict=True), width=1)
            eval_model_after_learning(self.y_train[:, 0:4], self.model.predict(self.x_train), self.y_train[:, 4:7])

        print("Walidacyjny zbior: ")
        pprint.pprint(self.model.evaluate(self.x_val, self.y_val, verbose=0, batch_size=16, return_dict=True), width=1)
        eval_model_after_learning(self.y_val[:, 0:4], self.model.predict(self.x_val), self.y_val[:, 4:7])

        if self.x_test is not None:
            print("Testowy zbior: ")
            pprint.pprint(self.model.evaluate(self.x_test, self.y_test, verbose=0, batch_size=16, return_dict=True), width=1)
            eval_model_after_learning(self.y_test[:, 0:4], self.model.predict(self.x_test), self.y_test[:, 4:7])

        if should_plot:
            plot_metric(self.history, 'loss')
            plot_metric(self.history, 'profit')

    @abstractmethod
    def hyper_tune_model(self):
        pass

    def get_path_for_saving_model(self):
        return saved_model_based_path + self.__class__.__name__

    def get_path_for_saving_weights(self):
        return saved_model_weights_base_path + self.__class__.__name__ + '/checkpoint'

    def print_summary_after_tuning(self, tuner, how_much_models):
        original_stdout = sys.stdout
        with open(f'./hypertuning/{self.__class__.__name__}/summary.txt', 'w') as summary_file:
            sys.stdout = summary_file
            best_hyparam = tuner.get_best_hyperparameters(how_much_models)
            for i, model in enumerate(tuner.get_best_models(how_much_models)):
                self.model = model
                print(f"Model {i + 1}:")
                pprint.pprint(best_hyparam[i].values, width=1)
                self.evaluate_model(False, False, best_hyparam[i].values)
                print()
            sys.stdout = original_stdout

    def evaluate_model_with_threshold(self, should_plot, should_print_train, hyperparams):
        if hyperparams == None:
            threshold = self.best_params["confidence_threshold"]
        else:
            threshold = hyperparams["confidence_threshold"]
        if should_print_train:
            print("Treningowy zbior: ")
            eval_model_after_learning_within_threshold(self.y_train[:, 0:3], self.model.predict(self.x_train), self.y_train[:, 4:7],
                                                       threshold)

        print("Walidacyjny zbior: ")
        eval_model_after_learning_within_threshold(self.y_val[:, 0:3], self.model.predict(self.x_val), self.y_val[:, 4:7],
                                                   threshold)

        if self.x_test is not None:
            print("Testowy zbior: ")
            # pprint.pprint(self.model.evaluate(self.x_test, self.y_test, verbose=0, batch_size=16, return_dict=True), width=1)
            eval_model_after_learning_within_threshold(self.y_test[:, 0:3], self.model.predict(self.x_test), self.y_test[:, 4:7],
                                                       threshold)

        if should_plot:
            plot_metric(self.history, 'loss')
            plot_metric(self.history, 'categorical_acc_with_bets')
            plot_metric(self.history, 'profit')
