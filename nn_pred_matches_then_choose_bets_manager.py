import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from constants import saved_weights_location
from nn_manager.common import eval_model_after_learning, plot_metric, save_model
from nn_manager.metrics import profit_wrapped_in_sqrt_loss, how_many_no_bets, only_best_prob_odds_profit
from nn_manager.neural_network_manager import NeuralNetworkManager
from nn_manager.nn_pred_matches_manager import NNPredictingMatchesManager
import numpy as np


class NNChoosingBetsThenDevelopingStrategyManager(NeuralNetworkManager):
    def __init__(self, train_set, val_set):
        previous_model = NNPredictingMatchesManager(train_set, val_set)
        previous_model.perform_model_learning()
        keras.backend.clear_session()
        self.model = self.create_model()
        self.y_train = train_set[1]
        self.x_train = np.concatenate((previous_model.model.predict(train_set[0]), 1.0 / self.y_train[:, 4:7], self.y_train[:, 4:7]), axis=1)
        self.y_val = val_set[1]
        self.x_val = np.concatenate((previous_model.model.predict(val_set[0]), 1.0 / self.y_val[:, 4:7], self.y_val[:, 4:7]), axis=1)
        self.history = None

    def create_model(self):
        factor = 0.01
        rate = 0.4

        # tf.compat.v1.disable_eager_execution()
        model = tf.keras.models.Sequential()
        model.add(keras.layers.BatchNormalization(momentum=0.85))
        model.add(keras.layers.Dense(16, activation='relu',
                                     # activity_regularizer=l2(factor),
                                     kernel_regularizer=l2(factor),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
        model.add(keras.layers.BatchNormalization(momentum=0.85))
        model.add(keras.layers.Dense(16, activation='relu',
                                     # activity_regularizer=l2(factor),
                                     kernel_regularizer=l2(factor),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
        model.add(keras.layers.BatchNormalization(momentum=0.85))
        model.add(keras.layers.Dense(4, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal()
                                     , kernel_regularizer=l2(factor)
                                     ))
        model.compile(loss=profit_wrapped_in_sqrt_loss,
                      optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                      metrics=[how_many_no_bets, only_best_prob_odds_profit()])
        return model

    def perform_model_learning(self):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=1000, batch_size=256, verbose=1, shuffle=False, validation_data=(self.x_val,
                                                                                                                                          self.y_val),
                                      validation_batch_size=25,
                                      callbacks=[EarlyStopping(patience=100, monitor='val_loss', mode='min', verbose=1),
                                                 ModelCheckpoint(saved_weights_location, save_best_only=True, save_weights_only=True, monitor='val_profit',
                                                                 mode='max', verbose=1)]
                                      # callbacks=[TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_graph=True)]
                                      # callbacks=[WeightChangeMonitor()]
                                      )

        self.model.load_weights(saved_weights_location)
        save_model(self.model)

    def evaluate_model(self):
        print("Treningowy zbior: ")
        eval_model_after_learning(self.y_train[:, 0:4], self.model.predict(self.x_train), self.y_train[:, 4:7])

        print("Walidacyjny zbior: ")
        eval_model_after_learning(self.y_val[:, 0:4], self.model.predict(self.x_val), self.y_val[:, 4:7])

        plot_metric(self.history, 'loss')
        plot_metric(self.history, 'profit')
