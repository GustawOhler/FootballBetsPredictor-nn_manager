import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from constants import saved_model_weights_base_path, saved_model_based_path
from nn_manager.common import eval_model_after_learning, plot_metric, save_model
from nn_manager.metrics import profit_wrapped_in_sqrt_loss, how_many_no_bets, only_best_prob_odds_profit
from nn_manager.neural_network_manager import NeuralNetworkManager


class NNChoosingBetsManager(NeuralNetworkManager):
    def __init__(self, train_set, val_set):
        super().__init__(train_set, val_set)

    def create_model(self):
        test_factor = 1e-9
        # factor = 0.000001
        factor = test_factor
        test_rate = 0.01
        # rate = test_rate
        rate = 0.5

        # tf.compat.v1.disable_eager_execution()
        model = tf.keras.models.Sequential()
        model.add(keras.layers.BatchNormalization(momentum=0.99))
        model.add(keras.layers.Dropout(rate))
        model.add(keras.layers.Dense(128, activation='relu',
                                     # activity_regularizer=l2(factor),
                                     kernel_regularizer=l2(factor),
                                     bias_regularizer=l2(factor),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
        model.add(keras.layers.Dropout(rate))
        model.add(keras.layers.Dense(64, activation='relu',
                                     # activity_regularizer=l2(factor / 2),
                                     kernel_regularizer=l2(factor),
                                     bias_regularizer=l2(factor),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
        model.add(keras.layers.Dropout(rate))
        model.add(keras.layers.Dense(64, activation='relu',
                                     # activity_regularizer=l2(factor / 2),
                                     kernel_regularizer=l2(factor),
                                     bias_regularizer=l2(factor),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
        model.add(keras.layers.Dropout(rate))
        model.add(keras.layers.Dense(32, activation='relu',
                                     # activity_regularizer=l2(factor / 2),
                                     kernel_regularizer=l2(factor),
                                     bias_regularizer=l2(factor),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
        model.add(keras.layers.Dropout(rate))
        model.add(keras.layers.Dense(16, activation='relu',
                                     # activity_regularizer=l2(factor / 2),
                                     kernel_regularizer=l2(factor),
                                     bias_regularizer=l2(factor),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
        # model.add(keras.layers.BatchNormalization(momentum=0.99))
        model.add(keras.layers.Dense(4, activation='softmax',
                                     kernel_regularizer=l2(factor),
                                     bias_regularizer=l2(factor)
                                     ))
        opt = keras.optimizers.Adam(learning_rate=0.0003)
        model.compile(loss=profit_wrapped_in_sqrt_loss,
                      optimizer=opt,
                      metrics=[how_many_no_bets, only_best_prob_odds_profit()])
        return model

    def perform_model_learning(self):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=1000, batch_size=128, verbose=1, shuffle=False, validation_data=(self.x_val,
                                                                                                                                          self.y_val),
                                      validation_batch_size=25,
                                      callbacks=[
                                          EarlyStopping(patience=200, monitor='val_loss', mode='min', verbose=1),
                                                 ModelCheckpoint(self.get_path_for_saving_weights(), save_best_only=True, save_weights_only=True,
                                                                 monitor='val_profit', mode='max', verbose=1)]
                                      # callbacks=[TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_graph=True)]
                                      # callbacks=[WeightChangeMonitor()]
                                      )

        self.model.load_weights(self.get_path_for_saving_weights())
        save_model(self.model, self.get_path_for_saving_model())

    def evaluate_model(self):
        print("Treningowy zbior: ")
        eval_model_after_learning(self.y_train[:, 0:4], self.model.predict(self.x_train), self.y_train[:, 4:7])

        print("Walidacyjny zbior: ")
        eval_model_after_learning(self.y_val[:, 0:4], self.model.predict(self.x_val), self.y_val[:, 4:7])

        plot_metric(self.history, 'loss')
        plot_metric(self.history, 'profit')
