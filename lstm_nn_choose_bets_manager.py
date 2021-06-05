import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from constants import saved_model_weights_base_path, saved_model_based_path
from nn_manager.common import eval_model_after_learning, plot_metric, save_model
from nn_manager.metrics import profit_wrapped_in_sqrt_loss, how_many_no_bets, only_best_prob_odds_profit
from nn_manager.neural_network_manager import NeuralNetworkManager


class LstmNNChoosingBetsManager(NeuralNetworkManager):
    def __init__(self, train_set, val_set):
        super().__init__(train_set, val_set)

    def create_model(self):
        # test_factor = 1e-9
        factor = 0.01
        gru_regularization_factor = 1e-4
        # factor = test_factor
        test_rate = 0.01
        # rate = test_rate
        rate = 0.5
        gru_dropout_rate = 0.15

        # tf.compat.v1.disable_eager_execution()

        home_input = tf.keras.layers.Input((self.x_train[0].shape[1], self.x_train[0].shape[2],))
        home_rnn = tf.keras.layers.LSTM(16,
                                             kernel_regularizer=l2(gru_regularization_factor),
                                             bias_regularizer=l2(gru_regularization_factor),
                                             recurrent_regularizer=l2(gru_regularization_factor/10),
                                             dropout=gru_dropout_rate,
                                             recurrent_dropout=gru_dropout_rate*2)(home_input)

        away_input = tf.keras.layers.Input((self.x_train[1].shape[1], self.x_train[1].shape[2],))
        away_model = tf.keras.layers.LSTM(16,
                                               kernel_regularizer=l2(gru_regularization_factor),
                                               bias_regularizer=l2(gru_regularization_factor),
                                               recurrent_regularizer=l2(gru_regularization_factor/10),
                                               dropout=gru_dropout_rate,
                                               recurrent_dropout=gru_dropout_rate*2)(away_input)

        rest_of_input = tf.keras.layers.Input((self.x_train[2].shape[1],))
        # main_model = tf.keras.models.Sequential()
        all_merged = tf.keras.layers.Concatenate()([
            home_rnn,
            away_model,
            rest_of_input
        ])
        # main_hidden = keras.layers.Dropout(0.3)(all_merged)
        main_hidden = keras.layers.Dense(64, activation='relu',
                                          kernel_regularizer=l2(factor),
                                          bias_regularizer=l2(factor),
                                          kernel_initializer=tf.keras.initializers.he_normal())(all_merged)
        main_hidden = keras.layers.Dense(4, activation='softmax')(main_hidden)
        main_model = keras.models.Model(inputs=[home_input, away_input, rest_of_input], outputs=main_hidden)

        opt = keras.optimizers.Adam(learning_rate=0.0005)
        main_model.compile(loss=profit_wrapped_in_sqrt_loss,
                           optimizer=opt,
                           metrics=[how_many_no_bets, only_best_prob_odds_profit()])
        return main_model

    def perform_model_learning(self, verbose=True):
        self.history = self.model.fit(x=[self.x_train[0], self.x_train[1], self.x_train[2]], y=self.y_train, epochs=100,
                                      batch_size=128,
                                      verbose=1 if verbose is True else 0,
                                      shuffle=False,
                                      validation_data=([self.x_val[0], self.x_val[1], self.x_val[2]], self.y_val),
                                      # validation_batch_size=125,
                                      callbacks=[
                                          EarlyStopping(patience=50, monitor='val_loss', mode='min', verbose=1 if verbose is True else 0),
                                          ModelCheckpoint(self.get_path_for_saving_weights(), save_best_only=True, save_weights_only=True,
                                                          monitor='val_profit', mode='max', verbose=1 if verbose is True else 0)]
                                      # callbacks=[TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_graph=True)]
                                      # callbacks=[WeightChangeMonitor()]
                                      )

        self.model.load_weights(self.get_path_for_saving_weights())
        # save_model(self.model, self.get_path_for_saving_model())

    def evaluate_model(self):
        print("Treningowy zbior: ")
        eval_model_after_learning(self.y_train[:, 0:4], self.model.predict(self.x_train), self.y_train[:, 4:7])

        print("Walidacyjny zbior: ")
        eval_model_after_learning(self.y_val[:, 0:4], self.model.predict(self.x_val), self.y_val[:, 4:7])

        plot_metric(self.history, 'loss')
        plot_metric(self.history, 'profit')
