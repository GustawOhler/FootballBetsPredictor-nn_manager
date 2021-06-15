import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from constants import saved_model_weights_base_path, saved_model_based_path
from nn_manager.common import eval_model_after_learning, plot_metric, save_model
from nn_manager.metrics import profit_wrapped_in_sqrt_loss, how_many_no_bets, only_best_prob_odds_profit
from nn_manager.neural_network_manager import NeuralNetworkManager


class GruNNChoosingBetsManager(NeuralNetworkManager):
    def __init__(self, train_set, val_set, should_hyper_tune):
        self.best_params = [8e-4, 5, 28, 0]
        super().__init__(train_set, val_set, should_hyper_tune)

    def create_model(self, hp: kt.HyperParameters = None):
        factor = self.best_params[0] if not self.should_hyper_tune else hp.Float('regularization_factor', 1e-4, 1e-2, step=1e-4)
        gru_regularization_factor = 1e-3
        number_of_gru_units = self.best_params[1] if not self.should_hyper_tune else hp.Choice('number_of_gru_units', [1, 2, 4, 8, 16])
        first_hidden_units = self.best_params[2] if not self.should_hyper_tune else hp.Int('number_of_hidden_units_input_layer', 4, 32, step=4)
        n_hidden_layers = self.best_params[3] if not self.should_hyper_tune else hp.Int('number_of_add_hidden_layers', 0, 2)

        home_input = tf.keras.layers.Input((self.x_train[0].shape[1], self.x_train[0].shape[2],))
        home_rnn = tf.keras.layers.GRU(number_of_gru_units,
                                       kernel_regularizer=l2(gru_regularization_factor),
                                       bias_regularizer=l2(gru_regularization_factor),
                                       recurrent_regularizer=l2(gru_regularization_factor / 3))(home_input)

        away_input = tf.keras.layers.Input((self.x_train[1].shape[1], self.x_train[1].shape[2],))
        away_model = tf.keras.layers.GRU(number_of_gru_units,
                                         kernel_regularizer=l2(gru_regularization_factor),
                                         bias_regularizer=l2(gru_regularization_factor),
                                         recurrent_regularizer=l2(gru_regularization_factor / 3))(away_input)

        rest_of_input = tf.keras.layers.Input((self.x_train[2].shape[1],))
        all_merged = tf.keras.layers.Concatenate()([
            home_rnn,
            away_model,
            rest_of_input
        ])
        main_hidden = keras.layers.Dense(first_hidden_units, activation='relu',
                                         kernel_regularizer=l2(factor),
                                         bias_regularizer=l2(factor),
                                         kernel_initializer=tf.keras.initializers.he_normal())(all_merged)
        for i in range(n_hidden_layers):
            main_hidden = keras.layers.Dense(hp.Int(f'number_of_hidden_units_{i}_layer', 4, 16, step=4,
                                                    parent_name='number_of_add_hidden_layers',
                                                    parent_values=range(i+1, n_hidden_layers+1)
                                                    ), activation='relu',
                                             kernel_regularizer=l2(factor),
                                             bias_regularizer=l2(factor),
                                             kernel_initializer=tf.keras.initializers.he_normal())(main_hidden)
        main_hidden = keras.layers.Dense(4, activation='softmax')(main_hidden)
        main_model = keras.models.Model(inputs=[home_input, away_input, rest_of_input], outputs=main_hidden)

        opt = keras.optimizers.Adam(learning_rate=0.0001)
        main_model.compile(loss=profit_wrapped_in_sqrt_loss,
                           optimizer=opt,
                           metrics=[how_many_no_bets, only_best_prob_odds_profit()])
        return main_model

    def perform_model_learning(self, verbose=True):
        self.history = self.model.fit(x=[self.x_train[0], self.x_train[1], self.x_train[2]], y=self.y_train, epochs=500,
                                      batch_size=128,
                                      verbose=1 if verbose is True else 0,
                                      shuffle=False,
                                      validation_data=([self.x_val[0], self.x_val[1], self.x_val[2]], self.y_val),
                                      # validation_batch_size=125,
                                      callbacks=[
                                          EarlyStopping(patience=50, monitor='val_profit', mode='max', verbose=1 if verbose is True else 0),
                                          ModelCheckpoint(self.get_path_for_saving_weights(), save_best_only=True, save_weights_only=True,
                                                          monitor='val_profit', mode='max', verbose=1 if verbose is True else 0)]
                                      # callbacks=[TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_graph=True)]
                                      # callbacks=[WeightChangeMonitor()]
                                      )

        self.model.load_weights(self.get_path_for_saving_weights())
        # save_model(self.model, self.get_path_for_saving_model())

    def hyper_tune_model(self):
        # tuner = kt.Hyperband(self.create_model,
        #              objective=kt.Objective('val_profit', 'max'),
        #              max_epochs=99,
        #              factor=3,
        #              hyperband_iterations=1)
        tuner = kt.RandomSearch(self.create_model,
                                objective=kt.Objective('val_profit', 'max'),
                                max_trials=35,
                                executions_per_trial=3,
                                directory='.\\hypertuning',
                                project_name=self.__class__.__name__,
                                overwrite=True)
        tuner.search(x=[self.x_train[0], self.x_train[1], self.x_train[2]], y=self.y_train, epochs=250, batch_size=128,
                     callbacks=[EarlyStopping(patience=60, monitor='val_profit', mode='max', verbose=1)],
                     validation_data=([self.x_val[0], self.x_val[1], self.x_val[2]], self.y_val))
        return tuner

    def evaluate_model(self):
        print("Treningowy zbior: ")
        eval_model_after_learning(self.y_train[:, 0:4], self.model.predict(self.x_train), self.y_train[:, 4:7])

        print("Walidacyjny zbior: ")
        eval_model_after_learning(self.y_val[:, 0:4], self.model.predict(self.x_val), self.y_val[:, 4:7])

        plot_metric(self.history, 'loss')
        plot_metric(self.history, 'profit')
