import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from constants import saved_model_weights_base_path, saved_model_based_path, ChoosingBetsStrategy
from nn_manager.common import eval_model_after_learning, plot_metric, save_model
from nn_manager.metrics import how_many_no_bets, only_best_prob_odds_profit, choose_loss_based_on_strategy, profit_metric_based_on_strategy
from nn_manager.neural_network_manager import NeuralNetworkManager
from timeit import default_timer as timer


class LstmNNChoosingBetsManager(NeuralNetworkManager):
    def __init__(self, train_set, val_set, should_hyper_tune, test_set):
        self.best_params = {
            "regularization_factor": 0.0096,
            "number_of_gru_units": 16,
            "number_of_hidden_units_input_layer": 32,
            "number_of_addit_hidden_layers": 0,
            "learning_rate": 0.0003,
            "gru_regularization_factor": 1e-4,
            "dropout_rate": 0,
            'strategy': ChoosingBetsStrategy.AllOnBestResult,
            'should_add_expotential': True
        }
        super().__init__(train_set, val_set, should_hyper_tune, test_set)

    def create_model(self, hp: kt.HyperParameters = None):
        factor = self.best_params["regularization_factor"] if not self.should_hyper_tune else hp.Float('regularization_factor', 1e-5, 1e-2, step=1e-5)
        gru_regularization_factor = self.best_params["gru_regularization_factor"] if not self.should_hyper_tune else \
            hp.Float('gru_regularization_factor', 1e-5, 1e-2, step=1e-5)
        recurrent_regulizer = self.best_params["gru_regularization_factor"] if not self.should_hyper_tune else \
            hp.Float('gru_reccurent_regularization_factor', 1e-5, 1e-3, step=1e-5)
        number_of_gru_units = self.best_params["number_of_gru_units"] if not self.should_hyper_tune else hp.Choice('number_of_gru_units', [1, 2, 4, 8, 16])
        first_hidden_units = self.best_params["number_of_hidden_units_input_layer"] if not self.should_hyper_tune else hp.Choice(
            'number_of_hidden_units_input_layer', [2, 4, 6, 8, 16, 32, 64])
        n_hidden_layers = self.best_params["number_of_addit_hidden_layers"] if not self.should_hyper_tune else hp.Int('number_of_addit_hidden_layers', 0, 3)
        learning_rate = self.best_params["learning_rate"] if not self.should_hyper_tune else hp.Float('learning_rate', 1e-6, 1e-3, step=1e-6)
        dropout_rate = self.best_params["dropout_rate"] if not self.should_hyper_tune else hp.Float('dropout_rate', 0, 0.65, step=0.025)

        home_input = tf.keras.layers.Input((self.x_train[0].shape[1], self.x_train[0].shape[2],))
        home_rnn = tf.keras.layers.LSTM(number_of_gru_units,
                                             kernel_regularizer=l2(gru_regularization_factor),
                                             bias_regularizer=l2(gru_regularization_factor),
                                             recurrent_regularizer=l2(recurrent_regulizer))(home_input)

        away_input = tf.keras.layers.Input((self.x_train[1].shape[1], self.x_train[1].shape[2],))
        away_model = tf.keras.layers.LSTM(number_of_gru_units,
                                               kernel_regularizer=l2(gru_regularization_factor),
                                               bias_regularizer=l2(gru_regularization_factor),
                                               recurrent_regularizer=l2(recurrent_regulizer))(away_input)

        rest_of_input = tf.keras.layers.Input((self.x_train[2].shape[1],))
        # main_model = tf.keras.models.Sequential()
        all_merged = tf.keras.layers.Concatenate()([
            home_rnn,
            away_model,
            rest_of_input
        ])
        main_hidden = keras.layers.Dropout(dropout_rate)(all_merged)
        main_hidden = keras.layers.Dense(first_hidden_units, activation='relu',
                                          kernel_regularizer=l2(factor),
                                          bias_regularizer=l2(factor),
                                          kernel_initializer=tf.keras.initializers.he_normal())(main_hidden)
        for i in range(n_hidden_layers):
            neurons_quantity = self.best_params["n_of_neurons"][i] if not self.should_hyper_tune else hp.Choice(f'number_of_neurons_{i}_layer',
                                                                                                                [2, 4, 6, 8, 16, 32, 64],
                                                                                                                parent_name='number_of_addit_hidden_layers',
                                                                                                                parent_values=list(
                                                                                                                    range(i + 1, n_hidden_layers + 1))
                                                                                                                )
            main_hidden = keras.layers.Dropout(dropout_rate)(main_hidden)
            main_hidden = keras.layers.Dense(neurons_quantity, activation='relu',
                                             kernel_regularizer=l2(factor),
                                             bias_regularizer=l2(factor),
                                             kernel_initializer=tf.keras.initializers.he_normal())(main_hidden)
        main_hidden = keras.layers.Dense(4, activation='softmax')(main_hidden)
        main_model = keras.models.Model(inputs=[home_input, away_input, rest_of_input], outputs=main_hidden)

        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        main_model.compile(loss=choose_loss_based_on_strategy(self.best_params['strategy'], True, self.best_params['should_add_expotential']),
                      optimizer=opt,
                      metrics=[how_many_no_bets, profit_metric_based_on_strategy()])
        return main_model

    def perform_model_learning(self, verbose=True):
        self.history = self.model.fit(x=[self.x_train[0], self.x_train[1], self.x_train[2]], y=self.y_train, epochs=25,
                                      batch_size=512,
                                      validation_batch_size=self.y_train.shape[0],
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

    def hyper_tune_model(self):
        tuner = kt.BayesianOptimization(self.create_model,
                                        objective=kt.Objective('val_profit', 'max'),
                                        max_trials=75,
                                        executions_per_trial=3,
                                        num_initial_points=25,
                                        directory='.\\hypertuning',
                                        project_name=self.__class__.__name__,
                                        overwrite=False)
        tuner.search(x=[self.x_train[0], self.x_train[1], self.x_train[2]], y=self.y_train, epochs=650, batch_size=512, verbose=2,
                     validation_batch_size=self.y_train.shape[0],
                     callbacks=[EarlyStopping(patience=50, monitor='val_loss', mode='min', verbose=1, min_delta=0.0001)],
                     validation_data=([self.x_val[0], self.x_val[1], self.x_val[2]], self.y_val))

        self.print_summary_after_tuning(tuner, 10)

        return tuner

    # def evaluate_model(self):
    #     print("Treningowy zbior: ")
    #     eval_model_after_learning(self.y_train[:, 0:4], self.model.predict(self.x_train), self.y_train[:, 4:7])
    #
    #     print("Walidacyjny zbior: ")
    #     eval_model_after_learning(self.y_val[:, 0:4], self.model.predict(self.x_val), self.y_val[:, 4:7])
    #
    #     plot_metric(self.history, 'loss')
    #     plot_metric(self.history, 'profit')
