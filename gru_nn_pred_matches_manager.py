import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from constants import saved_model_weights_base_path, saved_model_based_path
from nn_manager.common import eval_model_after_learning, plot_metric, save_model, eval_model_after_learning_within_threshold
from nn_manager.metrics import how_many_no_bets, only_best_prob_odds_profit, categorical_crossentropy_with_bets, \
    categorical_acc_with_bets, odds_profit_with_biggest_gap_over_threshold
from nn_manager.neural_network_manager import NeuralNetworkManager


class GruNNPredictingMatchesManager(NeuralNetworkManager):
    def __init__(self, train_set, val_set, should_hyper_tune, test_set):
        self.old_best = {
            "regularization_factor": 72e-4,
            "number_of_gru_units": 8,
            "number_of_hidden_units_input_layer": 8,
            "number_of_addit_hidden_layers": 0,
            "n_of_neurons": [],
            "learning_rate": 0.0001,
            "confidence_threshold": 0.06,
            "gru_regularization_factor": 1e-4,
            "dropout_rate": 0.25
        }
        self.best_params = {
            'confidence_threshold': 0.1,
            'dropout_rate': 0.25,
            'gru_regularization_factor': 0.0007900000000000001,
            'learning_rate': 0.0009999999999999998,
            'number_of_addit_hidden_layers': 0,
            'number_of_gru_units': 16,
            'number_of_hidden_units_input_layer': 64,
            'regularization_factor': 1e-05}
        super().__init__(train_set, val_set, should_hyper_tune, test_set)

    def create_model(self, hp: kt.HyperParameters = None):
        # test_factor = 1e-15
        # factor = test_factor
        # gru_regularization_factor = test_factor
        # gru_dropout_rate = test_factor
        factor = self.best_params["regularization_factor"] if not self.should_hyper_tune else hp.Float('regularization_factor', 1e-5, 1e-2, step=1e-5)
        gru_regularization_factor = self.best_params["gru_regularization_factor"] if not self.should_hyper_tune else \
            hp.Float('gru_regularization_factor', 1e-5, 1e-3, step=1e-5)
        number_of_gru_units = self.best_params["number_of_gru_units"] if not self.should_hyper_tune else hp.Choice('number_of_gru_units', [1, 2, 4, 8, 16])
        first_hidden_units = self.best_params["number_of_hidden_units_input_layer"] if not self.should_hyper_tune else hp.Choice(
            'number_of_hidden_units_input_layer', [2, 4, 6, 8, 16, 32, 64])
        n_hidden_layers = self.best_params["number_of_addit_hidden_layers"] if not self.should_hyper_tune else hp.Int('number_of_addit_hidden_layers', 0, 3)
        learning_rate = self.best_params["learning_rate"] if not self.should_hyper_tune else hp.Float('learning_rate', 1e-6, 1e-3, step=1e-6)
        confidence_threshold = self.best_params["confidence_threshold"] if not self.should_hyper_tune else hp.Float('confidence_threshold', 0.005, 0.1,
                                                                                                                    step=0.005)
        gru_dropout_rate = 0
        dropout_rate = self.best_params["dropout_rate"] if not self.should_hyper_tune else hp.Float('dropout_rate', 0, 0.65, step=0.025)

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
        main_hidden = keras.layers.Dense(3, activation='softmax')(main_hidden)
        main_model = keras.models.Model(inputs=[home_input, away_input, rest_of_input], outputs=main_hidden)

        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        main_model.compile(loss=categorical_crossentropy_with_bets,
                           optimizer=opt,
                           metrics=[categorical_acc_with_bets, odds_profit_with_biggest_gap_over_threshold(confidence_threshold)])
        return main_model

    def perform_model_learning(self, verbose=True):
        self.history = self.model.fit(x=[self.x_train[0], self.x_train[1], self.x_train[2]], y=self.y_train, epochs=600,
                                      batch_size=128,
                                      verbose=1 if verbose is True else 0,
                                      shuffle=False,
                                      validation_data=([self.x_val[0], self.x_val[1], self.x_val[2]], self.y_val),
                                      validation_batch_size=16,
                                      callbacks=[
                                          EarlyStopping(patience=100, monitor='val_loss', mode='min', verbose=1 if verbose is True else 0),
                                          ModelCheckpoint(self.get_path_for_saving_weights(), save_best_only=True, save_weights_only=True,
                                                          monitor='val_profit', mode='max', verbose=1 if verbose is True else 0)]
                                      # callbacks=[TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_graph=True)]
                                      # callbacks=[WeightChangeMonitor()]
                                      )

        self.model.load_weights(self.get_path_for_saving_weights())
        # save_model(self.model, self.get_path_for_saving_model())

    def hyper_tune_model(self):
        # tuner = kt.RandomSearch(self.create_model,
        #                                 objective=kt.Objective('val_profit', 'max'),
        #                                 max_trials=50,
        #                                 executions_per_trial=5,
        #                                 directory='.\\hypertuning',
        #                                 project_name=self.__class__.__name__,
        #                                 overwrite=False)
        tuner = kt.BayesianOptimization(self.create_model,
                                        objective=kt.Objective('val_profit', 'max'),
                                        max_trials=10,
                                        executions_per_trial=1,
                                        num_initial_points=5,
                                        directory='.\\hypertuning',
                                        project_name='test',
                                        overwrite=True)

        tuner.search(x=[self.x_train[0], self.x_train[1], self.x_train[2]], y=self.y_train, epochs=600, batch_size=128, verbose=2, validation_batch_size=16,
                     callbacks=[EarlyStopping(patience=75, monitor='val_loss', mode='min', verbose=1)],
                     validation_data=([self.x_val[0], self.x_val[1], self.x_val[2]], self.y_val))

        self.print_summary_after_tuning(tuner, 10)

        return tuner

    def evaluate_model(self, should_plot=True, should_print_train=True, hyperparams=None):
        self.evaluate_model_with_threshold(should_plot, should_print_train, hyperparams)
        # print("Treningowy zbior: ")
        # eval_model_after_learning_within_threshold(self.y_train[:, 0:3], self.model.predict(self.x_train), self.y_train[:, 4:7])
        #
        # print("Walidacyjny zbior: ")
        # eval_model_after_learning_within_threshold(self.y_val[:, 0:3], self.model.predict(self.x_val), self.y_val[:, 4:7])
        #
        # plot_metric(self.history, 'loss')
        # plot_metric(self.history, 'categorical_acc_with_bets')
        # plot_metric(self.history, 'profit')
