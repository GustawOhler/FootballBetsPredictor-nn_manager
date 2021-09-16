import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from constants import saved_model_weights_base_path, saved_model_based_path, PredMatchesStrategy
from nn_manager.common import eval_model_after_learning, plot_metric, save_model, eval_model_after_learning_within_threshold, plot_many_metrics
from nn_manager.custom_bayesian_tuner import CustomBayesianSearch
from nn_manager.metrics import categorical_crossentropy_with_bets, categorical_acc_with_bets, odds_profit_with_biggest_gap_over_threshold, \
    get_all_profit_metrics_for_pred_matches
from nn_manager.neural_network_manager import NeuralNetworkManager


class RecurrentNNPredictingMatchesManager(NeuralNetworkManager):
    def __init__(self, train_set, val_set, should_hyper_tune, test_set, **kwargs):
        self.best_params = {
            'confidence_threshold': 0.05,
            'dropout_rate': 0.5,
            'gru_reccurent_regularization_factor': 0.00003,
            'gru_regularization_factor': 0.0001,
            'learning_rate': 0.0025,
            'number_of_addit_hidden_layers': 2,
            'number_of_gru_units': 8,
            'number_of_neurons_0_layer': 64,
            'number_of_neurons_1_layer': 32,
            'recurrent_type': 'GRU',
            'regularization_factor': 0.00002,
            'use_bn_for_input': True,
            'use_bn_for_rest': True
        }
        self.best_params.update(kwargs)
        super().__init__(train_set, val_set, should_hyper_tune, test_set)

    def create_model(self, hp: kt.HyperParameters = None):
        factor = self.best_params["regularization_factor"] if not self.should_hyper_tune else hp.Float('regularization_factor', 0, 1e-2, step=1e-8)
        gru_regularization_factor = self.best_params["gru_regularization_factor"] if not self.should_hyper_tune else \
            hp.Float('gru_regularization_factor', 0, 1e-2, step=1e-8)
        recurrent_regulizer = self.best_params["gru_reccurent_regularization_factor"] if not self.should_hyper_tune else \
            hp.Float('gru_reccurent_regularization_factor', 0, 1e-3, step=1e-8)
        number_of_gru_units = self.best_params["number_of_gru_units"] if not self.should_hyper_tune else hp.Choice('number_of_gru_units', [1, 2, 4, 8, 16])
        first_hidden_units = self.best_params["number_of_neurons_0_layer"] if not self.should_hyper_tune else hp.Choice(
            'number_of_neurons_0_layer', [8, 16, 32, 64, 128, 256, 512])
        max_layers_quantity = 4
        n_hidden_layers = self.best_params["number_of_addit_hidden_layers"] if not self.should_hyper_tune else hp.Int('number_of_addit_hidden_layers', 1,
                                                                                                                      max_layers_quantity)
        learning_rate = self.best_params["learning_rate"] if not self.should_hyper_tune else hp.Float('learning_rate', 1e-5, 3e-3, step=1e-5)
        confidence_threshold = self.best_params["confidence_threshold"] if not self.should_hyper_tune else hp.Float('confidence_threshold', 0.005, 0.15,
                                                                                                                    step=0.005)

        dropout_rate = self.best_params["dropout_rate"] if not self.should_hyper_tune else hp.Float('dropout_rate', 0, 0.65, step=0.025)

        recurrent_type = 'SimpleRNN' if not self.should_hyper_tune else hp.Choice('recurrent_type', ['SimpleRNN', 'GRU', 'LSTM'])
        recurrent_type_callable = getattr(tf.keras.layers, recurrent_type)

        use_bn_for_input = self.best_params["use_bn_for_input"] if not self.should_hyper_tune else hp.Boolean('use_bn_for_input')
        use_bn_for_rest = self.best_params["use_bn_for_rest"] if not self.should_hyper_tune else hp.Boolean('use_bn_for_rest')

        home_input = tf.keras.layers.Input((self.x_train[0].shape[1], self.x_train[0].shape[2],))
        home_rnn = recurrent_type_callable(number_of_gru_units,
                                           kernel_regularizer=l2(gru_regularization_factor),
                                           bias_regularizer=l2(gru_regularization_factor),
                                           recurrent_regularizer=l2(recurrent_regulizer))(home_input)

        away_input = tf.keras.layers.Input((self.x_train[1].shape[1], self.x_train[1].shape[2],))
        away_model = recurrent_type_callable(number_of_gru_units,
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
        if use_bn_for_input:
            main_hidden = keras.layers.BatchNormalization()(all_merged)
            main_hidden = keras.layers.Dropout(dropout_rate)(main_hidden)
        else:
            main_hidden = keras.layers.Dropout(dropout_rate)(all_merged)
        main_hidden = keras.layers.Dense(first_hidden_units, activation='relu',
                                         kernel_regularizer=l2(factor),
                                         bias_regularizer=l2(factor),
                                         kernel_initializer=tf.keras.initializers.he_normal())(main_hidden)
        for i in range(1, n_hidden_layers):
            neurons_quantity = self.best_params[f"number_of_neurons_{i}_layer"] if not self.should_hyper_tune else hp.Choice(f'number_of_neurons_{i}_layer',
                                                                                                                             [4, 6, 8, 16, 32, 64, 128, 256],
                                                                                                                             parent_name='number_of_addit_hidden_layers',
                                                                                                                             parent_values=list(
                                                                                                                                 range(i + 1,
                                                                                                                                       max_layers_quantity + 1))
                                                                                                                             )
            if use_bn_for_rest:
                main_hidden = keras.layers.BatchNormalization()(main_hidden)
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
                           # get_all_profit_metrics_for_pred_matches(confidence_threshold)
                           metrics=[categorical_acc_with_bets, odds_profit_with_biggest_gap_over_threshold(confidence_threshold)])
        return main_model

    def perform_model_learning(self, verbose=True):
        self.history = self.model.fit(x=[self.x_train[0], self.x_train[1], self.x_train[2]], y=self.y_train, epochs=1000,
                                      batch_size=256,
                                      verbose=1 if verbose is True else 0,
                                      shuffle=False,
                                      validation_data=([self.x_val[0], self.x_val[1], self.x_val[2]], self.y_val),
                                      validation_batch_size=self.y_val.shape[0],
                                      callbacks=[
                                          EarlyStopping(patience=100, monitor='val_loss', mode='min',
                                                        verbose=1 if verbose is True else 0
                                                        # , min_delta=0.001
                                                        ),
                                          ModelCheckpoint(self.get_path_for_saving_weights(), save_best_only=True, save_weights_only=True,
                                                          monitor='val_profit', mode='max', verbose=1 if verbose is True else 0)]
                                      # callbacks=[TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_graph=True)]
                                      # callbacks=[WeightChangeMonitor()]
                                      )

        self.model.load_weights(self.get_path_for_saving_weights())
        # save_model(self.model, self.get_path_for_saving_model())

    def get_best_metric_value(self, metric_name):
        return max(self.history.history[metric_name])

    def get_best_strategies_value(self):
        strategy_metrics = {}
        for strategy in PredMatchesStrategy:
            metric_name = f"val_{strategy.value}"
            strategy_metrics.update({strategy: self.get_best_metric_value(metric_name)})
            # print(f'{strategy.value}: {self.get_best_metric_value(metric_name)}')
        # plot_many_metrics(self.history, [en.value for en in PredMatchesStrategy], True, True)
        return strategy_metrics

    def hyper_tune_model(self):
        tuner = CustomBayesianSearch(self.create_model,
                                     objective=kt.Objective('val_profit', 'max'),
                                     max_trials=300,
                                     executions_per_trial=5,
                                     num_initial_points=150,
                                     directory='.\\hypertuning',
                                     project_name=self.__class__.__name__,
                                     overwrite=False,
                                     beta=3.0)
        tuner.search(x=[self.x_train[0], self.x_train[1], self.x_train[2]], y=self.y_train, epochs=1000, batch_size=256,
                     validation_batch_size=self.y_val.shape[0], shuffle=True,
                     verbose=2, callbacks=[EarlyStopping(patience=100, monitor='val_loss', mode='min', verbose=1, min_delta=0.0005)],
                     validation_data=([self.x_val[0], self.x_val[1], self.x_val[2]], self.y_val))

        self.print_summary_after_tuning(tuner, 10)

        return tuner

    def evaluate_model(self, should_plot=True, should_print_train=True, hyperparams=None):
        self.evaluate_model_with_threshold(should_plot, should_print_train, hyperparams)
        # print("Treningowy zbior: ")
        # eval_model_after_learning_within_threshold(self.y_train[:, 0:3], self.model.predict(self.x_train), self.y_train[:, 4:7],
        #                                            self.best_params["confidence_threshold"])
        #
        # print("Walidacyjny zbior: ")
        # eval_model_after_learning_within_threshold(self.y_val[:, 0:3], self.model.predict(self.x_val), self.y_val[:, 4:7],
        #                                            self.best_params["confidence_threshold"])
        #
        # if self.x_test is not None:
        #     print("Testowy zbior: ")
        #     # pprint.pprint(self.model.evaluate(self.x_test, self.y_test, verbose=0, batch_size=16, return_dict=True), width=1)
        #     eval_model_after_learning_within_threshold(self.y_test[:, 0:3], self.model.predict(self.x_test), self.y_test[:, 4:7],
        #                                                self.best_params["confidence_threshold"])
        #
        # plot_metric(self.history, 'loss')
        # plot_metric(self.history, 'categorical_acc_with_bets')
        # plot_metric(self.history, 'profit')
