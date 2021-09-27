import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from constants import saved_model_weights_base_path, saved_model_based_path, ChoosingBetsStrategy, SHOULD_ADD_ODDS_AS_DATA_SEQUENCE
from nn_manager.common import eval_model_after_learning, plot_metric, save_model
from nn_manager.custom_bayesian_tuner import CustomBayesianSearch
from nn_manager.metrics import how_many_no_bets, only_best_prob_odds_profit, profit_and_loss_tuning, all_odds_profit, choose_loss_based_on_strategy, \
    profit_metric_based_on_strategy, choose_bets_precision, only_best_prob_odds_sum_profit, how_many_bets
from nn_manager.neural_network_manager import NeuralNetworkManager
from nn_manager.no_bet_early_stopping import NoBetEarlyStopping


class RecurrentNNChoosingBetsManager(NeuralNetworkManager):
    def __init__(self, train_set, val_set, should_hyper_tune, test_set=None, load_best_weights=False, **kwargs):
        # self.best_params = {'dropout_rate': 0.275,
        #                     'gru_reccurent_regularization_factor': 0.00051439,
        #                     'gru_regularization_factor': 6.144000000000001e-05,
        #                     'learning_rate': 0.0003,
        #                     'number_of_addit_hidden_layers': 2,
        #                     'number_of_gru_units': 16,
        #                     'number_of_neurons_0_layer': 512,
        #                     'number_of_neurons_1_layer': 4,
        #                     'recurrent_type': 'LSTM',
        #                     'regularization_factor': 0.00233147,
        #                     'use_bn_for_input': True,
        #                     'use_bn_for_rest': True,
        #                     'strategy': ChoosingBetsStrategy.AllOnBestResult,
        #                     'should_add_expotential': True
        #                     }
        self.best_params = {'dropout_rate': 0.3,
                            'gru_reccurent_regularization_factor': 1e-8,
                            'gru_regularization_factor': 1e-7,
                            'learning_rate': 0.0005,
                            'number_of_addit_hidden_layers': 3,
                            'number_of_gru_units': 8,
                            'number_of_neurons_0_layer': 32,
                            'number_of_neurons_1_layer': 16,
                            'number_of_neurons_2_layer': 8,
                            'recurrent_type': 'LSTM',
                            'regularization_factor': 3e-8,
                            'use_bn_for_input': True,
                            'use_bn_for_rest': True,
                            'strategy': ChoosingBetsStrategy.AllOnBestResult,
                            'should_add_expotential': True
                            }
        self.best_params.update(kwargs)
        super().__init__(train_set, val_set, should_hyper_tune, test_set, load_best_weights)

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
        dropout_rate = self.best_params["dropout_rate"] if not self.should_hyper_tune else hp.Float('dropout_rate', 0, 0.65, step=0.025)

        recurrent_type = self.best_params["recurrent_type"] if not self.should_hyper_tune else hp.Choice('recurrent_type', ['SimpleRNN', 'GRU', 'LSTM'])
        recurrent_type_callable = getattr(tf.keras.layers, recurrent_type)

        use_bn_for_input = self.best_params["use_bn_for_input"] if not self.should_hyper_tune else hp.Boolean('use_bn_for_input')
        use_bn_for_rest = self.best_params["use_bn_for_rest"] if not self.should_hyper_tune else hp.Boolean('use_bn_for_rest')

        home_input = tf.keras.layers.Input((self.x_train[0].shape[1], self.x_train[0].shape[2],))
        home_rnn = recurrent_type_callable(number_of_gru_units,
                                           kernel_regularizer=l2(gru_regularization_factor),
                                           bias_regularizer=l2(gru_regularization_factor),
                                           recurrent_regularizer=l2(recurrent_regulizer)
                                           # dropout=0.2,
                                           # recurrent_dropout=0.3
                                           )(home_input)

        away_input = tf.keras.layers.Input((self.x_train[1].shape[1], self.x_train[1].shape[2],))
        away_model = recurrent_type_callable(number_of_gru_units,
                                             kernel_regularizer=l2(gru_regularization_factor),
                                             bias_regularizer=l2(gru_regularization_factor),
                                             recurrent_regularizer=l2(recurrent_regulizer)
                                             # dropout=0.2,
                                             # recurrent_dropout=0.3
                                             )(away_input)

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
            quantity_possibilities = [4, 6, 8, 16, 32, 64, 128, 256]
            neurons_quantity = self.best_params[f'number_of_neurons_{i}_layer'] if not self.should_hyper_tune else hp.Choice(f'number_of_neurons_{i}_layer',
                                                                                                                             quantity_possibilities,
                                                                                                                             parent_name='number_of_addit_hidden_layers',
                                                                                                                             parent_values=list(
                                                                                                                                 range(i + 1,
                                                                                                                                       max_layers_quantity + 1))
                                                                                                                             )
            if use_bn_for_rest:
                main_hidden = keras.layers.BatchNormalization()(main_hidden)
            main_hidden = keras.layers.Dropout(dropout_rate)(main_hidden)
            if i == n_hidden_layers-1 and SHOULD_ADD_ODDS_AS_DATA_SEQUENCE:
                bets_input = tf.keras.layers.Input((self.x_train[3].shape[1],))
                if use_bn_for_rest and True:
                    bets_bn = keras.layers.BatchNormalization()(bets_input)
                    main_hidden = tf.keras.layers.Concatenate()([
                        main_hidden,
                        bets_bn
                    ])
                else:
                    main_hidden = tf.keras.layers.Concatenate()([
                        main_hidden,
                        bets_input
                    ])
            main_hidden = keras.layers.Dense(neurons_quantity, activation='relu',
                                             kernel_regularizer=l2(factor),
                                             bias_regularizer=l2(factor),
                                             kernel_initializer=tf.keras.initializers.he_normal())(main_hidden)
        main_hidden = keras.layers.Dense(4, activation='softmax')(main_hidden)
        input_arr = [home_input, away_input, rest_of_input]
        if SHOULD_ADD_ODDS_AS_DATA_SEQUENCE:
            input_arr.append(bets_input)
        main_model = keras.models.Model(inputs=input_arr, outputs=main_hidden)
        decayed_lr = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate, decay_steps=231, decay_rate=0.25)
        opt = keras.optimizers.Adam(learning_rate=decayed_lr)
        main_model.compile(loss=choose_loss_based_on_strategy(self.best_params['strategy'], True, self.best_params['should_add_expotential']),
                           optimizer=opt,
                           metrics=[how_many_no_bets, profit_metric_based_on_strategy(self.best_params['strategy']), choose_bets_precision(),
                                    how_many_bets, only_best_prob_odds_sum_profit(True)])
        return main_model

    def perform_model_learning(self, verbose=True):
        self.history = self.model.fit(x=self.x_train, y=self.y_train, epochs=250,
                                      batch_size=128,
                                      verbose=1 if verbose else 0,
                                      shuffle=True,
                                      validation_data=(self.x_val, self.y_val),
                                      validation_batch_size=self.y_val.shape[0],
                                      callbacks=[
                                          EarlyStopping(patience=150, monitor='val_loss', mode='min', verbose=1 if verbose else 0),
                                          # EarlyStopping(patience=500, monitor='val_profit', mode='max', verbose=1 if verbose else 0),
                                          # NoBetEarlyStopping(patience=150),
                                          ModelCheckpoint(self.get_path_for_saving_weights(), save_best_only=True, save_weights_only=True,
                                                          monitor='val_profit', mode='max', verbose=1 if verbose else 0)]
                                      # callbacks=[TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_graph=True)]
                                      # callbacks=[WeightChangeMonitor()]
                                      )

        self.model.load_weights(self.get_path_for_saving_weights())
        # save_model(self.model, self.get_path_for_saving_model())

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
        tuner.search(x=[self.x_train[0], self.x_train[1], self.x_train[2]], y=self.y_train, epochs=1250, batch_size=256, verbose=2,
                     validation_batch_size=self.y_val.shape[0],
                     callbacks=[EarlyStopping(patience=100, monitor='val_loss', mode='min', verbose=1, min_delta=0.001),
                                NoBetEarlyStopping(patience=75)],
                     shuffle=True,
                     validation_data=([self.x_val[0], self.x_val[1], self.x_val[2]], self.y_val))

        self.print_summary_after_tuning(tuner, 10)

        return tuner
