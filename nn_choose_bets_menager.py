import pprint

import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2

from constants import ChoosingBetsStrategy
from nn_manager.common import eval_model_after_learning, plot_metric, save_model
from nn_manager.custom_bayesian_tuner import CustomBayesianSearch
from nn_manager.metrics import how_many_no_bets, only_best_prob_odds_profit, profit_metric_based_on_strategy, choose_loss_based_on_strategy
from nn_manager.neural_network_manager import NeuralNetworkManager
from nn_manager.no_bet_early_stopping import NoBetEarlyStopping


class NNChoosingBetsManager(NeuralNetworkManager):
    def __init__(self, train_set, val_set, should_hyper_tune, test_set=None, **kwargs):
        self.old_best_params = {
            'dropout_rate': 0.15,
            'input_dropout_rate': 0.35,
            'layers_quantity': 1,
            'learning_rate': 0.00058,
            'n_of_neurons': [128],
            'regularization_factor': 0.0005368587,
            'use_bn_for_input': True,
            'use_bn_for_rest': True
        }
        self.best_params = {
            # 'dataset': 'BasicDatasetCreator',
            'dropout_rate': 0.025,
            'input_dropout_rate': 0.125,
            'layers_quantity': 3,
            'learning_rate': 3.5999999999999994e-05,
            'number_of_neurons_0_layer': 32,
            'number_of_neurons_1_layer': 16,
            'number_of_neurons_2_layer': 16,
            'regularization_factor': 0.0034858521000000003,
            'regularize_output_layer': False,
            'use_bn_for_input': True,
            'use_bn_for_rest': True,
            'strategy': ChoosingBetsStrategy.AllOnBestResult,
            'should_add_expotential': True
        }
        self.best_params.update(kwargs)
        super().__init__(train_set, val_set, should_hyper_tune, test_set)

    def create_model(self, hp: kt.HyperParameters = None):
        factor = self.best_params["regularization_factor"] if not self.should_hyper_tune else hp.Float('regularization_factor', 0, 5e-3, step=1e-10)
        input_dropout_rate = self.best_params["input_dropout_rate"] if not self.should_hyper_tune else hp.Float('input_dropout_rate', 0, 0.65, step=0.025)
        rate = self.best_params["dropout_rate"] if not self.should_hyper_tune else hp.Float('dropout_rate', 0, 0.65, step=0.025)
        max_layers_quantity = 6
        layers_quantity = self.best_params["layers_quantity"] if not self.should_hyper_tune else hp.Int('layers_quantity', 1, max_layers_quantity)
        learning_rate = self.best_params["learning_rate"] if not self.should_hyper_tune else hp.Float('learning_rate', 1e-6, 1e-3, step=1e-6)

        use_bn_for_input = self.best_params["use_bn_for_input"] if not self.should_hyper_tune else hp.Boolean('use_bn_for_input')
        use_bn_for_rest = self.best_params["use_bn_for_rest"] if not self.should_hyper_tune else hp.Boolean('use_bn_for_rest')

        regularize_output_layer = self.best_params["regularize_output_layer"] if not self.should_hyper_tune else hp.Boolean('regularize_output_layer')

        model = tf.keras.models.Sequential()
        if use_bn_for_input:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(input_dropout_rate))
        for i in range(layers_quantity):
            if not self.should_hyper_tune:
                neurons_quantity = self.best_params[f'number_of_neurons_{i}_layer']
            else:
                with hp.conditional_scope('layers_quantity', parent_values=list(range(i + 1, max_layers_quantity + 1))):
                    neurons_quantity = hp.Choice(f'number_of_neurons_{i}_layer', [8, 16, 32, 64, 128, 256, 512])
            model.add(keras.layers.Dense(neurons_quantity, activation='relu',
                                         kernel_regularizer=l2(factor),
                                         bias_regularizer=l2(factor),
                                         kernel_initializer=tf.keras.initializers.he_normal()))
            if i < layers_quantity - 1 and use_bn_for_rest:
                model.add(keras.layers.BatchNormalization())
            if i < layers_quantity - 1 or regularize_output_layer:
                model.add(keras.layers.Dropout(rate))
        model.add(keras.layers.Dense(4, activation='softmax',
                                     kernel_regularizer=l2(factor if regularize_output_layer else 0),
                                     bias_regularizer=l2(factor if regularize_output_layer else 0),
                                     kernel_initializer=tf.keras.initializers.he_normal()
                                     ))
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss=choose_loss_based_on_strategy(self.best_params['strategy'], True, self.best_params['should_add_expotential']),
                      optimizer=opt,
                      metrics=[how_many_no_bets, profit_metric_based_on_strategy()])
        return model

    def perform_model_learning(self, verbose=True):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=1000, batch_size=256,
                                      verbose=1 if verbose is True else 0,
                                      shuffle=False, validation_data=(self.x_val, self.y_val),
                                      validation_batch_size=self.y_val.shape[0],
                                      callbacks=[
                                          EarlyStopping(patience=100, monitor='val_loss', mode='min', verbose=1 if verbose is True else 0),
                                          NoBetEarlyStopping(patience=75),
                                          ModelCheckpoint(self.get_path_for_saving_weights(), save_best_only=True, save_weights_only=True,
                                                          monitor='val_profit', mode='max', verbose=1 if verbose is True else 0)]
                                      )

        self.model.load_weights(self.get_path_for_saving_weights())

    def hyper_tune_model(self):
        tuner = CustomBayesianSearch(self.create_model,
                                     objective=kt.Objective('val_profit', 'max'),
                                     max_trials=300,
                                     num_initial_points=150,
                                     executions_per_trial=5,
                                     directory='./hypertuning',
                                     project_name=self.__class__.__name__ + '_test3',
                                     overwrite=False,
                                     beta=3.5)

        tuner.search(epochs=1000, batch_size=256, verbose=2, shuffle=True,
                     callbacks=[EarlyStopping(patience=100, monitor='val_loss', mode='min', verbose=1, min_delta=0.001)])

        self.print_summary_after_tuning(tuner, 10, f'./hypertuning/{self.__class__.__name__}_test3')

        return tuner
