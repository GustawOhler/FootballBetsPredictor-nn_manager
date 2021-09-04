import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from constants import saved_model_weights_base_path
from nn_manager.common import eval_model_after_learning_within_threshold, plot_metric, save_model
from nn_manager.custom_bayesian_tuner import CustomBayesianSearch
from nn_manager.metrics import categorical_crossentropy_with_bets, categorical_acc_with_bets, odds_profit_within_threshold
from nn_manager.neural_network_manager import NeuralNetworkManager


class NNPredictingMatchesManager(NeuralNetworkManager):
    def __init__(self, train_set, val_set, should_hyper_tune, test_set):
        self.best_params = {
            "regularization_factor": 0.002,
            "dropout_rate": 0.4,
            "layers_quantity": 3,
            "n_of_neurons": [256, 128, 64],
            "confidence_threshold": 0.03,
            "learning_rate": 0.00025,
            "input_dropout_rate": 0.4,
            "use_bn_for_input": True,
            "use_bn_for_rest": False
        }
        super().__init__(train_set, val_set, should_hyper_tune, test_set)

    def create_model(self, hp: kt.HyperParameters = None):
        factor = self.best_params["regularization_factor"] if not self.should_hyper_tune else hp.Float('regularization_factor', 0, 1e-2, step=1e-10)
        input_dropout_rate = self.best_params["input_dropout_rate"] if not self.should_hyper_tune else hp.Float('dropout_rate', 0, 0.65, step=0.025)
        rate = self.best_params["dropout_rate"] if not self.should_hyper_tune else hp.Float('dropout_rate', 0, 0.65, step=0.025)
        max_layers_quantity = 6
        layers_quantity = self.best_params["layers_quantity"] if not self.should_hyper_tune else hp.Int('layers_quantity', 1, max_layers_quantity)
        confidence_threshold = self.best_params["confidence_threshold"] if not self.should_hyper_tune else hp.Float('confidence_threshold', 0.005, 0.15,
                                                                                                                    step=0.005)
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
                                         # activity_regularizer=l2(factor),
                                         kernel_regularizer=l2(factor),
                                         bias_regularizer=l2(factor),
                                         kernel_initializer=tf.keras.initializers.he_normal()))
            if use_bn_for_rest:
                model.add(keras.layers.BatchNormalization())
            if i < layers_quantity - 1 or regularize_output_layer:
                model.add(keras.layers.Dropout(rate))

        model.add(keras.layers.Dense(3, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal(),
                                     kernel_regularizer=l2(factor if regularize_output_layer else 0),
                                     bias_regularizer=l2(factor if regularize_output_layer else 0)))
        model.compile(loss=categorical_crossentropy_with_bets,
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=[categorical_acc_with_bets, odds_profit_within_threshold(confidence_threshold)])
        return model

    def perform_model_learning(self):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=750, batch_size=256, verbose=1, shuffle=False, validation_data=(self.x_val,
                                                                                                                                         self.y_val),
                                      validation_batch_size=16,
                                      callbacks=[EarlyStopping(patience=125, monitor='val_loss', mode='min', verbose=1),
                                                 ModelCheckpoint(self.get_path_for_saving_weights(), save_best_only=True, save_weights_only=True,
                                                                 monitor='val_profit', mode='max', verbose=1)]
                                      # callbacks=[TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_graph=True)]
                                      # callbacks=[WeightChangeMonitor()]
                                      )

        self.model.load_weights(self.get_path_for_saving_weights())
        save_model(self.model, self.get_path_for_saving_model())

    def hyper_tune_model(self):
        tuner = CustomBayesianSearch(self.create_model,
                                     objective=kt.Objective('val_profit', 'max'),
                                     max_trials=300,
                                     num_initial_points=150,
                                     executions_per_trial=5,
                                     directory='.\\hypertuning',
                                     project_name=self.__class__.__name__ + '_test2',
                                     overwrite=False,
                                     beta=3.5)
        tuner.search(batch_size=256, epochs=1000, shuffle=True, verbose=2,
                     callbacks=[EarlyStopping(patience=100, monitor='val_loss', mode='min', verbose=1, min_delta=0.0005)])

        self.print_summary_after_tuning(tuner, 10, f'./hypertuning/{self.__class__.__name__}_test2')

        return tuner

    def evaluate_model(self, should_plot=True, should_print_train=True, hyperparams=None):
        self.evaluate_model_with_threshold(should_plot, should_print_train, hyperparams)

    # def evaluate_model(self):
    #     print("Treningowy zbior: ")
    #     eval_model_after_learning_within_threshold(self.y_train[:, 0:3], self.model.predict(self.x_train), self.y_train[:, 4:7],
    #                                                self.best_params["confidence_threshold"])
    #     print("Walidacyjny zbior: ")
    #     eval_model_after_learning_within_threshold(self.y_val[:, 0:3], self.model.predict(self.x_val), self.y_val[:, 4:7],
    #                                                self.best_params["confidence_threshold"])
    #
    #     if self.x_test is not None:
    #         print("Testowy zbior: ")
    #         # pprint.pprint(self.model.evaluate(self.x_test, self.y_test, verbose=0, batch_size=16, return_dict=True), width=1)
    #         eval_model_after_learning_within_threshold(self.y_test[:, 0:3], self.model.predict(self.x_test), self.y_test[:, 4:7],
    #                                                    self.best_params["confidence_threshold"])
    #
    #     plot_metric(self.history, 'loss')
    #     plot_metric(self.history, 'categorical_acc_with_bets')
    #     plot_metric(self.history, 'profit')
