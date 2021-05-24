import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from constants import confidence_threshold, saved_weights_location
from nn_manager.common import eval_model_after_learning_within_threshold, plot_metric, save_model
from nn_manager.metrics import categorical_crossentropy_with_bets, categorical_acc_with_bets, odds_profit_within_threshold
from nn_manager.neural_network_manager import NeuralNetworkManager


class NNPredictingMatchesManager(NeuralNetworkManager):
    def __init__(self, train_set, val_set):
        super().__init__(train_set, val_set)

    def create_model(self):
        factor = 0.002
        rate = 0.4

        # tf.compat.v1.disable_eager_execution()
        model = tf.keras.models.Sequential()
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate))
        model.add(keras.layers.Dense(256, activation='relu',
                                     # activity_regularizer=l2(factor),
                                     kernel_regularizer=l2(factor),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
        # model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate))
        model.add(keras.layers.Dense(128, activation='relu',
                                     # activity_regularizer=l2(factor),
                                     kernel_regularizer=l2(factor),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
        # model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate))
        model.add(keras.layers.Dense(64, activation='relu',
                                     # activity_regularizer=l2(factor),
                                     kernel_regularizer=l2(factor),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
        model.add(keras.layers.Dense(3, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal()))
        model.compile(loss=categorical_crossentropy_with_bets,
                      optimizer=keras.optimizers.Adam(learning_rate=0.00025),
                      metrics=[categorical_acc_with_bets, odds_profit_within_threshold(confidence_threshold)])
        return model

    def perform_model_learning(self):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=50, batch_size=128, verbose=1, shuffle=False, validation_data=(self.x_val, self.y_val),
                                      callbacks=[EarlyStopping(patience=125, monitor='val_loss', mode='min', verbose=1),
                                                 ModelCheckpoint(saved_weights_location, save_best_only=True, save_weights_only=True, monitor='val_profit',
                                                                 mode='max', verbose=1)]
                                      # callbacks=[TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_graph=True)]
                                      # callbacks=[WeightChangeMonitor()]
                                      )

        self.model.load_weights(saved_weights_location)
        save_model(self.model)

    def evaluate_model(self):
        print("Treningowy zbior: ")
        eval_model_after_learning_within_threshold(self.y_train[:, 0:3], self.model.predict(self.x_train), self.y_train[:, 4:7])
        print("Walidacyjny zbior: ")
        eval_model_after_learning_within_threshold(self.y_val[:, 0:3], self.model.predict(self.x_val), self.y_val[:, 4:7])

        plot_metric(self.history, 'loss')
        plot_metric(self.history, 'categorical_acc_with_bets')
        plot_metric(self.history, 'profit')
