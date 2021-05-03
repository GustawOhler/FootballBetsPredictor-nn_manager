import math
from enum import Enum
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from tensorflow.python.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from nn_manager.common import plot_metric, eval_model_after_learning
from nn_manager.metrics import only_best_prob_odds_profit, odds_loss, how_many_no_bets


class Categories(Enum):
    HOME_WIN = 0
    TIE = 1
    AWAY_WIN = 2
    NO_BET = 3


class WeightChangeMonitor(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_weights = list(self.model.layers[1].get_weights())

    def on_epoch_end(self, epoch, logs=None):
        if self.start_weights is not None and len(self.start_weights) > 0:
            end_weights = self.model.layers[1].get_weights()
            bias_change = np.mean(np.abs(end_weights[1] - self.start_weights[1]))
            weight_change = np.mean(np.abs(end_weights[0] - self.start_weights[0]))
            print("Bias change of first layer: " + str(bias_change) + " weight change of first layer: " + str(weight_change))


saved_model_location = "./NN_full_model/"
saved_weights_location = "./NN_model_weights/checkpoint_weights"


def create_NN_model(x_train):
    factor = 0.000003
    rate = 0.475

    # tf.compat.v1.disable_eager_execution()
    model = tf.keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(2048, activation='relu',
                                 # activity_regularizer=l2(factor),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(512, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(256, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(128, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(64, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(64, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate/2))
    model.add(keras.layers.Dense(32, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate/2))
    model.add(keras.layers.Dense(16, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dense(4, activation='softmax', kernel_regularizer=l2(factor/10)))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=odds_loss,
                  optimizer=opt,
                  metrics=[how_many_no_bets, only_best_prob_odds_profit])
    return model


def save_model(model):
    model.save(saved_model_location, overwrite=True)


def load_model():
    return keras.models.load_model(saved_model_location)


def perform_nn_learning(model, train_set, val_set):
    x_train = train_set[0]
    y_train = train_set[1]
    x_val = val_set[0]
    y_val = val_set[1]

    # tf.compat.v1.disable_eager_execution()
    history = model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1, shuffle=False, validation_data=val_set[0:2],
                        callbacks=[EarlyStopping(patience=30, monitor='val_only_best_prob_odds_profit', mode='max', verbose=1),
                                   ModelCheckpoint(saved_weights_location, save_best_only=True, save_weights_only=True, monitor='val_only_best_prob_odds_profit',
                                                   mode='max', verbose=1)]
                        # callbacks=[TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_graph=True)]
                        # callbacks=[WeightChangeMonitor()]
                        )

    model.load_weights(saved_weights_location)

    print("Treningowy zbior: ")
    eval_model_after_learning(y_train[:, 0:4], model.predict(x_train), y_train[:, 4:7])

    print("Walidacyjny zbior: ")
    eval_model_after_learning(y_val[:, 0:4], model.predict(x_val), y_val[:, 4:7])

    plot_metric(history, 'loss')
    plot_metric(history, 'only_best_prob_odds_profit')
    save_model(model)
    return model
