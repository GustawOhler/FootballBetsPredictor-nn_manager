# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2

from constants import saved_model_location, saved_weights_location, confidence_threshold
from nn_manager.common import plot_metric, eval_model_after_learning, eval_model_after_learning_within_threshold
from nn_manager.metrics import only_best_prob_odds_profit, odds_loss, how_many_no_bets, categorical_crossentropy_with_bets, categorical_acc_with_bets, \
    only_best_prob_odds_profit_within_threshold, odds_profit_within_threshold
from constants import saved_weights_location, confidence_threshold
from nn_manager.common import plot_metric, eval_model_after_learning_within_threshold, save_model
from nn_manager.metrics import categorical_crossentropy_with_bets, categorical_acc_with_bets, \
    only_best_prob_odds_profit_within_threshold


class NeuralNetworkManager(metaclass=ABCMeta):
    def __init__(self):
        self.model = create_NN_model()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def perform_model_learning(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass


def create_NN_model(x_train):
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


def perform_nn_learning(model, train_set, val_set):
    x_train = train_set[0]
    y_train = train_set[1]
    x_val = val_set[0]
    y_val = val_set[1]

    # tf.compat.v1.disable_eager_execution()
    history = model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1, shuffle=False, validation_data=val_set[0:2],
                        callbacks=[EarlyStopping(patience=125, monitor='val_loss', mode='min', verbose=1),
                                   ModelCheckpoint(saved_weights_location, save_best_only=True, save_weights_only=True, monitor='val_profit',
                                                   mode='max', verbose=1)]
                        # callbacks=[TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_graph=True)]
                        # callbacks=[WeightChangeMonitor()]
                        )

    model.load_weights(saved_weights_location)

    print("Treningowy zbior: ")
    eval_model_after_learning_within_threshold(y_train[:, 0:3], model.predict(x_train), y_train[:, 4:7])
    print("Walidacyjny zbior: ")
    eval_model_after_learning_within_threshold(y_val[:, 0:3], model.predict(x_val), y_val[:, 4:7])

    plot_metric(history, 'loss')
    plot_metric(history, 'categorical_acc_with_bets')
    plot_metric(history, 'profit')
    save_model(model)
    return model
