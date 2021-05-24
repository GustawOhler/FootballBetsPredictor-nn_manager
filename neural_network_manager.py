# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2

from constants import saved_model_location, saved_weights_location
from nn_manager.common import plot_metric, eval_model_after_learning
from nn_manager.metrics import only_best_prob_odds_profit, odds_loss, how_many_no_bets, profit_wrapped_in_sqrt_loss


def create_NN_model(x_train):
    test_factor = 1e-10
    # factor = 0.000001
    factor = test_factor
    test_rate = 0.01
    # rate = test_rate
    rate = 0.45

    keras.backend.clear_session()

    # tf.compat.v1.disable_eager_execution()
    model = tf.keras.models.Sequential()
    model.add(keras.layers.BatchNormalization(momentum=0.99))
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(128, activation='relu',
                                 # activity_regularizer=l2(factor),
                                 kernel_regularizer=l2(factor),
                                 bias_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(64, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor),
                                 bias_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(64, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor),
                                 bias_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(32, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor),
                                 bias_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(16, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor),
                                 bias_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    # model.add(keras.layers.BatchNormalization(momentum=0.99))
    model.add(keras.layers.Dense(4, activation='softmax',
                                 kernel_regularizer=l2(factor),
                                 bias_regularizer=l2(factor)
                                 ))
    opt = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(loss=profit_wrapped_in_sqrt_loss,
                  optimizer=opt,
                  metrics=[how_many_no_bets, only_best_prob_odds_profit()])
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
    history = model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=1, shuffle=False, validation_data=val_set[0:2], validation_batch_size=25,
                        callbacks=[EarlyStopping(patience=100, monitor='val_loss', mode='min', verbose=1),
                                   ModelCheckpoint(saved_weights_location, save_best_only=True, save_weights_only=True, monitor='val_profit',
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
    plot_metric(history, 'profit')
    # save_model(model)
    return model
