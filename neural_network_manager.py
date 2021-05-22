import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from constants import saved_model_location, saved_weights_location, confidence_threshold
from nn_manager.common import plot_metric, eval_model_after_learning, eval_model_after_learning_within_threshold
from nn_manager.metrics import only_best_prob_odds_profit, odds_loss, how_many_no_bets, categorical_crossentropy_with_bets, categorical_acc_with_bets, \
    only_best_prob_odds_profit_within_threshold, odds_profit_within_threshold, profit_wrapped_in_sqrt_loss
import numpy as np


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
                  metrics=[categorical_acc_with_bets])
    return model


def create_betting_strategy_model():
    factor = 0.01
    rate = 0.4

    # tf.compat.v1.disable_eager_execution()
    model = tf.keras.models.Sequential()
    model.add(keras.layers.BatchNormalization(momentum=0.85))
    model.add(keras.layers.Dense(16, activation='relu',
                                 # activity_regularizer=l2(factor),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization(momentum=0.85))
    model.add(keras.layers.Dense(16, activation='relu',
                                 # activity_regularizer=l2(factor),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization(momentum=0.85))
    model.add(keras.layers.Dense(4, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal()
                                 , kernel_regularizer=l2(factor)
                                 ))
    model.compile(loss=profit_wrapped_in_sqrt_loss,
                  optimizer=keras.optimizers.Adam(learning_rate=0.0002),
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
    history = model.fit(x_train, y_train, epochs=400, batch_size=128, verbose=1, shuffle=False, validation_data=val_set[0:2],
                        callbacks=[EarlyStopping(patience=60, monitor='val_loss', mode='min', verbose=1),
                                   ModelCheckpoint(saved_weights_location, save_best_only=True, save_weights_only=True, monitor='val_loss',
                                                   mode='min', verbose=1)]
                        # callbacks=[TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_graph=True)]
                        # callbacks=[WeightChangeMonitor()]
                        )

    model.load_weights(saved_weights_location)
    model.evaluate(x_train, y_train)
    model.evaluate(x_val, y_val)
    plot_metric(history, 'loss')
    plot_metric(history, 'categorical_acc_with_bets')

    x2_train = np.concatenate((model.predict(x_train), 1.0 / y_train[:, 4:7], y_train[:, 4:7]), axis=1)
    x2_val = np.concatenate((model.predict(x_val), 1.0 / y_val[:, 4:7], y_val[:, 4:7]), axis=1)

    betting_model = create_betting_strategy_model()
    history2 = betting_model.fit(x2_train, y_train, epochs=1000, batch_size=256, verbose=1, shuffle=False, validation_data=(x2_val, y_val),
                                 callbacks=[EarlyStopping(patience=150, monitor='val_loss', mode='min', verbose=1),
                                            ModelCheckpoint(saved_weights_location, save_best_only=True, save_weights_only=True, monitor='val_profit',
                                                            mode='max', verbose=1)]
                                 )

    print("Treningowy zbior: ")
    betting_model.evaluate(x2_train, y_train, batch_size=256)
    eval_model_after_learning(y_train[:, 0:3], betting_model.predict(x2_train), y_train[:, 4:7])
    print("Walidacyjny zbior: ")
    betting_model.evaluate(x2_val, y_val, batch_size=1)
    eval_model_after_learning(y_val[:, 0:3], betting_model.predict(x2_val), y_val[:, 4:7])

    plot_metric(history2, 'loss')
    plot_metric(history2, 'profit')
    # save_model(model)
    return model, betting_model
