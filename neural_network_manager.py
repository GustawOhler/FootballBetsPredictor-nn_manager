# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from constants import saved_model_location, saved_weights_location, confidence_threshold
from nn_manager.common import plot_metric, eval_model_after_learning, eval_model_after_learning_within_threshold
from nn_manager.metrics import only_best_prob_odds_profit, odds_loss, how_many_no_bets, categorical_crossentropy_with_bets, categorical_acc_with_bets, \
    only_best_prob_odds_profit_within_threshold


def create_NN_model(x_train):
    factor = 0.0003
    rate = 0.5

    # tf.compat.v1.disable_eager_execution()
    model = tf.keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(4096, activation='relu',
                                 # activity_regularizer=l2(factor),
                                 # kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(512, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 # kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(512, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 # kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(256, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 # kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(256, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 # kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(128, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 # kernel_regularizer=l2(factor/100),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(128, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 # kernel_regularizer=l2(factor/100),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.Dense(64, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 # kernel_regularizer=l2(factor/10),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate/2))
    model.add(keras.layers.Dense(64, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 # kernel_regularizer=l2(factor/10),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate/2))
    model.add(keras.layers.Dense(32, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 # kernel_regularizer=l2(factor/3),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate/3))
    model.add(keras.layers.Dense(16, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 # kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dense(3, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal()))
    model.compile(loss=categorical_crossentropy_with_bets,
                  optimizer=keras.optimizers.Adam(learning_rate=0.0015),
                  metrics=[categorical_acc_with_bets, only_best_prob_odds_profit_within_threshold(confidence_threshold)])
    # only_best_prob_odds_profit
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
    history = model.fit(x_train, y_train, epochs=350, batch_size=128, verbose=1, shuffle=False, validation_data=val_set[0:2],
                        callbacks=[EarlyStopping(patience=60, monitor='val_profit', mode='max', verbose=1),
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
    plot_metric(history, 'profit')
    save_model(model)
    return model
