# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2

from nn_manager.common import plot_metric, eval_model_after_learning
from nn_manager.metrics import only_best_prob_odds_profit, categorical_crossentropy_with_bets, categorical_acc_with_bets

results_to_description_dict = {0: 'Wygrana gospodarzy', 1: 'Remis', 2: 'Wygrana gości', 3: 'Brak zakładu'}
saved_model_location = "./nn_manager/NN_full_model/"
saved_weights_location = "./nn_manager/NN_model_weights/checkpoint_weights"
confidence_threshold = 0.015


def create_NN_model(x_train):
    factor = 0.0003
    rate = 0.05

    # tf.compat.v1.disable_eager_execution()
    model = tf.keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(4096, activation='relu',
                                 activity_regularizer=l2(factor/2),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(1024, activation='relu',
    #                              activity_regularizer=l2(factor/2),
    #                              kernel_regularizer=l2(factor),
    #                              kernel_initializer=tf.keras.initializers.he_normal()))
    # model.add(keras.layers.Dropout(rate))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1024, activation='relu',
                                 activity_regularizer=l2(factor/2),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation='relu',
                                 # activity_regularizer=l2(factor/4),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate / 2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, activation='relu',
                                 # activity_regularizer=l2(factor / 10),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate / 2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(32, activation='relu',
                                 # activity_regularizer=l2(factor / 10),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate / 4))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(16, activation='relu',
                                 # activity_regularizer=l2(factor / 10),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dense(3, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal()))
    model.compile(loss=categorical_crossentropy_with_bets,
                  optimizer=keras.optimizers.Adam(learning_rate=0.0015),
                  metrics=[categorical_acc_with_bets, only_best_prob_odds_profit])
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
                        callbacks=[EarlyStopping(patience=60, min_delta=0.0001, monitor='val_only_best_prob_odds_profit', mode='max', verbose=1),
                                   ModelCheckpoint(saved_weights_location, save_best_only=True, save_weights_only=True,
                                                   monitor='val_only_best_prob_odds_profit',
                                                   mode='max', verbose=1)]
                        # TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_images=True, write_graph=True)]
                        )

    model.load_weights(saved_weights_location)

    print("Treningowy zbior: ")
    eval_model_after_learning(y_train[:, 0:3], model.predict(x_train), y_train[:, 4:7])
    print("Walidacyjny zbior: ")
    eval_model_after_learning(y_val[:, 0:3], model.predict(x_val), y_val[:, 4:7])

    plot_metric(history, 'loss')
    plot_metric(history, 'only_best_prob_odds_profit')
    plot_metric(history, 'categorical_acc_with_bets')
    save_model(model)
    return model
