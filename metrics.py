import tensorflow as tf
from tensorflow import keras

from nn_manager.common import confidence_threshold


def odds_loss(y_true, y_pred):
    win_home_team = y_true[:, 0:1]
    draw = y_true[:, 1:2]
    win_away = y_true[:, 2:3]
    no_bet = y_true[:, 3:4]
    odds_a = y_true[:, 4:5]
    odds_draw = y_true[:, 5:6]
    odds_b = y_true[:, 6:7]
    gain_loss_vector = tf.concat([win_home_team * (odds_a - 1) + (1 - win_home_team) * -1,
                                  draw * (odds_draw - 1) + (1 - draw) * -1,
                                  win_away * (odds_b - 1) + (1 - win_away) * -1,
                                  tf.zeros_like(odds_a)], axis=1)
    return -1 * tf.reduce_mean(tf.reduce_sum(gain_loss_vector * y_pred, axis=1))


def only_best_prob_odds_profit(y_true, y_pred):
    win_home_team = y_true[:, 0:1]
    draw = y_true[:, 1:2]
    win_away = y_true[:, 2:3]
    no_bet = y_true[:, 3:4]
    odds_a = y_true[:, 4:5]
    odds_draw = y_true[:, 5:6]
    odds_b = y_true[:, 6:7]
    gain_loss_vector = tf.concat([win_home_team * (odds_a - 1) + (1 - win_home_team) * -1,
                                  draw * (odds_draw - 1) + (1 - draw) * -1,
                                  win_away * (odds_b - 1) + (1 - win_away) * -1
                                  # tf.zeros_like(odds_a)
                                  ], axis=1)
    outcome_possibilities = 1.0/y_true[:, 4:7]
    zerod_prediction = tf.where(
        tf.not_equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
        tf.zeros_like(y_pred),
        y_pred
    )
    predictions_above_threshold = tf.where(
        tf.greater_equal(tf.subtract(zerod_prediction, outcome_possibilities), confidence_threshold),
        tf.ones_like(y_pred),
        tf.zeros_like(y_pred)
    )
    return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * predictions_above_threshold, axis=1))


def how_many_no_bets(y_true, y_pred):
    all_predictions = y_pred[:, 0:4]
    classes = tf.math.argmax(all_predictions, 1)
    wanted_class = tf.constant(3, dtype="int64")
    logical = tf.math.equal(classes, wanted_class)
    return tf.reduce_sum(tf.cast(logical, tf.float32)) * 100.0 / tf.cast(tf.shape(y_pred)[0], tf.float32)


def categorical_crossentropy_with_bets(y_true, y_pred):
    return keras.losses.categorical_crossentropy(y_true[:, 0:3], y_pred)


def categorical_acc_with_bets(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true[:, 0:3], y_pred)