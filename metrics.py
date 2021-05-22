import tensorflow as tf
from tensorflow import keras


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
    return -1.0 * tf.reduce_mean(tf.reduce_sum(gain_loss_vector * y_pred, axis=1))


def profit_wrapped_in_sqrt_loss(y_true, y_pred):
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
    # zeros = tf.zeros_like(y_pred, dtype='float32')
    # ones = tf.ones_like(y_pred, dtype='float32')
    # zerod_prediction = tf.where(
    #     tf.not_equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
    #     y_pred * 0.1,
    #     y_pred
    # )
    epsilon = 0.000001
    # log = (tf.math.log(tf.clip_by_value(y_pred*y_true[:, 0:4], epsilon, 1.0 - epsilon))+tf.math.log(tf.clip_by_value((1.0-y_pred)*(1.0-y_true[:, 0:4]), epsilon,
    #                                                                                                            1.0 - epsilon)))*gain_loss_vector
    # function_value = log * gain_loss_vector
    predicted_gain_los = y_pred * gain_loss_vector
    predicted_gain_loss_abs = tf.clip_by_value(tf.math.abs(predicted_gain_los), epsilon, tf.float32.max)
    # tf.math.pow(0.5, predicted_gain_los)-(predicted_gain_los / predicted_gain_loss_abs * tf.math.sqrt(predicted_gain_loss_abs))
    function_value = -(predicted_gain_los / predicted_gain_loss_abs * tf.math.sqrt(predicted_gain_loss_abs))
    return tf.reduce_mean(tf.reduce_sum(function_value, axis=1))


def only_best_prob_odds_profit():
    def inner_metric(y_true, y_pred):
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
        zerod_prediction = tf.where(
            tf.not_equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
            tf.zeros_like(y_pred),
            tf.ones_like(y_pred)
        )
        return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * zerod_prediction, axis=1))

    inner_metric.__name__ = 'profit'
    return inner_metric


def only_best_prob_odds_profit_within_threshold(threshold):
    def inner_metric(y_true, y_pred):
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
        outcome_possibilities = 1.0 / y_true[:, 4:7]
        zerod_prediction = tf.where(
            tf.not_equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
            tf.zeros_like(y_pred),
            y_pred
        )
        predictions_above_threshold = tf.where(
            tf.greater_equal(tf.subtract(zerod_prediction, outcome_possibilities), threshold),
            tf.ones_like(y_pred),
            tf.zeros_like(y_pred)
        )
        return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * predictions_above_threshold, axis=1))

    inner_metric.__name__ = 'profit'
    return inner_metric


def odds_profit_within_threshold(threshold):
    def inner_metric(y_true, y_pred):
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
        outcome_possibilities = 1.0 / y_true[:, 4:7]
        prediction_diff = tf.subtract(y_pred, outcome_possibilities)
        highest_gap_prediction = tf.reduce_max(prediction_diff, axis=1, keepdims=True)
        zerod_prediction = tf.where(
            tf.not_equal(highest_gap_prediction, prediction_diff),
            tf.zeros_like(prediction_diff),
            prediction_diff
        )
        predictions_above_threshold = tf.where(
            tf.greater_equal(zerod_prediction, threshold),
            tf.ones_like(zerod_prediction),
            tf.zeros_like(zerod_prediction)
        )
        return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * predictions_above_threshold, axis=1))

    inner_metric.__name__ = 'profit'
    return inner_metric


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
