import tensorflow as tf
from tensorflow import keras

from constants import ChoosingBetsStrategy, PredMatchesStrategy


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


# Warianty:
# 0 - 5 - Wszystkie prawdopodobieństwa zsumowane
# 0 - Kara za nieobstawianie, funkcja tf.math.pow(0.5, summed_gain_loss - 1.0) - (tf.math.log(predicted_gain_loss_clipped + 1.0))
# 1 - Kara za nieobstawianie, funkcja tf.math.pow(0.5, summed_gain_loss - 1.0)
# 2 - Kara za nieobstawianie, funkcja - (tf.math.log(predicted_gain_loss_clipped + 1.0))
# 3 - Bez kary za nieobstawianie, funkcja tf.math.pow(0.5, summed_gain_loss - 1.0) - (tf.math.log(predicted_gain_loss_clipped + 1.0))
# 4 - Bez kary za nieobstawianie, funkcja tf.math.pow(0.5, summed_gain_loss - 1.0)
# 5 - Bez kary za nieobstawianie, funkcja - (tf.math.log(predicted_gain_loss_clipped + 1.0))
# 6 - 11 - To samo tylko że tylko z najwyższym prawdopodobieństwem
# 6 - Kara za nieobstawianie, funkcja tf.math.pow(0.5, summed_gain_loss - 1.0) - (tf.math.log(predicted_gain_loss_clipped + 1.0))
# 7 - Kara za nieobstawianie, funkcja tf.math.pow(0.5, summed_gain_loss - 1.0)
# 8 - Kara za nieobstawianie, funkcja - (tf.math.log(predicted_gain_loss_clipped + 1.0))
# 9 - Bez kary za nieobstawianie, funkcja tf.math.pow(0.5, summed_gain_loss - 1.0) - (tf.math.log(predicted_gain_loss_clipped + 1.0))
# 10 - Bez kary za nieobstawianie, funkcja tf.math.pow(0.5, summed_gain_loss - 1.0)
# 11 - Bez kary za nieobstawianie, funkcja - (tf.math.log(predicted_gain_loss_clipped + 1.0))
def profit_and_loss_tuning(variant: int):
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
                                      tf.ones_like(odds_a) * (-0.01 if variant in [0, 1, 2, 6, 7, 8] else 0)], axis=1)
        if variant >= 6:
            y_pred = tf.where(
                tf.not_equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
                0.0,
                y_pred
            )
        epsilon = 1e-10
        predicted_gain_loss = y_pred * gain_loss_vector
        predicted_gain_loss_log_clipped = tf.clip_by_value(y_pred * gain_loss_vector, -2.0 + epsilon, tf.float32.max)
        predicted_gain_loss_abs = tf.clip_by_value(tf.math.abs(predicted_gain_loss), epsilon, tf.float32.max)
        summed_gain_loss = tf.reduce_sum(predicted_gain_loss, axis=1)
        predicted_gain_loss_clipped = tf.clip_by_value(summed_gain_loss, epsilon, tf.float32.max)
        if variant in [0, 3, 6, 9]:
            function_value = tf.math.pow(0.5, summed_gain_loss - 1.0) - (tf.math.log(predicted_gain_loss_clipped + 1.0))
        elif variant in [1, 4, 7, 10]:
            function_value = tf.math.pow(0.5, summed_gain_loss - 1.0)
        elif variant in [2, 5, 8, 11]:
            function_value = - (tf.math.log(predicted_gain_loss_clipped + 1.0))
        return tf.reduce_mean(function_value)

    return profit_wrapped_in_sqrt_loss


def all_bets_profit_wrapped_loss(should_add_expotential: bool):
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
        epsilon = 1e-10
        predicted_gain_loss = y_pred * gain_loss_vector
        summed_gain_loss = tf.reduce_sum(predicted_gain_loss, axis=1)
        if should_add_expotential:
            predicted_gain_loss_clipped = tf.clip_by_value(summed_gain_loss, epsilon, tf.float32.max)
            function_value = tf.math.pow(0.5, summed_gain_loss - 1.0) - (tf.math.log(predicted_gain_loss_clipped + 1.0))
        else:
            predicted_gain_loss_clipped = tf.clip_by_value(summed_gain_loss, -1.0 + epsilon, tf.float32.max)
            function_value = -(tf.math.log(predicted_gain_loss_clipped + 1.0))
        return tf.reduce_mean(function_value)

    return profit_wrapped_in_sqrt_loss


def one_bet_profit_wrapped_loss(should_add_expotential: bool):
    def one_bet_profit_wrapped_in_sqrt_loss(y_true, y_pred):
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
        y_pred = tf.where(
            tf.not_equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
            0.0,
            y_pred
        )
        epsilon = 1e-10
        predicted_gain_loss = y_pred * gain_loss_vector
        summed_gain_loss = tf.reduce_sum(predicted_gain_loss, axis=1)
        if should_add_expotential:
            predicted_gain_loss_clipped = tf.clip_by_value(summed_gain_loss, epsilon, tf.float32.max)
            function_value = tf.math.pow(0.5, summed_gain_loss - 1.0) - (tf.math.log(predicted_gain_loss_clipped + 1.0))
        else:
            predicted_gain_loss_clipped = tf.clip_by_value(summed_gain_loss, -1.0 + epsilon, tf.float32.max)
            function_value = -(tf.math.log(predicted_gain_loss_clipped + 1.0))
        return tf.reduce_mean(function_value)

    return one_bet_profit_wrapped_in_sqrt_loss


def choose_loss_based_on_strategy(strategy: ChoosingBetsStrategy, choose_bets: bool, should_add_expotential: bool):
    if choose_bets:
        if strategy == ChoosingBetsStrategy.AllOnBestResult or strategy == ChoosingBetsStrategy.BetOnBestResultWithRetProb:
            return one_bet_profit_wrapped_loss(should_add_expotential)
        elif strategy == ChoosingBetsStrategy.MalafosseUnlessNoBet or strategy == ChoosingBetsStrategy.OriginalMalafosse:
            return all_bets_profit_wrapped_loss(should_add_expotential)



def only_best_prob_odds_profit(should_zero_one_probs: bool):
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
        if should_zero_one_probs:
            zerod_prediction = tf.where(
                tf.not_equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
                tf.zeros_like(y_pred, dtype='float32'),
                tf.ones_like(y_pred, dtype='float32')
            )
        else:
            zerod_prediction = tf.where(
                tf.not_equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
                tf.zeros_like(y_pred, dtype='float32'),
                y_pred
            )
        return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * zerod_prediction, axis=1))

    inner_metric.__name__ = 'profit'
    return inner_metric


def only_best_prob_odds_sum_profit(should_zero_one_probs: bool):
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
        if should_zero_one_probs:
            zerod_prediction = tf.where(
                tf.not_equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
                tf.zeros_like(y_pred, dtype='float32'),
                tf.ones_like(y_pred, dtype='float32')
            )
        else:
            zerod_prediction = tf.where(
                tf.not_equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
                tf.zeros_like(y_pred, dtype='float32'),
                y_pred
            )
        return tf.reduce_sum(gain_loss_vector * zerod_prediction)

    inner_metric.__name__ = 'accumulated_profit'
    return inner_metric


def all_odds_profit(exclude_no_bet:bool):
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
        if exclude_no_bet:
            no_bet_pred = y_pred[:, 3:4]
            only_no_bets = tf.where(
                tf.equal(y_pred, no_bet_pred),
                y_pred,
                tf.zeros_like(y_pred, dtype='float32')
            )
            without_no_bets_prediction = tf.where(
                tf.equal(tf.reduce_max(y_pred, axis=1, keepdims=True), no_bet_pred),
                only_no_bets,
                y_pred
            )
            return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * without_no_bets_prediction, axis=1))
        else:
            return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * y_pred, axis=1))

    inner_metric.__name__ = 'profit'
    return inner_metric


def profit_metric_based_on_strategy(strategy: ChoosingBetsStrategy):
    if strategy == ChoosingBetsStrategy.AllOnBestResult:
        return only_best_prob_odds_profit(True)
    elif strategy == ChoosingBetsStrategy.BetOnBestResultWithRetProb:
        return only_best_prob_odds_profit(False)
    elif strategy == ChoosingBetsStrategy.MalafosseUnlessNoBet:
        return all_odds_profit(True)
    elif strategy == ChoosingBetsStrategy.OriginalMalafosse:
        return all_odds_profit(False)


def get_all_profit_metrics_for_pred_matches(threshold):
    metrics = []
    for option in PredMatchesStrategy:
        if option == PredMatchesStrategy.AllOnBiggestDifferenceOverThreshold:
            profit_metric = odds_profit_with_biggest_gap_over_threshold(threshold)
        elif option == PredMatchesStrategy.AllOnBestOverThreshold:
            profit_metric = only_best_prob_odds_profit_within_threshold(threshold)
        elif option in [PredMatchesStrategy.RelativeOnBiggestDifferenceOverThreshold, PredMatchesStrategy.RelativeOnResultsOverThreshold,
                        PredMatchesStrategy.RelativeOnBestOverThreshold]:
            profit_metric = relative_profit_over_threshold(threshold*2.0, option)
        elif option == PredMatchesStrategy.KellyCriterion:
            profit_metric = relative_profit_with_kelly_criterion()
        else:
            break
        profit_metric.__name__ = option.value
        metrics.append(profit_metric)
    return metrics


def get_predictions_over_threshold(y_true, y_pred, threshold):
    outcome_possibilities = 1.0 / y_true[:, 4:7]
    return tf.where(
        tf.greater_equal(tf.subtract(y_pred, outcome_possibilities), threshold),
        y_pred,
        tf.zeros_like(y_pred)
    )


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
                                      ], axis=1)
        predictions_over_threshold = get_predictions_over_threshold(y_true, y_pred, threshold)
        only_most_probable_prediction = tf.where(
            tf.equal(tf.reduce_max(predictions_over_threshold, axis=1, keepdims=True), predictions_over_threshold),
            predictions_over_threshold,
            tf.zeros_like(predictions_over_threshold)
        )
        return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * only_most_probable_prediction, axis=1))

    inner_metric.__name__ = 'profit'
    return inner_metric


def odds_profit_with_biggest_gap_over_threshold(threshold):
    def inner_metric(y_true, y_pred):
        win_home_team = y_true[:, 0:1]
        draw = y_true[:, 1:2]
        win_away = y_true[:, 2:3]
        odds_a = y_true[:, 4:5]
        odds_draw = y_true[:, 5:6]
        odds_b = y_true[:, 6:7]
        gain_loss_vector = tf.concat([win_home_team * (odds_a - 1) + (1 - win_home_team) * -1,
                                      draw * (odds_draw - 1) + (1 - draw) * -1,
                                      win_away * (odds_b - 1) + (1 - win_away) * -1
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


def pred_matches_precision(threshold):
    prec = tf.keras.metrics.Precision()
    def inner_metric(y_true, y_pred):
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
        prec.reset_state()
        prec.update_state(y_true[:, 0:3], predictions_above_threshold)
        return prec.result()

    inner_metric.__name__ = 'precision'
    return inner_metric


def pred_matches_how_many_no_bets(threshold):
    def inner_metric(y_true, y_pred):
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
        elements_equal_to_value = tf.equal(tf.reduce_max(predictions_above_threshold, axis=1, keepdims=True), tf.constant(0, dtype=tf.float32))
        as_ints = tf.cast(elements_equal_to_value, tf.float32)
        count = tf.reduce_sum(as_ints)
        return count * 100.0 / tf.cast(tf.shape(y_pred)[0], tf.float32)

    inner_metric.__name__ = 'how_many_no_bets'
    return inner_metric


def relative_profit_over_threshold(threshold, chosen_strategy: PredMatchesStrategy):
    def inner_metric(y_true, y_pred):
        win_home_team = y_true[:, 0:1]
        draw = y_true[:, 1:2]
        win_away = y_true[:, 2:3]
        odds_a = y_true[:, 4:5]
        odds_draw = y_true[:, 5:6]
        odds_b = y_true[:, 6:7]
        gain_loss_vector = tf.concat([win_home_team * (odds_a - 1) + (1 - win_home_team) * -1,
                                      draw * (odds_draw - 1) + (1 - draw) * -1,
                                      win_away * (odds_b - 1) + (1 - win_away) * -1
                                      ], axis=1)
        outcome_possibilities = 1.0 / y_true[:, 4:7]
        prediction_diff = tf.subtract(y_pred, outcome_possibilities)
        prediction_diff_without_negative = tf.where(
            tf.greater_equal(prediction_diff, tf.constant(0, dtype=tf.float32)),
            prediction_diff,
            tf.zeros_like(prediction_diff)
        )
        highest_gap_prediction = tf.reduce_max(prediction_diff, axis=1, keepdims=True)
        if chosen_strategy == PredMatchesStrategy.RelativeOnBestOverThreshold:
            zerod_prediction = tf.where(
                tf.not_equal(prediction_diff_without_negative, tf.constant(0, dtype=tf.float32)),
                y_pred,
                tf.zeros_like(y_pred)
            )
            only_most_probable_prediction = tf.where(
                tf.equal(tf.reduce_max(zerod_prediction, axis=1, keepdims=True), zerod_prediction),
                tf.divide(prediction_diff_without_negative, threshold),
                tf.zeros_like(prediction_diff_without_negative)
            )
            capped_at_1_matrix = tf.where(
                tf.greater_equal(only_most_probable_prediction, 1.0),
                tf.ones_like(only_most_probable_prediction),
                only_most_probable_prediction
            )
            return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * capped_at_1_matrix, axis=1))
        elif chosen_strategy == PredMatchesStrategy.RelativeOnBiggestDifferenceOverThreshold:
            only_most_probable_prediction = tf.where(
                tf.equal(highest_gap_prediction, prediction_diff_without_negative),
                tf.divide(prediction_diff_without_negative, threshold),
                tf.zeros_like(prediction_diff_without_negative)
            )
            capped_at_1_matrix = tf.where(
                tf.greater_equal(only_most_probable_prediction, 1.0),
                tf.ones_like(only_most_probable_prediction),
                only_most_probable_prediction
            )
            return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * capped_at_1_matrix, axis=1))
        elif chosen_strategy == PredMatchesStrategy.RelativeOnResultsOverThreshold:
            relative_stake = tf.divide(prediction_diff_without_negative, threshold)
            return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * relative_stake, axis=1))
        else:
            raise ValueError(f"Chosen strategy can't be {chosen_strategy.value}")

    inner_metric.__name__ = 'profit'
    return inner_metric


def get_gain_loss_vector(y_true):
    win_home_team = y_true[:, 0:1]
    draw = y_true[:, 1:2]
    win_away = y_true[:, 2:3]
    odds_a = y_true[:, 4:5]
    odds_draw = y_true[:, 5:6]
    odds_b = y_true[:, 6:7]
    gain_loss_vector = tf.concat([win_home_team * (odds_a - 1) + (1 - win_home_team) * -1,
                                  draw * (odds_draw - 1) + (1 - draw) * -1,
                                  win_away * (odds_b - 1) + (1 - win_away) * -1
                                  ], axis=1)
    return gain_loss_vector


def relative_profit_with_kelly_criterion():
    def inner_metric(y_true, y_pred):
        outcome_possibilities = 1.0 / y_true[:, 4:7]
        prediction_diff = tf.subtract(y_pred, outcome_possibilities)
        highest_gap_prediction = tf.reduce_max(prediction_diff, axis=1, keepdims=True)
        zerod_prediction = tf.where(
            tf.equal(highest_gap_prediction, prediction_diff),
            y_pred,
            tf.zeros_like(prediction_diff)
        )
        all_odds = y_true[:, 4:7]
        profitability_rate = zerod_prediction * all_odds - tf.constant(1.0, dtype=tf.float32)
        only_positive_prof_rates = tf.where(tf.greater_equal(profitability_rate, tf.constant(0.0, dtype=tf.float32)),
                                            profitability_rate,
                                            tf.zeros_like(profitability_rate))
        stake = only_positive_prof_rates/(all_odds-tf.constant(1.0, dtype=tf.float32))
        return tf.reduce_mean(tf.reduce_sum(get_gain_loss_vector(y_true) * stake, axis=1))

    inner_metric.__name__ = 'profit'
    return inner_metric


def how_many_no_bets(y_true, y_pred):
    all_predictions = y_pred[:, 0:4]
    classes = tf.math.argmax(all_predictions, 1)
    wanted_class = tf.constant(3, dtype="int64")
    logical = tf.math.equal(classes, wanted_class)
    return tf.reduce_sum(tf.cast(logical, tf.float32)) * 100.0 / tf.cast(tf.shape(y_pred)[0], tf.float32)


def how_many_bets(y_true, y_pred):
    all_predictions = y_pred[:, 0:4]
    classes = tf.math.argmax(all_predictions, 1)
    wanted_class = tf.constant(3, dtype="int64")
    logical = tf.math.not_equal(classes, wanted_class)
    return tf.reduce_sum(tf.cast(logical, tf.float32)) * 100.0 / tf.cast(tf.shape(y_pred)[0], tf.float32)


def choose_bets_precision():
    prec = tf.keras.metrics.Precision()
    def inner_metric(y_true, y_pred):
        prec.reset_state()
        only_results = y_true[:, 0:3]
        zerod_pred = tf.where(
            tf.equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
            tf.ones_like(y_pred),
            tf.zeros_like(y_pred)
        )
        prec.update_state(only_results, zerod_pred[:, 0:3])
        return prec.result()

    inner_metric.__name__ = 'precision'
    return inner_metric


def categorical_crossentropy_with_bets(y_true, y_pred):
    return keras.losses.categorical_crossentropy(y_true[:, 0:3], y_pred)


def categorical_acc_with_bets(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true[:, 0:3], y_pred)
