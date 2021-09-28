import datetime
import math
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import pandas as pd
from constants import results_to_description_dict, saved_model_based_path
from dataset_manager.class_definitions import DatasetSplit
from dataset_manager.common_funtions import get_curr_dataset_column_names
from dataset_manager.dataset_manager import load_ids_in_right_order
from models import Match


def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    if metric == 'loss':
        concated_metrics = np.concatenate((np.asarray(train_metrics), np.asarray(val_metrics)))
        avg = np.average(concated_metrics)
        std_dev = math.sqrt(np.sum(concated_metrics * concated_metrics) / len(concated_metrics) - avg ** 2)
        start = avg - 3 * std_dev
        end = avg + 2 * std_dev
        plt.ylim([start, end])
    if metric == 'profit':
        plt.axhline(y=0, color='r')
    plt.show()


def plot_profit_for_thesis(history):
    metric = 'profit'
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.xlabel("Epoki")
    plt.ylabel("Zysk")
    plt.legend(["Zysk dla zbioru treningowego", 'Zysk dla zbioru walidacyjnego'])
    plt.axhline(y=0, color='r')
    plt.ylim([-0.055, 0.025])
    plt.grid()
    plt.savefig(f'./thesis_plots/profit_plot_{datetime.datetime.now().timestamp()}.png', dpi=900)


def plot_many_metrics(history, metrics: list, print_only_validation:bool, do_profit_line: bool):
    legend_array = []
    for metric in metrics:
        train_metrics = history.history[metric]
        val_metrics = history.history['val_' + metric]
        epochs = range(1, len(train_metrics) + 1)
        if not print_only_validation:
            plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title('Training and validation')
        if not print_only_validation:
            legend_array.append("train_" + metric)
        legend_array.append('val_' + metric)
    plt.xlabel("Epochs")
    plt.legend(legend_array, loc=4, fontsize='xx-small')
    if do_profit_line:
        plt.axhline(y=0, color='r')
    plt.show()
    plt.savefig('./requested_metrics.png', dpi=900)


def get_profits(predicted_classes, actual_classes, odds):
    profits = []
    for i in range(predicted_classes.shape[0]):
        if predicted_classes[i] == Categories.NO_BET.value:
            profits.append(0.0)
        elif predicted_classes[i] == actual_classes[i]:
            profits.append(odds[i][actual_classes[i]] - 1.0)
        else:
             profits.append(-1.0)
    return profits

def show_winnings(predicted_classes, actual_classes, odds):
    winnings = 0.0
    for i in range(predicted_classes.shape[0]):
        # Jesli siec zdecydowala sie nie obstawiac meczu
        if predicted_classes[i] == Categories.NO_BET.value:
            continue
        elif predicted_classes[i] == actual_classes[i]:
            winnings = winnings + odds[i][actual_classes[i]] - 1.0
        else:
            winnings = winnings - 1.0
    print("Bilans wygranych/strat z potencjalnych zakładów: " + str("{:.2f}".format(winnings)) + " na " + str(len(predicted_classes)) + " meczow (" +
          str("{:.2f}%)".format(winnings / len(predicted_classes) * 100)))


def show_winnings_for_classes(predicted_classes, actual_classes, odds):
    for idx, j in enumerate(Categories):
        winnings = 0.0
        for i in range(predicted_classes.shape[0]):
            if predicted_classes[i] != j.value:
                continue
            elif predicted_classes[i] == Categories.NO_BET.value:
                continue
            elif predicted_classes[i] == actual_classes[i]:
                winnings = winnings + odds[i][actual_classes[i]] - 1.0
            else:
                winnings = winnings - 1.0
        print("Bilans wygranych: " + str("{:.2f}".format(winnings)) + " dla klasy " + results_to_description_dict[idx])


# sprawdzanie wygranych dla obstawiania zakladow z najwiekszym prawdopodobienstwem (i roznica w podanym progu)
def show_winnings_within_threshold_only_highest_prob(classes_possibilities, actual_classes, odds, threshold):
    winnings = 0.0
    no_bet = 0
    outcome_possibilities = 1.0 / odds
    prediction_diff = classes_possibilities - outcome_possibilities
    chosen_class = classes_possibilities.argmax(axis=-1)
    for i in range(prediction_diff.shape[0]):
        # Jesli siec zdecydowala sie nie obstawiac meczu
        if prediction_diff[i][chosen_class[i]] < threshold:
            no_bet += 1
            continue
        elif chosen_class[i] == actual_classes[i]:
            winnings = winnings + odds[i][actual_classes[i]] - 1.0
        else:
            winnings = winnings - 1.0
    print("Bilans wygranych/strat z potencjalnych zakładów: " + str("{:.2f}".format(winnings)) + " na " + str(prediction_diff.shape[0]) + " meczow (" +
          str("{:.2f}%)".format(winnings / prediction_diff.shape[0] * 100)))


# sprawdzanie wygranych dla obstawiania zakladow z najwieksza roznica pomiedzy prawdopodobienstwem przewidzianym przez bukmachera a
# prawdopodobienstwem przewidzianym przez siec
def show_winnings_within_threshold_every_bet(classes_possibilities, actual_classes, odds, threshold):
    winnings = 0.0
    no_bet = 0
    outcome_possibilities = 1.0 / odds
    prediction_diff = classes_possibilities - outcome_possibilities
    prediction_diff = np.where(prediction_diff == np.amax(prediction_diff, axis=1, keepdims=1), prediction_diff, np.zeros_like(prediction_diff))
    for i in range(prediction_diff.shape[0]):
        for j in range(prediction_diff.shape[1]):
            # Jesli siec zdecydowala sie nie obstawiac meczu
            if prediction_diff[i][j] < threshold:
                no_bet += 1
                continue
            elif j == actual_classes[i]:
                winnings = winnings + odds[i][actual_classes[i]] - 1.0
            else:
                winnings = winnings - 1.0
    print("Bilans wygranych/strat z potencjalnych zakładów: " + str("{:.2f}".format(winnings)) + " na " + str(prediction_diff.shape[0]) + " meczow (" +
          str("{:.2f}%)".format(winnings / prediction_diff.shape[0] * 100)))


def show_accuracy_for_classes(predicted_classes, actual_classes):
    predicted_classes_as_int = predicted_classes
    actual_classes_as_int = actual_classes
    comparison_array = actual_classes_as_int == predicted_classes_as_int
    for i in np.unique(actual_classes_as_int):
        current_actual_class_indexes = [index for index, class_value in enumerate(actual_classes_as_int) if class_value == i]
        current_predicted_class_indexes = [index for index, class_value in enumerate(predicted_classes_as_int) if class_value == i]
        true_positives = sum(1 for comparison in comparison_array[current_actual_class_indexes] if comparison)
        false_positives = sum(1 for comparison in comparison_array[current_predicted_class_indexes] if not comparison)
        all_actual_class_examples = len(current_actual_class_indexes)
        all_predicted_class_examples = len(current_predicted_class_indexes)
        print("Procent odgadniętych przykładów na wszystkie przykłady z klasą \"" + results_to_description_dict[i]
              + "\" = {:.1f}".format(100 * true_positives / all_actual_class_examples if all_actual_class_examples != 0 else 0)
              + "% (" + str(true_positives) + "/" + str(all_actual_class_examples) + ")")
        print("Ilosc falszywie przewidzianych dla klasy \"" + results_to_description_dict[i]
              + "\" = {:.1f}".format(100 * false_positives / all_predicted_class_examples if all_predicted_class_examples != 0 else 0)
              + "% (" + str(false_positives) + "/" + str(all_predicted_class_examples) + ")")
    not_bet_logical = predicted_classes_as_int == Categories.NO_BET.value
    not_bet_sum = sum(1 for logic_1 in not_bet_logical if logic_1)
    all_classes_len = len(predicted_classes_as_int)
    print("Ilosc nieobstawionych zakladow = {:.1f}".format(100 * not_bet_sum / all_classes_len if all_actual_class_examples != 0 else 0)
          + "% (" + str(not_bet_sum) + "/" + str(all_classes_len) + ")")


def show_accuracy_within_threshold_only_highest_prob(classes_possibilities, actual_classes, odds, threshold):
    outcome_possibilities = 1.0 / odds
    prediction_diff = classes_possibilities - outcome_possibilities
    chosen_class = classes_possibilities.argmax(axis=-1)
    for index, c in enumerate(chosen_class):
        if prediction_diff[index, c] < threshold:
            chosen_class[index] = Categories.NO_BET.value
    show_accuracy_for_classes(chosen_class, actual_classes)


def show_accuracy_within_threshold_every_bet(classes_possibilities, actual_classes, odds, threshold):
    outcome_possibilities = 1.0 / odds
    prediction_diff = classes_possibilities - outcome_possibilities
    prediction_diff = np.where(prediction_diff == np.amax(prediction_diff, axis=1, keepdims=1), prediction_diff, np.zeros_like(prediction_diff))
    chosen_class = prediction_diff.argmax(axis=-1)
    for index, c in enumerate(chosen_class):
        if prediction_diff[index, c] < threshold:
            chosen_class[index] = Categories.NO_BET.value
    show_accuracy_for_classes(chosen_class, actual_classes)


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


def eval_model_after_learning(y_true, y_pred, odds):
    y_pred_classes = y_pred.argmax(axis=-1)
    y_true_classes = y_true.argmax(axis=-1)
    show_winnings(y_pred_classes, y_true_classes, odds)
    show_winnings_for_classes(y_pred_classes, y_true_classes, odds)
    show_accuracy_for_classes(y_pred_classes, y_true_classes)


def eval_model_after_learning_within_threshold(y_true, y_pred, odds, threshold):
    y_pred_classes = y_pred.argmax(axis=-1)
    y_true_classes = y_true.argmax(axis=-1)
    show_winnings_within_threshold_every_bet(y_pred, y_true_classes, odds, threshold)
    show_accuracy_within_threshold_every_bet(y_pred, y_true_classes, odds, threshold)


def get_debug_infos(x, y_pred, y_true, odds):
    df = pd.DataFrame(data=x, columns=get_curr_dataset_column_names())
    df = df.assign(predictions=pd.Series(y_pred),
                   true_labels=pd.Series(y_true),
                   odds_home=pd.Series(odds[:, 0]),
                   odds_draw=pd.Series(odds[:, 1]),
                   odds_away=pd.Series(odds[:, 2]))
    return df


def get_debug_infos_within_threshold(x, y_pred, y_true, threshold):
    odds = y_true[:, 4:7]
    y_classes = y_true[:, 0:3]
    outcome_possibilities = 1.0 / odds
    prediction_diff = y_pred - outcome_possibilities
    indexes_to_drop = np.where(np.all(prediction_diff < threshold, axis=1))
    x_val_dropped = np.delete(x, indexes_to_drop, axis=0)
    prediction_diff_dropped = np.delete(prediction_diff, indexes_to_drop, axis=0)
    prediction_classes = prediction_diff_dropped.argmax(axis=1)
    y_classes = np.delete(y_classes, indexes_to_drop, axis=0).argmax(axis=1)
    odds_dropped = np.delete(odds, indexes_to_drop, axis=0)
    ids = np.delete(load_ids_in_right_order(DatasetSplit.VAL), indexes_to_drop)
    debug_infos = get_debug_infos(x_val_dropped, prediction_classes, y_classes, odds_dropped).assign(match_ids=pd.Series(ids))
    return {'dataset_object': debug_infos, 'db_object': Match.select().where(Match.id << ids)}


def save_model(model, path):
    model.save(path, overwrite=True)


def load_model(path):
    return keras.models.load_model(path)