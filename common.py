import math
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

confidence_threshold = 0.02
results_to_description_dict = {0: 'Wygrana gospodarzy', 1: 'Remis', 2: 'Wygrana gości', 3: 'Brak zakładu'}


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
        concated_metrics = concated_metrics[concated_metrics < 30]
        avg = np.average(concated_metrics)
        std_dev = math.sqrt(np.sum(concated_metrics * concated_metrics) / len(concated_metrics) - avg ** 2)
        start = avg - 2 * std_dev
        end = avg + 2 * std_dev
        plt.ylim([start, end])
    plt.show()


def show_winnings(predicted_classes, actual_classes, odds):
    winnings = 0.0
    for i in range(predicted_classes.shape[0]):
        # Jesli siec zdecydowala sie nie obstawiac meczu
        # todo: czytelniej
        if predicted_classes[i] == Categories.NO_BET.value:
            continue
        elif predicted_classes[i] == actual_classes[i]:
            winnings = winnings + odds[i][actual_classes[i]] - 1.0
        else:
            winnings = winnings - 1.0
    print("Bilans wygranych/strat z potencjalnych zakładów: " + str(winnings))


def show_winnings_within_threshold(classes_possibilities, actual_classes, odds):
    winnings = 0.0
    no_bet = 0
    outcome_possibilities = 1.0 / odds
    prediction_diff = classes_possibilities - outcome_possibilities
    chosen_class = classes_possibilities.argmax(axis=-1)
    for i in range(prediction_diff.shape[0]):
        # Jesli siec zdecydowala sie nie obstawiac meczu
        if prediction_diff[i][chosen_class[i]] < confidence_threshold:
            no_bet += 1
            continue
        elif chosen_class[i] == actual_classes[i]:
            winnings = winnings + odds[i][actual_classes[i]] - 1.0
        else:
            winnings = winnings - 1.0
    print("Bilans wygranych/strat z potencjalnych zakładów: " + str(winnings))
    print("Ilosc nieobstawionych zakładów z powodu zbyt niskiej pewnosci: " + str(no_bet))


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


def show_accuracy_within_threshold(classes_possibilities, actual_classes, odds):
    outcome_possibilities = 1.0 / odds
    prediction_diff = classes_possibilities - outcome_possibilities
    chosen_class = classes_possibilities.argmax(axis=-1)
    for index, c in enumerate(chosen_class):
        if prediction_diff[index, c] < confidence_threshold:
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
    show_accuracy_for_classes(y_pred_classes, y_true_classes)


def eval_model_after_learning_within_threshold(y_true, y_pred, odds):
    y_pred_classes = y_pred.argmax(axis=-1)
    y_true_classes = y_true.argmax(axis=-1)
    show_winnings_within_threshold(y_pred, y_true_classes, odds)
    show_accuracy_within_threshold(y_pred, y_true_classes, odds)
