import math

import numpy as np
import pandas as pd
from flatten_dict import flatten
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
from constants import is_model_rnn, DatasetType, ModelType, ChoosingBetsStrategy, PredMatchesStrategy, best_model
from dataset_manager.class_definitions import DatasetSplit
from dataset_manager.dataset_manager import get_whole_dataset, get_dataset_ready_to_learn, get_splitted_dataset
from nn_manager.nn_pred_matches_manager import NNPredictingMatchesManager
from nn_manager.recurrent_nn_pred_matches_manager import RecurrentNNPredictingMatchesManager
from nn_manager.nn_choose_bets_menager import NNChoosingBetsManager
from nn_manager.recurrent_nn_choose_bets_manager import RecurrentNNChoosingBetsManager


def perform_standard_k_fold(X, y, nn_manager):
    k_folder = KFold(n_splits=10, shuffle=True)
    metrics = []
    metrics_names = []
    loop_index = 1
    for train_index, val_index in k_folder.split(y):
        print("Rozpoczynam uczenie modelu nr " + str(loop_index), end="\r")
        loop_index += 1
        if is_model_rnn:
            X_train, X_val = [X[i][train_index] for i in range(len(X))], [X[i][val_index] for i in range(len(X))]
        else:
            X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        curr_nn_manager = nn_manager((X_train, y_train), (X_val, y_val), False)
        curr_nn_manager.perform_model_learning(verbose=False)
        metrics.append(curr_nn_manager.model.evaluate(X_val, y_val, verbose=1))
        if len(metrics_names) == 0:
            metrics_names = curr_nn_manager.model.metrics_names
    mean_metrics = np.asarray(metrics).mean(axis=0)
    print("Srednie wyniki dla modelu: ")
    for i, name in enumerate(metrics_names):
        print(name + ": " + str(mean_metrics[i]))


def perform_k_fold_with_different_parameters(X, y, nn_manager):
    k_folder = KFold(n_splits=10, shuffle=True)
    variant_possible = [0, 3, 6, 7, 9]
    metrics = [[] for i in range(len(variant_possible))]
    metrics_names = []
    loop_index = 1
    for train_index, val_index in k_folder.split(y):
        print("Rozpoczynam uczenie modelu nr " + str(loop_index), end="\r")
        loop_index += 1
        if is_model_rnn:
            X_train, X_val = [X[i][train_index] for i in range(len(X))], [X[i][val_index] for i in range(len(X))]
        else:
            X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        for i, variant in enumerate(variant_possible):
            curr_nn_manager = nn_manager((X_train, y_train), (X_val, y_val), False, None, variant)
            curr_nn_manager.perform_model_learning(verbose=False)
            metrics[i].append(curr_nn_manager.model.evaluate(X_val, y_val, batch_size=y_val.shape[0], verbose=0))
            if len(metrics_names) == 0:
                metrics_names = curr_nn_manager.model.metrics_names
    mean_metrics = np.empty((0, len(metrics_names) + 1), dtype=float)
    for idx, variant in enumerate(variant_possible):
        mean_metrics = np.append(mean_metrics, np.concatenate((np.array([[variant]]), np.asarray(metrics[idx]).mean(axis=0).reshape(1, 3)), axis=1), axis=0)
    mean_metrics_sorted = mean_metrics[(-mean_metrics)[:, -1].argsort()]
    for idx in range(mean_metrics_sorted.shape[0]):
        print(f"Srednie wyniki dla {idx + 1}. najlepszego wyniku czyli wariantu nr {int(mean_metrics_sorted[idx, 0])}:")
        for i, name in enumerate(metrics_names):
            print(name + ": " + "{:.4f}".format(mean_metrics_sorted[idx, 1 + i]))


def perform_k_fold_with_different_datasets(nn_manager):
    rep_k_folder = RepeatedKFold(n_repeats=3, n_splits=10)
    possible_datasets_types = [dsoption for dsoption in DatasetType]
    possible_datasets = {ds_type: get_whole_dataset(False, ds_type) for ds_type in possible_datasets_types}
    tracked_metrics = {ds_type: {} for ds_type in possible_datasets_types}
    base_ids = possible_datasets[DatasetType.BASIC]['match_id']
    loop_index = 1
    for train_index, val_index in rep_k_folder.split(base_ids):
        print("Rozpoczynam uczenie dla próbek nr " + str(loop_index), end="\r")
        loop_index += 1
        train_ids, val_ids = base_ids.iloc[train_index], base_ids.iloc[val_index]
        for ds_type, ds in possible_datasets.items():
            train_ds, val_ds = ds.loc[ds['match_id'].isin(train_ids)], ds.loc[ds['match_id'].isin(val_ids)]
            train_x, train_y = get_dataset_ready_to_learn(train_ds, DatasetSplit.WHOLE, False, False)
            val_x, val_y = get_dataset_ready_to_learn(val_ds, DatasetSplit.WHOLE, False, False)
            curr_nn_manager = nn_manager((train_x, train_y), (val_x, val_y), False, None)
            curr_nn_manager.perform_model_learning(verbose=False)
            if len(tracked_metrics[ds_type]) == 0:
                tracked_metrics[ds_type].update({metric_name: [] for metric_name in curr_nn_manager.model.metrics_names})
            for index, metric_name in enumerate(curr_nn_manager.model.metrics_names):
                tracked_metrics[ds_type][metric_name].append(curr_nn_manager.model.evaluate(val_x, val_y, batch_size=val_y.shape[0], verbose=0)[index])
    return tracked_metrics


def check_if_model_is_rnn(model_type: ModelType):
    return model_type.value in [ModelType.RNN.value, ModelType.GRU.value, ModelType.LSTM.value, ModelType.GRU_pred_matches.value,
                                ModelType.RNN_pred_matches.value, ModelType.LSTM_pred_matches.value]


def perform_k_fold_with_different_models():
    rep_k_folder = RepeatedStratifiedKFold(n_repeats=1, n_splits=3)
    ffnn_dataset = get_whole_dataset(False, DatasetType.BASIC)
    recurrent_dataset = get_whole_dataset(False, DatasetType.SEPARATED_MATCHES)
    possible_models = {ModelType.RNN: RecurrentNNChoosingBetsManager, ModelType.RNN_pred_matches: RecurrentNNPredictingMatchesManager}
    tracked_metrics = {model_type: {} for model_type in possible_models}
    base_ids = ffnn_dataset['match_id']
    whole_x, whole_y = recurrent_dataset.drop('result', axis='columns'), recurrent_dataset[['result']]
    loop_index = 1
    for train_index, val_index in rep_k_folder.split(whole_x, whole_y):
        print("Rozpoczynam uczenie dla próbek nr " + str(loop_index), end="\r")
        loop_index += 1
        train_ids, val_ids = base_ids.iloc[train_index], base_ids.iloc[val_index]
        for model_type, model_constructor in possible_models.items():
            if check_if_model_is_rnn(model_type):
                train_ds, val_ds = recurrent_dataset.loc[recurrent_dataset['match_id'].isin(train_ids)], recurrent_dataset.loc[
                    recurrent_dataset['match_id'].isin(val_ids)]
                train_x, train_y = get_dataset_ready_to_learn(train_ds, DatasetSplit.TRAIN, True, False)
                val_x, val_y = get_dataset_ready_to_learn(val_ds, DatasetSplit.VAL, True, False)
            else:
                train_ds, val_ds = ffnn_dataset.loc[ffnn_dataset['match_id'].isin(train_ids)], ffnn_dataset.loc[ffnn_dataset['match_id'].isin(val_ids)]
                train_x, train_y = get_dataset_ready_to_learn(train_ds, DatasetSplit.WHOLE, False, False)
                val_x, val_y = get_dataset_ready_to_learn(val_ds, DatasetSplit.WHOLE, False, False)
            curr_nn_manager = model_constructor((train_x, train_y), (val_x, val_y), False, None)
            curr_nn_manager.perform_model_learning(verbose=False)
            if len(tracked_metrics[model_type]) == 0:
                tracked_metrics[model_type].update({metric_name: [] for metric_name in curr_nn_manager.model.metrics_names})
            for index, metric_name in enumerate(curr_nn_manager.model.metrics_names):
                tracked_metrics[model_type][metric_name].append(curr_nn_manager.model.evaluate(val_x, val_y, batch_size=val_y.shape[0], verbose=0)[index])
    return tracked_metrics


def perform_k_fold_on_last_2():
    rep_k_folder = RepeatedStratifiedKFold(n_repeats=5, n_splits=10)
    recurrent_dataset = get_whole_dataset(False, DatasetType.SEPARATED_MATCHES)
    possible_models = {ModelType.RNN: RecurrentNNChoosingBetsManager, ModelType.RNN_pred_matches: RecurrentNNPredictingMatchesManager}
    tracked_metrics = {model_type: {} for model_type in possible_models}
    whole_x, whole_y = recurrent_dataset.drop('result', axis='columns'), recurrent_dataset[['result']]
    loop_index = 1
    for train_index, val_index in rep_k_folder.split(whole_x, whole_y):
        print("Rozpoczynam uczenie dla próbek nr " + str(loop_index), end="\r")
        loop_index += 1
        for model_type, model_constructor in possible_models.items():
            train_ds, val_ds = recurrent_dataset.iloc[train_index], recurrent_dataset.iloc[val_index]
            train_x, train_y = get_dataset_ready_to_learn(train_ds, DatasetSplit.TRAIN, True, False)
            val_x, val_y = get_dataset_ready_to_learn(val_ds, DatasetSplit.VAL, True, False)
            curr_nn_manager = model_constructor((train_x, train_y), (val_x, val_y), False, None)
            curr_nn_manager.perform_model_learning(verbose=False)
            if len(tracked_metrics[model_type]) == 0:
                tracked_metrics[model_type].update({metric_name: [] for metric_name in curr_nn_manager.model.metrics_names})
            for index, metric_name in enumerate(curr_nn_manager.model.metrics_names):
                tracked_metrics[model_type][metric_name].append(curr_nn_manager.model.evaluate(val_x, val_y, batch_size=val_y.shape[0], verbose=0)[index])
    return tracked_metrics

def perform_test_check_on_last_2():
    possible_models = {ModelType.RNN: RecurrentNNChoosingBetsManager, ModelType.RNN_pred_matches: RecurrentNNPredictingMatchesManager}
    tracked_metrics = {model_type: {} for model_type in possible_models}
    for loop_index in range(1, 11):
        print("Rozpoczynam uczenie dla próbek nr " + str(loop_index), end="\r")
        datasets = get_splitted_dataset(False, True, 0.1, 0.5)
        train_set = datasets[0]
        val_set = datasets[1]
        test_set = datasets[2]
        for model_type, model_constructor in possible_models.items():
            curr_nn_manager = model_constructor(train_set, val_set, False, test_set)
            curr_nn_manager.perform_model_learning(verbose=False)
            if len(tracked_metrics[model_type]) == 0:
                tracked_metrics[model_type].update({metric_name: {'val': [], 'test': []} for metric_name in curr_nn_manager.model.metrics_names})
            for index, metric_name in enumerate(curr_nn_manager.model.metrics_names):
                tracked_metrics[model_type][metric_name]['val'].append(curr_nn_manager.model.evaluate(val_set[0], val_set[1], batch_size=val_set[1].shape[0],
                                                                                                     verbose=0)[index])
                tracked_metrics[model_type][metric_name]['test'].append(curr_nn_manager.model.evaluate(test_set[0], test_set[1],
                                                                                                      batch_size=test_set[1].shape[0],
                                                                                                      verbose=0)[index])
    return tracked_metrics

def perform_k_fold_on_expotential(nn_manager, dataset):
    rep_k_folder = RepeatedKFold(n_repeats=3, n_splits=5)
    possibilities_dict = {'should_add_expotential': [False, True]}
    tracked_metrics = {should_add_expotential_value: {} for should_add_expotential_value in possibilities_dict['should_add_expotential']}
    loop_index = 1
    for train_index, val_index in rep_k_folder.split(dataset):
        print("Rozpoczynam uczenie dla próbek nr " + str(loop_index), end="\r")
        loop_index += 1
        train_ds, val_ds = dataset.iloc[train_index], dataset.iloc[val_index]
        train_x, train_y = get_dataset_ready_to_learn(train_ds, DatasetSplit.TRAIN, True, False)
        val_x, val_y = get_dataset_ready_to_learn(val_ds, DatasetSplit.VAL, True, False)
        for should_add_expotential_value in possibilities_dict['should_add_expotential']:
            curr_nn_manager = nn_manager((train_x, train_y), (val_x, val_y), False, None, should_add_expotential=should_add_expotential_value)
            curr_nn_manager.perform_model_learning(verbose=False)
            if len(tracked_metrics[should_add_expotential_value]) == 0:
                tracked_metrics[should_add_expotential_value].update({metric_name: [] for metric_name in curr_nn_manager.model.metrics_names})
            for index, metric_name in enumerate(curr_nn_manager.model.metrics_names):
                tracked_metrics[should_add_expotential_value][metric_name].append(curr_nn_manager.model.evaluate(val_x, val_y, batch_size=val_y.shape[0],
                                                                                                                 verbose=0)[index])
    return tracked_metrics


def perform_k_fold_for_different_strategies(nn_manager, dataset, is_rnn, is_for_choosing_bets):
    rep_k_folder = RepeatedKFold(n_repeats=3, n_splits=10)
    if is_for_choosing_bets:
        possible_strategies = [strat for strat in ChoosingBetsStrategy]
        tracked_metrics = {strat: {} for strat in possible_strategies}
    else:
        possible_strategies = [strat for strat in PredMatchesStrategy]
        tracked_metrics = {strat: [] for strat in possible_strategies}
    loop_index = 1
    if is_for_choosing_bets:
        for train_index, val_index in rep_k_folder.split(dataset):
            print("Rozpoczynam uczenie dla próbek nr " + str(loop_index), end="\r")
            loop_index += 1
            train_ds, val_ds = dataset.iloc[train_index], dataset.iloc[val_index]
            train_x, train_y = get_dataset_ready_to_learn(train_ds, DatasetSplit.TRAIN, is_rnn, False)
            val_x, val_y = get_dataset_ready_to_learn(val_ds, DatasetSplit.VAL, is_rnn, False)
            for strategy in possible_strategies:
                curr_nn_manager = nn_manager((train_x, train_y), (val_x, val_y), False, None, strategy=strategy)
                curr_nn_manager.perform_model_learning(verbose=False)
                if len(tracked_metrics[strategy]) == 0:
                    tracked_metrics[strategy].update({metric_name: [] for metric_name in curr_nn_manager.model.metrics_names})
                for index, metric_name in enumerate(curr_nn_manager.model.metrics_names):
                    tracked_metrics[strategy][metric_name].append(curr_nn_manager.model.evaluate(val_x, val_y, batch_size=val_y.shape[0],
                                                                                                 verbose=0)[index])
    else:
        for train_index, val_index in rep_k_folder.split(dataset):
            print("Rozpoczynam uczenie dla próbek nr " + str(loop_index), end="\r")
            loop_index += 1
            train_ds, val_ds = dataset.iloc[train_index], dataset.iloc[val_index]
            train_x, train_y = get_dataset_ready_to_learn(train_ds, DatasetSplit.TRAIN, is_rnn, False)
            val_x, val_y = get_dataset_ready_to_learn(val_ds, DatasetSplit.VAL, is_rnn, False)
            curr_nn_manager = nn_manager((train_x, train_y), (val_x, val_y), False, None)
            curr_nn_manager.perform_model_learning(verbose=False)
            # if len(tracked_metrics[strategy]) == 0:
            #     tracked_metrics[strategy].update({metric_name: [] for metric_name in curr_nn_manager.model.metrics_names})
            for strategy, result in curr_nn_manager.get_best_strategies_value().items():
                tracked_metrics[strategy].append(result)
    return tracked_metrics


def search_for_best_configuration(datasets):
    model_to_search = best_model
    best_profit = -math.inf
    train_set = datasets[0]
    val_set = datasets[1]
    test_set = datasets[2]
    for loop_index in range(1, 16):
        print("Rozpoczynam uczenie dla próbek nr " + str(loop_index))
        curr_nn_manager = (globals()[model_to_search.value])(train_set, val_set, False, test_set)
        curr_nn_manager.perform_model_learning(verbose=False)
        profit_index = curr_nn_manager.model.metrics_names.index('profit')
        curr_model_profit = curr_nn_manager.model.evaluate(val_set[0], val_set[1], batch_size=val_set[1].shape[0], verbose=0)[profit_index]
        if curr_model_profit > best_profit:
            best_profit = curr_model_profit
            curr_nn_manager.save_best_weights()
            print(f"Model {model_to_search.value} zapisany z zyskiem {best_profit}")
    print(f"Najlepszy zysk: {best_profit}")


def print_results_to_csv(tracked_metrics: dict, choosed_path: str):
    pandas_df = pd.DataFrame(flatten(tracked_metrics, reducer='underscore'))
    pandas_df.to_csv(choosed_path, index=False, float_format='%.6f')
