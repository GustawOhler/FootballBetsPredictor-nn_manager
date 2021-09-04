import numpy as np
from sklearn.model_selection import KFold
from constants import is_model_rnn


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