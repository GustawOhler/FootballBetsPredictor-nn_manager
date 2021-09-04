import keras_tuner as kt
import constants
from dataset_manager.dataset_manager import get_splitted_dataset
from constants import DatasetType, VALIDATION_TO_TRAIN_SPLIT_RATIO, TEST_TO_VALIDATION_SPLIT_RATIO
import keras_tuner.engine.trial as trial_lib


class CustomBayesianSearch(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        # 32, 64, 128,
        # kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', [64, 128, 256, 512])
        valid_model = True
        try:
            for i in range(1, trial.hyperparameters.get('layers_quantity')):
                if trial.hyperparameters.get(f'number_of_neurons_{i}_layer') > 2 * trial.hyperparameters.get(f'number_of_neurons_{i-1}_layer'):
                    valid_model = False
                    break
        except:
            pass
        if valid_model:
            if not constants.is_model_rnn:
                dataset_option = trial.hyperparameters.Choice('dataset', [dsoption.value for dsoption in DatasetType])
                constants.curr_dataset_name = dataset_option
                dataset = get_splitted_dataset(False, False, VALIDATION_TO_TRAIN_SPLIT_RATIO, TEST_TO_VALIDATION_SPLIT_RATIO)
                kwargs['x'] = dataset[0][0]
                kwargs['y'] = dataset[0][1]
                kwargs['validation_data'] = dataset[1]
                kwargs['validation_batch_size'] = dataset[1][1].shape[0]
            super(CustomBayesianSearch, self).run_trial(trial, *args, **kwargs)
        else:
            print("Skipped model")
            self.oracle.end_trial(trial.trial_id, trial_lib.TrialStatus.INVALID)

    def on_trial_end(self, trial):
        if trial.status != trial_lib.TrialStatus.INVALID:
            super(CustomBayesianSearch, self).on_trial_end(trial)
