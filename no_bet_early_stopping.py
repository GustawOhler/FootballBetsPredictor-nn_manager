import numpy as np
from keras.callbacks import Callback


class NoBetEarlyStopping(Callback):
    def __init__(self, patience=0):
        super(NoBetEarlyStopping, self).__init__()
        self.patience = patience

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_profit = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        v_no_bets = logs.get("val_how_many_no_bets")
        v_profit = logs.get("val_profit")
        if v_no_bets > 95.0:
            if v_profit > self.best_profit:
                self.best_profit = v_profit
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    self.stopped_epoch = epoch
        elif self.wait > 0:
            self.wait = 0
            self.best_profit = -np.Inf


    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
