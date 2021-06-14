# RA, 2021-06-14

import tensorflow


class FunctionalLogger(tensorflow.keras.callbacks.Callback):
    def __init__(self, ff: dict):
        super().__init__()
        self.ff = ff
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for (k, f) in self.ff.items():
                logs[k] = f(self.model)


class ConditionalAbort(tensorflow.keras.callbacks.Callback):
    def __init__(self, predicate):
        super().__init__()
        self.predicate = predicate

    def on_epoch_end(self, epoch, logs=None):
        if (logs is not None):
            self.model.stop_training |= self.predicate({'epoch': epoch, **logs})
