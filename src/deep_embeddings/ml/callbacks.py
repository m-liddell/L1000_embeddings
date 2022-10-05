import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class LogToAzure(Callback):
    """
    Keras Callback for realtime logging to Azure ML
    """
    def __init__(self, run):
        super(LogToAzure, self).__init__()
        self.run = run

    def on_train_batch_end(self, batch, logs=None):
        #log all log data to Azure
        for k, v in logs.items():
            self.run.log(k, v)