import math
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from azureml.core import Run

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from sklearn.preprocessing import minmax_scale

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
from tensorflow.keras.models import save_model
from tensorflow_addons.optimizers import AdamW

from eval import recall_at_k
from model.create_model_1dcnn import create_model
from callbacks import LogToAzure

seed = 24
tf.random.set_seed(seed)
np.random.seed(seed)

def slice_by_index(X, idx):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    elif isinstance(X, np.ndarray):
        return np.take(X, idx, axis=0)
    else:
        raise(TypeError("Please pass pandas dataframe or numpy array"))

def grouped_train_test_split(X, y, groups, test_size=0.2):
    """
    Train/test split which doesn't split up groups
    https://stackoverflow.com/questions/61337373/split-on-train-and-test-separating-by-group
    """  
    gs = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=42)
    train_ix, test_ix = next(gs.split(X, y, groups=groups))
    return slice_by_index(X, train_ix), slice_by_index(X, test_ix), slice_by_index(y, train_ix), slice_by_index(y, test_ix)

def make_X_y(df):
    """
    Split into gene expression values (X) and perturbagen labels (y)
    """
    df = df.drop(['pert_dose', 
    'pert_dose_unit', 
    'pert_id', 
    'pert_mfc_id',
    'pert_time', 
    'pert_time_unit', 
    'pert_type', 
    'rna_plate', 
    'rna_well',
    'cell_id',
    'det_plate',
    'det_well'
    ], axis=1)

    X = df.loc[:, df.columns != 'pert_iname']
    y = df['pert_iname']
    return (X, y)

def encode_y(y_train):
    """
    Encode pertubagen labels as integers
    """
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y_train = encoder.transform(y_train)
    return encoded_y_train

def scale_X(X_train, X_test):
    """
    Standardise gene expression values
    Same transform is applied to test data to prevent data leakage
    """
    X_scaler = StandardScaler()
    X_scaler.fit(X_train)
    X_train_scale = X_scaler.transform(X_train)
    X_test_scale = X_scaler.transform(X_test)
    return (X_train_scale, X_test_scale)

def quantile_X(X_train, X_test):
    """
    Quantile transformer
    Same transform is applied to test data to prevent data leakage
    """
    qt = QuantileTransformer(n_quantiles=100, output_distribution='normal')
    qt.fit(X_train)
    X_train_scale = qt.transform(X_train)
    X_test_scale = qt.transform(X_test)
    return (X_train_scale, X_test_scale)

def run_training():

    run = Run.get_context()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, dest='data_dir', default='data', help='data folder mounting point')
    parser.add_argument('--data-file', type=str, dest='data_file', default='clean.parquet', help='clean data file')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=32, help='mini batch size for training')
    parser.add_argument('--epochs', type=int, dest='epochs', default=5, help='number of epochs to run for training')
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01, help='learning rate')
    parser.add_argument('--weight-decay', type=float, dest='weight_decay', default=1e-3, help='weight decay')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='location of the model or checkpoint files from where to resume the training')
    args = parser.parse_args()

    data_dir = args.data_dir
    data_file = args.data_file
    data_file_path = Path(data_dir, data_file)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    df = pd.read_parquet(data_file_path)

    X, y = make_X_y(df)
    X_train, X_test, y_train, y_test = grouped_train_test_split(X, y, y, test_size=0.2)
    X_train_scale, X_test_scale = scale_X(X_train, X_test)
    y_train_encode = encode_y(y_train)

    #create train dataset object
    train_dataset = Dataset.from_tensor_slices((X_train_scale, y_train_encode))
    label_dataset = Dataset.from_tensor_slices(y_train_encode)
    dataset = Dataset.zip((train_dataset, label_dataset)).shuffle(100).batch(batch_size)
        
    num_classes = len(np.unique(y_train))

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = create_model(num_classes, batch_size, input_length=X.shape[1], embedding_size=32, max_m=0.25)
        model.compile(loss='SparseCategoricalCrossentropy',
                        optimizer=AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
                        metrics=[metrics.sparse_categorical_accuracy])
    model.fit(dataset, 
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[LogToAzure(run)],
            verbose=1)

    save_model(model, './outputs/saved_model/')

    #eval model
    embeddings_model = Model(inputs=model.get_layer('gene_expression_vector').input, outputs=model.get_layer('lambda').output) #get network up to and inc l2norm of embedding layer
    embedded = embeddings_model.predict(X_test_scale)
    _, embedded_sample, _, labs_sample = grouped_train_test_split(embedded, y_test, y_test, test_size=250)

    recall = recall_at_k(embedded_sample, embedded, y_test)
    quantile = minmax_scale(np.arange(1, embedded.shape[0]), feature_range=(0, 1))
    auc = np.trapz(recall, quantile)

    fig, ax = plt.subplots()
    ax.plot(quantile, recall)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    auc_lab = f"AUC {auc:.2f}"
    ax.text(0.73, 0.1, auc_lab, transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', bbox=props)
    plt.title("Compound Retrieval for Embedded Signatures in Test Set")
    plt.xlabel("Proportion of Results Included")
    plt.ylabel("Proportion of Compound Instances Identified")
    plt.show()
    
    metrics_path = Path("./outputs/metrics")
    metrics_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(metrics_path / "quantile_recall.png")
    run.log_image('quantile_recall', plot=plt)

if __name__ == "__main__":
    run_training()
