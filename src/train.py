import pickle
import argparse
import os

from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from .network import make_network
import mlflow

def load_npy(x):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--expname",
                        default="waveform-dry",
                        help="Name of mlflow experiment.")
    parser.add_argument("--bucket",
                        default="seismic_processing",
                        help="S3 bucket for data")
    parser.add_argument("--data",
                        default="ae_stress_classifier/data/dry_waves.npy",
                        help="training data object name")
    parser.add_argument("--target",
                        default="ae_stress_classifier/data/dry_stress.npy",
                        help="training target object name")
    parser.add_argument("--checkpoints", help="path to checkpoint save")
    parser.add_argument("--model", help="path to model save")
    parser.add_argument("--resblocks",
                        default=15,
                        type=int,
                        help="number of residual blocks in network")
    parser.add_argument("--epochs",
                        default=1,
                        type=int,
                        help="number of epochs to train")

    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ("MLFLOW_SERVER"))
    mlflow.set_experiment(args.experiment)
    run = mlflow.start_run()

    # load training and test data
    # get data from where it's gonna be
    X = load_npy(args.data)
    y = load_npy(args.labels)

    # scale the data

    X_test, X_train, y_test, y_train = train_test_split(X,y,train_size=0.7)

    model = make_network()
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics="mse",
    )

    model.fit(
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=2,
            callbacks=None,
            validation_split=0.1,
    )

    # log all the parameters and artifacts

    mlflow.end_run()