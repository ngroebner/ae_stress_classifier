import pickle
import argparse
import io
import os

import mlflow
import boto3
import numpy as np

import matplotlib.pyplot as plt
#from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .network import make_network


def load_npy(bucket, key):

    obj = boto3.resource("s3").Object(bucket, key)
    with io.BytesIO(obj.get()["Body"].read()) as f:
        f.seek(0)  # rewind the file
        X, y = np.load(f)

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
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="Mini batch size")
    parser.add_argument("--epochs",
                        default=1,
                        type=int,
                        help="number of epochs to train")
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="Learning rate for optimizer")

    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ("MLFLOW_SERVER"))
    mlflow.set_experiment(args.expname)
    run = mlflow.start_run()

    # log parameters
    mlflow.log_params({
        #"data file": args.bucket+"/"+args.data,
        #"target file":args.bucket+"/"+args.target,
        "residual blocks": args.resblocks,
        "epochs": args.epochs,
        "learning rate": args.learning_rate,
        "batch size": args.batch_size
    })

    opt = Adam(learning_rate=args.learning_rate)

    # load training and test data
    # get data from where it's gonna be
    X = load_npy(args.bucket, args.data)
    y = load_npy(args.bucket, args.labels)

    # scale the data
    y_scaler = StandardScaler().fit(y)
    y_scaled = y_scaler.transform(y)

    X_test, X_train, y_test, y_train = train_test_split(X,y_scaled,train_size=0.7)

    model = make_network(input_shape=(X.shape(1),1), nblocks=args.resblocks)
    model.compile(
        optimizer=opt,
        loss="mse",
        metrics="mse",
    )

    model.fit(
            x=X_train,
            y=y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            callbacks=None,
            validation_split=0.1,
    )

    score = model.evaluate(X_test, y_test)
    # plot results for test and train
    y_test_predicted = y_scaler.inverse_transform(model.predict(X_test))
    np.save("artifacts/y_test_predicted.npy", y_test_predicted)
    y_train_predicted = y_scaler.inverse_transform(model.predict(X_train))
    np.save("artifacts/y_train_predicted.npy", y_train_predicted)

    f = plt.figure(figsize=(7,7))
    plt.scatter(X_train, y_train_predicted, c="orange", label="Training set")
    plt.scatter(X_test, y_test_predicted, c="blue", alpha=0.3, label="Test set")
    plt.plot([0,130],[0,130], label="Perfect prediction")
    # regression of test data
    b, m = np.polyfit(X_test, y_test_predicted, deg=1)
    x_fit = np.arange(0,130)
    plt.line(x_fit, m*x_fit+ b)
    plt.xlim([0,130])
    plt.ylim([0,130])
    plt.xlabel("True Differential Stress (MPa)")
    plt.ylabel("Predicted Differential Stress (MPa)")
    plt.title(f"Neural Network")

    plt.savefig("artifacts/results.png")

    mlflow.log_artifacts("artifacts/")
    # log all the parameters and artifacts

    mlflow.end_run()