import pickle
import argparse

from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from .network import make_network

def load(x):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="path to data")
    parser.add_argument("--labels", help="path to labels")
    parser.add_argument("--checkpoints", help="path to checkpoint save")
    parser.add_argument("--model", help="path to model save")
    parser.add_argument("--epochs", help="number of epochs to train")

    args = parser.parse_args()

    # load training and test data
    # get data from where it's gonna be
    X = load(args.data)
    y = load(args.labels)

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
