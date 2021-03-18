import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split

def main():
    train_data = pd.read_csv('data/train.csv', header=0).values
    test_data = pd.read_csv('data/test.csv', header=0).values
    y_train = np.array([int(x[0]) for x in train_data])

    # Normalise values
    X_train = np.array([x[1:] for x in train_data]) / 255.0
    X_test = np.array([x for x in test_data]) / 255.0

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(784),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', metrics=['accuracy'],
                  loss=SparseCategoricalCrossentropy(from_logits=True))

    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    predictions = np.argmax(model.predict(X_test), axis=-1)
    submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
                                "Label": predictions})
    submissions.to_csv("mnist_tfkeras.csv", index=False)


if __name__ == '__main__':
    main()
