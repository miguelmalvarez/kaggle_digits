import tensorflow as tf
import numpy as np
import pandas as pd

def main():
    train_data = pd.read_csv('data/train.csv', header=0).values
    test_data = pd.read_csv('data/test.csv', header=0).values

    # Code to do some sampling for quick experimentation
    sample = 50000
    train_data = train_data[:sample]
    test_data = test_data[:sample]
    y_train = np.array([int(x[0]) for x in train_data[:sample]])
    # --------

    # Normalise values
    x_train = np.array([x[1:] for x in train_data]) / 255.0
    x_test = np.array([x for x in test_data]) / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(784),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)
    predictions = np.argmax(model.predict(x_test), axis=-1)

    submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
                                "Label": predictions})
    submissions.to_csv("mnist_tfkeras.csv", index=False)


if __name__ == '__main__':
    main()
