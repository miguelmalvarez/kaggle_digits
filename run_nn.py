import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    train_data = pd.read_csv('data/train.csv', header=0).values
    test_data = pd.read_csv('data/test.csv', header=0).values

    # Code to do some sampling for quick experimentation
    sample = 100000
    train_data = train_data[:sample]
    test_data = test_data[:sample]
    y_all_train = np.array([[int(x[0])] for x in train_data])
    # --------

    epochs = 50
    batch_size = 250

    # Normalise values and create validation set
    x_test = np.array([x for x in test_data]) / 255.0
    x_all_train = np.array([x[1:] for x in train_data]) / 255.0
    x_train, x_val, y_train, y_val = train_test_split(x_all_train, y_all_train, test_size=0.2)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(784, 1)),
        tf.keras.layers.Conv2D(8, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.1),
        # Fully connected after this
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    predictions = np.argmax(model.predict(x_test), axis=-1)

    submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
                                "Label": predictions})
    submissions.to_csv("mnist_tfkeras.csv", index=False)
    model.summary()


if __name__ == '__main__':
    main()
