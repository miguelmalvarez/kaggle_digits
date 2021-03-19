import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42

def main():
    train_data = pd.read_csv('data/train.csv', header=0).values
    test_data = pd.read_csv('data/test.csv', header=0).values

    # Code to do some sampling for quick experimentation
    sample = 1000000
    train_data = train_data[:sample]
    test_data = test_data[:sample]
    y_all_train = np.array([int(x[0]) for x in train_data])
    # --------

    epochs = 30
    batch_size = 250

    # Normalise values and create validation set
    X_test = np.array([x for x in test_data]) / 255.0
    X_all_train = np.array([x[1:] for x in train_data]) / 255.0
    # Reshape data
    X_all_train = X_all_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    X_train, X_val, y_train, y_val = train_test_split(X_all_train, y_all_train, test_size=0.1, random_state=SEED)

    model = tf.keras.models.Sequential([
        Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        MaxPool2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(0.2),

        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPool2D(pool_size=(2,2)),
        BatchNormalization(),
        Dropout(0.2),

        # Fully connected after this
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        Dense(10, activation='softmax')
    ])
    print(model.summary())

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Data Augmentation
    aug = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1)
    aug.fit(X_train)

    # Early-stop function
    earlystopper = EarlyStopping(patience=3)

    model.fit(aug.flow(X_train, y_train, batch_size=batch_size),
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_val, y_val),
              callbacks=[earlystopper])

    predictions = np.argmax(model.predict(X_test), axis=-1,)
    submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
                                "Label": predictions})
    submissions.to_csv("mnist_tfkeras.csv", index=False)


if __name__ == '__main__':
    main()
