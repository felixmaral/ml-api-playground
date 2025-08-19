# train.py
import json, tensorflow as tf
from tensorflow import keras

def build_model(num_classes=10, input_shape=(32,32,3)):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1./255)(inputs)

    # Bloque 1
    x = keras.layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D((2,2))(x)

    # Bloque 2
    x = keras.layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D((2,2))(x)

    # Bloque 3
    x = keras.layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Fully Connected
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)

def main():
    (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
    ytr = ytr.squeeze(); yte = yte.squeeze()

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(xtr, ytr,
              epochs=20,
              batch_size=128,
              validation_split=0.1,
              verbose=1)

    model.evaluate(xte, yte, verbose=1)

    model.save("model.keras")

    labels = {i: name for i, name in enumerate(
        ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    )}
    with open("labels.json","w") as f: json.dump(labels, f)
    print("Saved model.keras and labels.json")

if __name__ == "__main__":
    main()