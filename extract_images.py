import os
import numpy as np
from tensorflow import keras
from PIL import Image

def main():
    # Cargar dataset CIFAR-10
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()

    # Crear carpeta destino
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    # Definir nombres de clases
    labels = ["airplane","automobile","bird","cat","deer",
              "dog","frog","horse","ship","truck"]

    # Seleccionar primeras 1000 imágenes
    x_subset = x_train[:1000]
    y_subset = y_train[:1000]

    for i, (img, label) in enumerate(zip(x_subset, y_subset)):
        cls = labels[label]
        # Guardar como JPG
        im = Image.fromarray(img)
        fname = f"{cls}_{i:04d}.jpg"
        im.save(os.path.join(out_dir, fname))

    print(f"Saved {len(x_subset)} images into '{out_dir}/'")

if __name__ == "__main__":
    main()