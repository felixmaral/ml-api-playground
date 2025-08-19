from tensorflow import keras
import matplotlib.pyplot as plt

(_, _), (xte, yte) = keras.datasets.cifar10.load_data()
img = xte[0]  # primera imagen del test
plt.imsave("sample.jpg", img)
print("Imagen guardada como sample.jpg")