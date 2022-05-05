import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.layers.core import Dense
from keras.models import Sequential

from dataset import get_dataset

X_train, X_test, y_train, y_test = get_dataset(True)
PER_GROUP = 30
y = list(y_train) + list(y_test)
min_y = min(y)
max_y = max(y)

mapped_train = []
mapped_test = []
for i in y_test:
    mapped_test.append((i - min_y) // PER_GROUP)
for i in y_train:
    mapped_train.append((i - min_y) // PER_GROUP)

model = Sequential()
model.add(Dense(60, input_dim=len(X_train[0]), activation="relu"))
model.add(Dense(60, activation="relu"))

model.add(Dense(1 + (max_y - min_y) // PER_GROUP, activation="softmax"))
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
try:
    model.load_weights("nn30.net")
except:
    model.fit(
        X_train,
        tf.keras.utils.to_categorical(np.array(mapped_train)),
        batch_size=3,
        epochs=800,
        verbose=1,
    )
    model.save_weights("nn30.net")

values = model.predict(X_test)
pred = np.argmax(values, 1)
errors = []
for i in range(len(pred)):
    errors.append(abs(pred[i] - mapped_test[i]))
    if abs(pred[i] - mapped_test[i]) > 0:
        print("bingo")

import matplotlib.pyplot as plt

print(len(errors))
plt.hist(x=errors, color="#0504aa", bins=list(range(0, 100)), rwidth=0.95)
plt.grid(axis="y", alpha=0.75)
plt.xlabel("Broj klasifikovanoh postova")
plt.ylabel("kategorija")
plt.title("")
plt.show()
