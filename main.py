import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


def generateDataset():
    operation = np.random.choice(["+", "-", "x", "/"])

    X = []
    y = []

    for _ in range(10000):
        z1 = np.random.rand() + np.random.rand() * 1j
        z2 = np.random.rand() + np.random.rand() * 1j

        operation = np.random.choice(["+", "-", "x", "/"])

        if operation == "+":
            result = z1 + z2
            opcode = 0
        elif operation == "-":
            result = z1 - z2
            opcode = 1
        elif operation == "x":
            result = z1 * z2
            opcode = 2
        elif operation == "/":
            result = z1 / z2
            opcode = 3

        X.append([z1.real, z1.imag, z2.real, z2.imag, opcode])
        y.append([result.real, result.imag])

    X = np.array(X)
    y = np.array(y)

    return X, y


X, y = generateDataset()
np.savetxt("dataset.csv", np.concatenate((X, y), axis=1), delimiter=",")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Dense(64, input_dim=5, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(2, activation="linear"))

model.compile(loss="mse", optimizer="adam")

model.fit(X_train, y_train, epochs=100, batch_size=32)

score = model.evaluate(X_test, y_test, batch_size=32)
print("Test loss:", score)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Mean squared error:", mse)
