import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Generate a random complex number
z = np.random.rand() + np.random.rand() * 1j

# Generate a random arithmetic operation
operation = np.random.choice(["+", "-", "x", "/"])

X = []
y = []

for i in range(1000):
    # Generate two random complex numbers
    z1 = np.random.rand() + np.random.rand() * 1j
    z2 = np.random.rand() + np.random.rand() * 1j

    # Generate a random arithmetic operation
    operation = np.random.choice(["+", "-", "x", "/"])

    # Compute the result of the arithmetic operation
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

    # Append the input and output to the dataset
    X.append([z1.real, z1.imag, z2.real, z2.imag, opcode])
    y.append([result.real, result.imag])

X = np.array(X)
y = np.array(y)

# Create a neural network
model = Sequential()
model.add(Dense(64, input_dim=5, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(2, activation="linear"))

# Compile the neural network
model.compile(loss="mse", optimizer="adam")

# Train the neural network
model.fit(X, y, epochs=100, batch_size=32)

# Generate a new set of random complex numbers
z1 = np.random.rand() + np.random.rand() * 1j
z2 = np.random.rand() + np.random.rand() * 1j

# Predict the result using the neural network
result = model.predict(np.array([[z1.real, z1.imag, z2.real, z2.imag, 0]]))

# Print the result
print(f"{z1} + {z2} = {result[0][0]} + {result[0][1]}i")
