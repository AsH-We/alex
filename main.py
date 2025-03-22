import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
# ergertghregerhe
ewfg = True
alpha = 0.02
Nx = 150
Nt = 2000
Lx = 1.0
T_final = 0.1

dx = Lx / (Nx - 1)
dt = T_final / Nt

x = np.linspace(0, Lx, Nx)
u0 = np.sin(np.pi * x)

u = np.copy(u0)
for _ in range(Nt):
    u_new = np.copy(u)
    u_new[1:-1] = u[1:-1] + alpha * dt / dx**2 * (u[:-2] - 2*u[1:-1] + u[2:])
    u = u_new

u_t1 = u

X_train = u_t1.reshape(-1, Nx, 1)
y_train = u0.reshape(-1, Nx, 1)

def build_cnn():
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=(Nx, 1)),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.Conv1D(1, kernel_size=3, activation='linear', padding='same')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

cnn_model = build_cnn()

cnn_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

predicted_u0 = cnn_model.predict(X_train).flatten()

plt.plot(x, u0, label="True Initial Condition", linestyle='dashed')
plt.plot(x, predicted_u0, label="Predicted Initial Condition", linestyle='solid')
plt.xlabel("x")
plt.ylabel("Temperature")
plt.legend()
plt.show()

noise_level = 0.05
X_noisy = X_train + noise_level * np.random.normal(size=X_train.shape)
predicted_u0_noisy = cnn_model.predict(X_noisy).flatten()

plt.plot(x, u0, label="True Initial Condition", linestyle='dashed')
plt.plot(x, predicted_u0_noisy, label="Predicted with Noise", linestyle='solid')
plt.xlabel("x")
plt.ylabel("Temperature")
plt.legend()
plt.show()

