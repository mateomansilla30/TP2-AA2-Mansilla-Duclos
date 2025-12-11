import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os

# Carga de la Q-table final usada como dataset
with open('flappy_birds_q_table_final.pkl', 'rb') as f:
    q_table = pickle.load(f)

# Conversión a arrays para entrenamiento
X = np.array([s for s in q_table.keys()], dtype=np.float32)
y = np.array([q for q in q_table.values()], dtype=np.float32)

print(len(q_table))

# Cálculo y guardado de parámetros de normalización
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

with open('normalization_params.pkl', 'wb') as f:
    pickle.dump({'mean': X_mean, 'std': X_std}, f)

# División en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X_norm, y, test_size=0.2, random_state=42
)

# Definición del modelo denso para aproximar Q-values
model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='linear')
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='mse',
    metrics=['mae']
)

# Callbacks para control de entrenamiento
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
]

# Entrenamiento del modelo
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Guardado final del modelo
model.save('flappy_q_nn_model.keras')
