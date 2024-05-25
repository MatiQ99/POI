import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Wczytanie danych z pliku CSV
df = pd.read_csv(r'C:\Users\Matiq\PycharmProjects\lab3.2_Mateusz_Nycz_254338\Zebrane_dane_z_wycinków.csv', sep=',')
data = df.to_numpy()

x = data[:, :-1].astype('float')
y = data[:, -1]

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)

# Kodowanie one-hot
onehot_encoder = OneHotEncoder(sparse_output=False)  # Zmieniono sparse na sparse_output
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(x, onehot_encoded, test_size=0.3)

# Definiowanie modelu
model = Sequential()
model.add(Dense(units=10, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(units=y_train.shape[1], activation='softmax'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Podsumowanie modelu
model.summary()

# Trenowanie modelu
model.fit(x=X_train, y=y_train, epochs=100, batch_size=10, shuffle=True)

# Przewidywanie na danych testowych
y_pred = model.predict(X_test)
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)

# Macierz pomyłek
cm = confusion_matrix(y_test_int, y_pred_int)
print(cm)
