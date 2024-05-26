import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Wczytanie danych z pliku CSV
file_path = r'C:\Users\Matiq\PycharmProjects\Zad4_POI\Zebrane_dane_z_wycinków_zad4.csv'
df = pd.read_csv(file_path, sep=';')

# Podział danych na cechy (x) i etykiety (y)
x = df.iloc[:, 1:].astype('float').values
y = df.iloc[:, 0].values

# Wybierz odpowiednie zakresy indeksów dla każdej kategorii
gres_indices = list(range(2, 9))
laminat_indices = list(range(10, 18))
tynk_indices = list(range(19, 27))

# Połącz indeksy z każdej kategorii w jeden zestaw indeksów treningowych
selected_indices = gres_indices + laminat_indices + tynk_indices

# Wybierz odpowiednie przykłady i etykiety
x_train = x[selected_indices]
y_train = y[selected_indices]

# Kodowanie etykiet
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
onehot_encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train_encoded, test_size=0.3, random_state=42)

print("Kształt danych treningowych X_train:", X_train.shape)
print("Kształt danych testowych X_test:", X_test.shape)
print("Kształt etykiet treningowych y_train:", y_train.shape)
print("Kształt etykiet testowych y_test:", y_test.shape)

# Definiowanie modelu
model = Sequential()
model.add(Dense(units=10, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(units=y_train.shape[1], activation='softmax'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

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
