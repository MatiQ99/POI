import os
import cv2
import numpy as np
import pandas as pd
from skimage import io, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def crop_and_save_images(katalogi_zrodlowe, katalog_docelowy, rozmiar_przyciecia=(128, 128)):
    if not os.path.exists(katalog_docelowy):
        os.makedirs(katalog_docelowy)

    print("Przycinanie obrazów")
    liczba_przycietych_obrazow = 0

    # Przetwarzaj każdy katalog źródłowy
    for katalog_zrodlowy in katalogi_zrodlowe:
        # Znajdź wszystkie pliki .jpg w katalogu źródłowym
        pliki = [plik for plik in os.listdir(katalog_zrodlowy) if plik.endswith('.jpg')]

        # Przetwarzaj każdy obraz
        for nazwa_pliku in pliki:
            sciezka_obrazu = os.path.join(katalog_zrodlowy, nazwa_pliku)
            obraz = cv2.imread(sciezka_obrazu)
            if obraz is None:
                print(f"Nie udana próba załadowania obrazu {nazwa_pliku} z {katalog_zrodlowy}")
                continue
            wysokosc, szerokosc, _ = obraz.shape  # Pobierz wymiary obrazu

            # Przycinanie obrazu
            indeks_przyciecia = 0
            for y in range(0, wysokosc - rozmiar_przyciecia[1] + 1, rozmiar_przyciecia[1]):
                for x in range(0, szerokosc - rozmiar_przyciecia[0] + 1, rozmiar_przyciecia[0]):
                    przyciety_obraz = obraz[y:y + rozmiar_przyciecia[1], x:x + rozmiar_przyciecia[0]]
                    sciezka_przycietego_obrazu = os.path.join(katalog_docelowy, f"{os.path.splitext(nazwa_pliku)[0]}_{indeks_przyciecia}.jpg")
                    cv2.imwrite(sciezka_przycietego_obrazu, przyciety_obraz)
                    indeks_przyciecia += 1
                    liczba_przycietych_obrazow += 1
            print(f"Wykonano {indeks_przyciecia} przycięć zdjęć z {sciezka_obrazu}")
    print(f"Liczba przyciętych zjęć wynosi: {liczba_przycietych_obrazow}")


def extract_texture_features(katalog_z_obrazami, odleglosci=[1, 3, 5], katy=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    features_list = []
    print("Ekstrakcja cech tekstury")

    pliki = os.listdir(katalog_z_obrazami)
    if not pliki:
        print("Brak zdjęć w katalogu.")
        return pd.DataFrame()

    # Przetwarzaj każdy obraz w katalogu
    for indeks, nazwa_pliku in enumerate(pliki):
        if nazwa_pliku.endswith('.jpg'):
            print(f"Zdjęcie: {nazwa_pliku} ({indeks + 1}/{len(pliki)})...")
            sciezka_obrazu = os.path.join(katalog_z_obrazami, nazwa_pliku)
            obraz = io.imread(sciezka_obrazu)
            if obraz is None or obraz.size == 0:
                print(f"Nie udana próba załadowania zdjęcia {nazwa_pliku}")
                continue

            # Konwersja obrazu do skali szarości
            if len(obraz.shape) == 3:
                obraz_szarosci = color.rgb2gray(obraz)
            else:
                obraz_szarosci = obraz  # Zakładamy, że jest już w skali szarości

            # Przeskalowanie wartości pikseli
            obraz_szarosci = img_as_ubyte(obraz_szarosci)
            obraz_szarosci //= 4  # Redukcja do 5 bitów na piksel (64 poziomy)

            # Obliczanie GLCM i cech tekstur
            glcm = graycomatrix(obraz_szarosci, odleglosci, katy, 64, symmetric=True, normed=True)
            cechy = {'plik': nazwa_pliku,
                     'kategoria': nazwa_pliku.split('_')[0]}  # Użyj części nazwy pliku jako kategorii

            for cecha in ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity']:
                for odleglosc in odleglosci:
                    for kat in katy:
                        cechy[f'{cecha}_d{odleglosc}_a{int(np.degrees(kat))}'] = graycoprops(glcm, cecha)[
                            odleglosci.index(odleglosc), katy.index(kat)]

            features_list.append(cechy)

    cechy_df = pd.DataFrame(features_list)
    print(f"Udane wyekstrahowane cechy dla {len(cechy_df)} zdjęć.")
    return cechy_df


def classify_features(cechy_df):
    if cechy_df.empty or 'kategoria' not in cechy_df:
        print("Brak danych lub kategorii.")
        return

    X = cechy_df.drop(['plik', 'kategoria'], axis=1)
    y = cechy_df['kategoria']

    if len(set(y)) < 2:
        print("Liczba klas jest niewystarczająca do klasyfikacji.")
        return

    X_treningowe, X_testowe, y_treningowe, y_testowe = train_test_split(X, y, test_size=0.3, random_state=42)

    klasyfikator_knn = KNeighborsClassifier(n_neighbors=3)
    klasyfikator_knn.fit(X_treningowe, y_treningowe)
    y_pred = klasyfikator_knn.predict(X_testowe)

    dokladnosc = accuracy_score(y_testowe, y_pred)
    print(f"Dokładność: {dokladnosc:.4f}")


def save_features_to_csv(cechy_df, plik_csv):
    if cechy_df.empty:
        print("Brak danych do zapisania.")
        return
    cechy_df.to_csv(plik_csv, index=False)
    print(f"Zapisane dane: {plik_csv}")

# Ścieżki do folderów źródłowych
katalogi_zrodlowe = [
    r"C:\Users\Matiq\Desktop\Semestr 1\Programowanie w obliceniach inteligentnych\Zadania\Zad 3\Gres",
    r"C:\Users\Matiq\Desktop\Semestr 1\Programowanie w obliceniach inteligentnych\Zadania\Zad 3\Laminat",
    r"C:\Users\Matiq\Desktop\Semestr 1\Programowanie w obliceniach inteligentnych\Zadania\Zad 3\Tynk"
]

# Ścieżka do folderu docelowego
katalog_docelowy = r"C:\Users\Matiq\Desktop\Semestr 1\Programowanie w obliceniach inteligentnych\Zadania\Zad 3\Tekstury"

# Wykonanie przycinania obrazów ze wszystkich folderów źródłowych
crop_and_save_images(katalogi_zrodlowe, katalog_docelowy)

# Ekstrakcja cech tekstur z przyciętych obrazów
cechy_df = extract_texture_features(katalog_docelowy)

if not cechy_df.empty:
    # Zapisanie cech do pliku CSV
    save_features_to_csv(cechy_df, 'Zebrane_dane_z_wycinków.csv')

    # Klasyfikacja cech i wydrukowanie dokładności
    classify_features(cechy_df)
else:
    print("Brak danych do klasyfikacji.")
