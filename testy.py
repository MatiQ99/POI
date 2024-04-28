import numpy as np
from numpy.linalg import svd
from sklearn.cluster import k_means

def dopasuj_plaszczyzne_RANSAC(punkty, iteracje=1000, prog=0.01):
    najlepsza_plaszczyzna = None
    najlepsze_punkty_wykresu = None
    max_wykresy = 0

    for _ in range(iteracje):
        # Losowo wybierz trzy punkty
        indeksy_wyboru = np.random.choice(len(punkty), 3, replace=False)
        punkty_wyboru = punkty[indeksy_wyboru]

        # Dopasuj plaszczyzne do wybranych punktów
        A = np.column_stack((punkty_wyboru, np.ones(3)))
        _, _, V = svd(A)
        parametry_plaszczyzny = V[-1, :]

        # Oblicz odleglosc od plaszczyzny dla wszystkich punktów
        odleglosci = np.abs(np.dot(punkty, parametry_plaszczyzny[:-1]) + parametry_plaszczyzny[-1]) / np.linalg.norm(parametry_plaszczyzny[:-1])

        # Licz wewnętrzne punkty
        wykresy = punkty[odleglosci < prog]
        liczba_wykresow = len(wykresy)

        # Aktualizuj najlepsza plaszczyzne, jeśli znaleziono lepszy model
        if liczba_wykresow > max_wykresy:
            max_wykresy = liczba_wykresow
            najlepsza_plaszczyzna = parametry_plaszczyzny
            najlepsze_punkty_wykresu = wykresy

    return najlepsza_plaszczyzna, najlepsze_punkty_wykresu

def wczytaj_chmure_punktow(plik):
    punkty = []
    with open(plik, 'r') as plik:
        for linia in plik:
            # Pomijaj komentarze
            if linia.startswith('#'):
                continue
            # Podziel linie na elementy, zakładając, że są oddzielone przecinkiem
            elementy = linia.strip().split(',')
            # Wczytaj wspolrzedne x, y, z
            x, y, z = map(float, elementy[:3])
            punkty.append([x, y, z])
    return np.array(punkty)

# Wczytaj chmurę punktów z pliku xyz
try:
    punkty_pionowe = wczytaj_chmure_punktow("punkty_pionowe.xyz")
    punkty_poziome = wczytaj_chmure_punktow("punkty_poziome.xyz")
    punkty_cylindryczne = wczytaj_chmure_punktow("punkty_cylindryczne.xyz")

    # Znajdź rozłączne chmury punktów za pomocą algorytmu k-średnich
    k = 3
    centroids_v, labels_v = k_means(punkty_pionowe, k)
    centroids_h, labels_h = k_means(punkty_poziome, k)
    centroids_c, labels_c = k_means(punkty_cylindryczne, k)

    # Dopasuj płaszczyzny do każdej chmury punktów
    def print_orientation(plane_name, points, labels):
        for i in range(k):
            normal_vector = wczytaj_chmure_punktow(points[labels == i])[0][:-1]
            if abs(normal_vector[2]) > abs(normal_vector[0]) and abs(normal_vector[2]) > abs(normal_vector[1]):
                orientation = "Pozioma"
            else:
                orientation = "Pionowa"
            print(f"{plane_name} {i + 1}:")
            print(f"Chmura punktów {i + 1}:")
            print("Wektor normalny:", normal_vector)
            print("Orientacja płaszczyzny:", orientation)
            print()

    print_orientation("Plaszczyzna pionowa", punkty_pionowe, labels_v)
    print_orientation("Plaszczyzna pozioma", punkty_poziome, labels_h)
    print_orientation("Plaszczyzna cylindryczna", punkty_cylindryczne, labels_c)

except FileNotFoundError:
    print("Plik nie zostal znaleziony. Upewnij sie, ze podales prawidlowa nazwe pliku i sprawdz sciezke dostepu.")