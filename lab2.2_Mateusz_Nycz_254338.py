import numpy as np
from numpy.linalg import svd
from sklearn.cluster import KMeans
import warnings

def dopasuj_plaszczyzne_RANSAC(punkty, iteracje=1000, prog=0.01):
    najlepsza_plaszczyzna = None
    najlepsze_punkty = None
    max_punkty = 0

    for _ in range(iteracje):
        indeksy_wyboru = np.random.choice(len(punkty), 3, replace=False)
        wybrane_punkty = punkty[indeksy_wyboru]

        A = np.column_stack((wybrane_punkty, np.ones(3)))
        _, _, V = svd(A)
        parametry_plaszczyzny = V[-1, :]

        odleglosci = np.abs(np.dot(punkty, parametry_plaszczyzny[:-1]) + parametry_plaszczyzny[-1]) / np.linalg.norm(parametry_plaszczyzny[:-1])

        punkty_na_plaszczyznie = punkty[odleglosci < prog]
        liczba_punktow = len(punkty_na_plaszczyznie)

        if liczba_punktow > max_punkty:
            max_punkty = liczba_punktow
            najlepsza_plaszczyzna = parametry_plaszczyzny
            najlepsze_punkty = punkty_na_plaszczyznie

    return najlepsza_plaszczyzna, najlepsze_punkty

def wczytaj_chmure_punktow(plik):
    punkty = []
    with open(str(plik), 'r') as f:
        for linia in f:
            if linia.startswith('#'):
                continue
            elementy = linia.strip().split(',')
            x, y, z = map(float, elementy[:3])
            punkty.append([x, y, z])
    return np.array(punkty)

def print_orientacje(nazwa_plaszczyzny, punkty, centroidy, etykiety):
    for i, centroid in enumerate(centroidy):
        punkty_klastra = punkty[etykiety == i]

        parametry_plaszczyzny, _ = dopasuj_plaszczyzne_RANSAC(punkty_klastra)
        wektor_normalny = parametry_plaszczyzny[:-1]

        if abs(wektor_normalny[2]) > abs(wektor_normalny[0]) and abs(wektor_normalny[2]) > abs(wektor_normalny[1]):
            orientacja = "Pozioma"
        else:
            orientacja = "Pionowa"

        print(f"{nazwa_plaszczyzny} {i + 1}:")
        print(f"Klaster {i + 1} punktów:")
        print("Wektor normalny do plaszczyzny:", wektor_normalny)
        print("Orientacja:", orientacja)
        print()

punkty_pionowa = wczytaj_chmure_punktow("punkty_pionowa.xyz")
punkty_pozioma = wczytaj_chmure_punktow("punkty_pozioma.xyz")
punkty_cylindryczna = wczytaj_chmure_punktow("punkty_cylindryczna.xyz")

k = 3
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", UserWarning)
    kmeans_v = KMeans(n_clusters=k).fit(punkty_pionowa)
    kmeans_h = KMeans(n_clusters=k).fit(punkty_pozioma)
    kmeans_c = KMeans(n_clusters=k).fit(punkty_cylindryczna)

print_orientacje("Plaszczyzna pionowa", punkty_pionowa, kmeans_v.cluster_centers_, kmeans_v.labels_)
print_orientacje("Plaszczyzna pozioma", punkty_pozioma, kmeans_h.cluster_centers_, kmeans_h.labels_)
print_orientacje("Plaszczyzna cylindryczna", punkty_cylindryczna, kmeans_c.cluster_centers_, kmeans_c.labels_)
