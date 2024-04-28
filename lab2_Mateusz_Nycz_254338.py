import numpy as np
import pyransac3d as pyrsc

def wczytaj_chmure_punktow(plik):
    punkty = []
    with open(plik, 'r') as plik:
        for linia in plik:
            if linia.startswith('#'):
                continue
            elementy = linia.strip().split(',')
            x, y, z = map(float, elementy[:3])
            punkty.append([x, y, z])
    return np.array(punkty)

def dopasuj_płaszczyznę_pyransac(punkty, prog=0.01, min_punktow=3):
    płaszczyzna = pyrsc.Plane()
    najlepsza_płaszczyzna, najlepsze_punkty_wykresu = płaszczyzna.fit(punkty, thresh=prog, minPoints=min_punktow)

    if najlepsza_płaszczyzna is not None:
        wektor_normalny = najlepsza_płaszczyzna[:-1]
        return wektor_normalny, najlepsze_punkty_wykresu
    else:
        return None, None

def wyświetl_orientację_pyransac(nazwa_płaszczyzny, punkty):
    wektor_normalny, punkty_klastra = dopasuj_płaszczyznę_pyransac(punkty)
    if wektor_normalny is not None:
        if abs(wektor_normalny[2]) > abs(wektor_normalny[0]) and abs(wektor_normalny[2]) > abs(wektor_normalny[1]):
            orientacja = "Pionowa"
        else:
            orientacja = "Pozioma"
        print(f"{nazwa_płaszczyzny}:")
        print("Wektor normalny:", wektor_normalny)
        print("Orientacja płaszczyzny:", orientacja)
        print()
    else:
        print(f"{nazwa_płaszczyzny}:")
        print("Nie udało się znaleźć płaszczyzny")
        print()

def oddziel_klastry_DBSCAN(punkty, eps=0.1, min_próbek=50):
    płaszczyzna = pyrsc.Plane()
    etykiety = płaszczyzna.fit_predict(punkty, eps=eps, min_samples=min_próbek)

    # Znajdź unikalne etykiety klastrów
    unikalne_etykiety = np.unique(etykiety)

    klastry = []
    for etykieta in unikalne_etykiety:
        if etykieta == -1:
            continue  # Pomijamy punkty uznane za szum
        punkty_klastra = punkty[etykiety == etykieta]
        klastry.append(punkty_klastra)

    return klastry

punkty_pionowe = wczytaj_chmure_punktow("punkty_pionowa.xyz")
punkty_poziome = wczytaj_chmure_punktow("punkty_pozioma.xyz")
punkty_cylindryczne = wczytaj_chmure_punktow("punkty_cylindryczna.xyz")

wyświetl_orientację_pyransac("Płaszczyzna pionowa", punkty_pionowe)
wyświetl_orientację_pyransac("Płaszczyzna pozioma", punkty_poziome)
wyświetl_orientację_pyransac("Płaszczyzna cylindryczna", punkty_cylindryczne)