import numpy as np
from scipy.stats import norm

def generuj_punkty_pozioma(liczba_punktów: int = 2000, szerokość: float = 200, długość: float = 200, wysokość: float = 1,
                                    nazwa_pliku: str = "punkty.txt"):

    distribution_x = norm(loc=0, scale=szerokość)
    distribution_y = norm(loc=0, scale=długość)

    x = distribution_x.rvs(size=liczba_punktów)
    y = distribution_y.rvs(size=liczba_punktów)
    z = np.random.uniform(0, wysokość, liczba_punktów)

    punkty = np.column_stack((x, y, z))

    np.savetxt(nazwa_pliku, punkty, fmt='%.3f', delimiter=',', header='x,y,z')

    return punkty

def generuj_punkty_pionowa(liczba_punktów: int = 2000, szerokość: float = 200, wysokość: float = 200, długość: float = 1,
                                  nazwa_pliku: str = "punkty.txt"):

    distribution_x = norm(loc=0, scale=szerokość)
    distribution_z = norm(loc=0, scale=wysokość)

    x = distribution_x.rvs(size=liczba_punktów)
    z = distribution_z.rvs(size=liczba_punktów)
    y = np.zeros(liczba_punktów)

    punkty = np.column_stack((x, y, z))

    np.savetxt(nazwa_pliku, punkty, fmt='%.3f', delimiter=',', header='x,y,z')

    return punkty

def generuj_punkty_cylindryczna(liczba_punktów: int = 2000, promień: float = 100, wysokość: float = 200,
                                        nazwa_pliku: str = "punkty.txt"):

    theta = np.random.uniform(0, 2 * np.pi, liczba_punktów)
    z = np.random.uniform(0, wysokość, liczba_punktów)
    x = promień * np.cos(theta)
    y = promień * np.sin(theta)

    punkty = np.column_stack((x, y, z))

    np.savetxt(nazwa_pliku, punkty, fmt='%.3f', delimiter=',', header='x,y,z')

    return punkty

#Generowanie punktów na płaskiej poziomej powierzchni
punkty_płaska_pozioma = generuj_punkty_pozioma(2000, szerokość=200, długość=200, wysokość=1, nazwa_pliku="punkty_pozioma.xyz")

#Generowanie punktów na płaskiej pionowej powierzchni
punkty_płaska_pionowa = generuj_punkty_pionowa(2000, szerokość=200, wysokość=200, długość=1, nazwa_pliku="punkty_pionowa.xyz")

#Generowanie punktów na powierzchni cylindrycznej
punkty_cylindryczna = generuj_punkty_cylindryczna(2000, promień=100, wysokość=200, nazwa_pliku="punkty_cylindryczna.xyz")
