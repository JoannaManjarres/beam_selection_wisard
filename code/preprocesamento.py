import numpy as np
import csv

def read_all_coordinates():
    filename = "../data/coordinates/CoordVehiclesRxPerScene_s008.csv"
    limit_ep_train = 1564

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        number_of_rows = len(list(reader))

    all_info_coord = np.zeros([number_of_rows, 6], dtype=object)

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        cont = 0
        for row in reader:
            all_info_coord[cont] = int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
            cont += 1

    # all_info_coord = np.array(all_info_coord)

    # Separacao do conjunto de dados em treinamento e teste
    coord_train = all_info_coord[(all_info_coord[:, 0] < limit_ep_train + 1)]
    coord_test = all_info_coord[(all_info_coord[:, 0] > limit_ep_train)]

    return all_info_coord

