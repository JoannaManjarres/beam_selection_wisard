import numpy as np


def read_all_Qs_matrix():

    path = "/Users/Joanna/git/beam_selection_wisard/data/coordinates/processed/"

    input_cache_file = np.load(path + "all_train_coord.npz", allow_pickle=True)
    all_coord_train = input_cache_file["coord_training"]

    input_cache_file = np.load(path + "all_test_coord.npz", allow_pickle=True)
    all_coord_test = input_cache_file["coor_test"]

    return all_coord_train, all_coord_test

def read_Qs_matrix_with_LOS_data():
    path = "/Users/Joanna/git/beam_selection_wisard/data/coordinates/processed/"

    input_cache_file = np.load(path + "LOS_train_coord.npz", allow_pickle=True)
    LOS_coord_train = input_cache_file["coord_LOS_training"]
    input_LOS_coord_train = LOS_coord_train.tolist()

    input_cache_file = np.load(path + "LOS_test_coord.npz", allow_pickle=True)
    LOS_coord_test = input_cache_file["coor_LOS_test"]
    input_LOS_coord_test = LOS_coord_test.tolist()

    return input_LOS_coord_train, input_LOS_coord_test

def read_Qs_matrix_with_NLOS_data():
    path = "/Users/Joanna/git/beam_selection_wisard/data/coordinates/processed/"

    input_cache_file = np.load(path + "NLOS_train_coord.npz", allow_pickle=True)
    NLOS_coord_train = input_cache_file["coord_NLOS_training"]

    input_cache_file = np.load(path + "NLOS_test_coord.npz", allow_pickle=True)
    NLOS_coord_test = input_cache_file["coor_NLOS_test"]

    return NLOS_coord_train, NLOS_coord_test

def read_all_beams():
    path = "/Users/Joanna/git/beam_selection_wisard/data/beams/all_index_beam/"

    input_cache_file = np.load(path + "index_beams_rx_test.npz", allow_pickle=True)
    index_beam_rx_test = input_cache_file["all_beams_rx_test"].astype(str)

    input_cache_file = np.load(path + "index_beams_rx_train.npz", allow_pickle=True)
    index_beam_rx_train = input_cache_file["all_beams_rx_train"].astype(str)

    input_cache_file = np.load(path + "index_beams_tx_train.npz", allow_pickle=True)
    index_beam_tx_train = input_cache_file["all_beams_tx_train"].astype(str)

    input_cache_file = np.load(path + "index_beams_tx_test.npz", allow_pickle=True)
    index_beam_tx_test = input_cache_file["all_beams_tx_test"].astype(str)

    return index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test

def read_LOS_beams():
    path = "/Users/Joanna/git/beam_selection_wisard/data/beams/LOS_index_beam/"

    input_cache_file = np.load(path + "beam_LOS_rx_train.npz", allow_pickle=True)
    index_beam_rx_LOS_train = input_cache_file["beam_LOS_rx_train"].astype(str)
    label_rx_LOS_train =index_beam_rx_LOS_train.tolist()

    input_cache_file = np.load(path + "beam_LOS_rx_test.npz", allow_pickle=True)
    index_beam_rx_LOS_test = input_cache_file["beam_LOS_rx_test"].astype(str)
    label_rx_LOS_test = index_beam_rx_LOS_test.tolist()

    input_cache_file = np.load(path + "beam_LOS_tx_train.npz", allow_pickle=True)
    index_beam_tx_LOS_train = input_cache_file["beam_LOS_tx_train"].astype(str)
    label_tx_LOS_train = index_beam_tx_LOS_train.tolist()

    input_cache_file = np.load(path + "beam_LOS_tx_test.npz", allow_pickle=True)
    index_beam_tx_LOS_test = input_cache_file["beam_LOS_tx_test"].astype(str)
    label_tx_LOS_test = index_beam_tx_LOS_test.tolist()

    return label_rx_LOS_train, label_rx_LOS_test, label_tx_LOS_train, label_tx_LOS_test

def read_NLOS_beams():
    path = "/Users/Joanna/git/beam_selection_wisard/data/beams/NLOS_index_beam/"

    input_cache_file = np.load(path + "beam_NLOS_rx_train.npz", allow_pickle=True)
    index_beam_rx_NLOS_train = input_cache_file["beam_NLOS_rx_train"].astype(str)

    input_cache_file = np.load(path + "beam_NLOS_rx_test.npz", allow_pickle=True)
    index_beam_rx_NLOS_test = input_cache_file["beam_NLOS_rx_test"].astype(str)

    input_cache_file = np.load(path + "beam_NLOS_tx_train.npz", allow_pickle=True)
    index_beam_tx_NLOS_train = input_cache_file["beam_NLOS_tx_train"].astype(str)

    input_cache_file = np.load(path + "beam_NLOS_tx_test.npz", allow_pickle=True)
    index_beam_tx_NLOS_test = input_cache_file["beam_NLOS_tx_test"].astype(str)

    return index_beam_rx_NLOS_train, index_beam_rx_NLOS_test, index_beam_tx_NLOS_train, index_beam_tx_NLOS_test
