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

def read_valid_coordinates():
    filename = '/Users/Joanna/git/Analise_de_dados/data/coordinates/CoordVehiclesRxPerScene_s008.csv'
    limit_ep_train = 1564

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        number_of_rows = len(list(reader))

    all_info_coord_val = np.zeros([11194, 5], dtype=object)

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        cont = 0
        for row in reader:
            if row['Val'] == 'V':
                all_info_coord_val[cont] = int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS']
                cont += 1

    # all_info_coord = np.array(all_info_coord)

    coord_train = all_info_coord_val[(all_info_coord_val[:, 0] < limit_ep_train + 1)]
    coord_test = all_info_coord_val[(all_info_coord_val[:, 0] > limit_ep_train)]

    return all_info_coord_val

def coord_to_Qs_matrix(coordenadas):
    ''''Este metodo recibe o obj coordenadas com a ordem dos seguintes dados
        (EpisodeID, coord_x, coord_y, coord_z, LOS, val)'''

    min_coord_x = np.min(coordenadas[:, 1])
    min_coord_y = np.min(coordenadas[:, 2])

    max_coord_x = np.max(coordenadas[:, 1])
    max_coord_y = np.max(coordenadas[:, 2])


    area_interese_mtrs = np.array([23, 250], dtype=int)
    num_positions_of_internal_matrix = area_interese_mtrs[0] * area_interese_mtrs[1]

    coordenadas_x_mtrs = np.array((coordenadas[:, 1] - min_coord_x) / max_coord_x * float(area_interese_mtrs[0]), dtype=float)
    coordenadas_y_mtrs = np.array((coordenadas[:, 2] - min_coord_y) / max_coord_y * float(area_interese_mtrs[1]), dtype=float)
    coordenadas_x_mtrs = np.round(coordenadas_x_mtrs)
    coordenadas_y_mtrs = np.round(coordenadas_y_mtrs)

    Qs_all = np.zeros(area_interese_mtrs,dtype=int)  # Matriz que representa a posicao de TODOS os veiculos num episodio

    vector_qs_1 = []

    amostra = 0
    coord_val_Qs=[]
    Qs_val= np.zeros(area_interese_mtrs, dtype=int)

    for i in range(2086):
        for a in range(10):
            x = int(coordenadas_x_mtrs[amostra])
            y = int(coordenadas_y_mtrs[amostra])
            Qs_all[x, y] = 1
            amostra += 1

        episodio = coordenadas[amostra-1,0]
        flag_valid_coord = coordenadas[amostra-1, 5]
        flag_LOS_or_NLOS = coordenadas[amostra-1, 4]

        vector_qs_1.append([episodio, flag_valid_coord, flag_LOS_or_NLOS, Qs_all.reshape(1, num_positions_of_internal_matrix)])
        Qs_all = np.zeros(area_interese_mtrs, dtype=int)


    coordenadas_validas = coordenadas[coordenadas[:, 5] == 'V']
    coord_val_x_mtrs = np.array((coordenadas_validas[:, 1] - min_coord_x) / max_coord_x * float(area_interese_mtrs[0]),dtype=float)
    coord_val_y_mtrs = np.array((coordenadas_validas[:, 2] - min_coord_y) / max_coord_y * float(area_interese_mtrs[1]),dtype=float)
    coord_val_x_mtrs = np.round(coord_val_x_mtrs)
    coord_val_y_mtrs = np.round(coord_val_y_mtrs)

    for i in range(len(coordenadas_validas)):
        x_val = int(coord_val_x_mtrs[i])
        y_val = int(coord_val_y_mtrs[i])
        Qs_val[x_val, y_val] = 1

        episodio_val = coordenadas_validas[i,0]
        flag_LOS_or_NLOS_val = coordenadas_validas[i,4]

        coord_val_Qs.append([episodio_val, flag_LOS_or_NLOS_val, Qs_val.reshape(1, num_positions_of_internal_matrix)])
        Qs_val= np.zeros(area_interese_mtrs, dtype=int)



    Qs=[]
    for j in range(len(vector_qs_1)):
        for i in range(len(coord_val_Qs)):
            if(coord_val_Qs[i][0] == vector_qs_1[j][0]):
                z = np.concatenate([coord_val_Qs[i][2],vector_qs_1[j][3]],axis=1)
                Qs.append([coord_val_Qs[i][1], z])

    train = Qs[:9234]
    test =  Qs[9234:]

    separed_coord_LOS = True
    coordenadas_LOS_train = []
    coordenadas_LOS_train1 = []
    coordenadas_NLOS_train = []
    coordenadas_NLOS_train1 = []
    coordenadas_LOS_test = []
    coordenadas_LOS_test1 = []
    coordenadas_NLOS_test = []
    coordenadas_NLOS_test1 = []
    temp_train = []
    temp_test = []
    coordenadas_all_train = []
    coordenadas_all_test = []

    if separed_coord_LOS:
        for i in range(len(train)):
            temp_train.append(train[i][1].tolist())
            if train[i][0] == 'LOS=1':
                coordenadas_LOS_train1.append(train[i][1].tolist())
            else:
                coordenadas_NLOS_train1.append(train[i][1].tolist())
        for i in range(len(test)):
            temp_test.append(test[i][1].tolist())
            if test[i][0] == 'LOS=1':
                coordenadas_LOS_test1.append(test[i][1].tolist())
            else:
                coordenadas_NLOS_test1.append(test[i][1].tolist())

    for i in range(len(coordenadas_LOS_train1)):
        coordenadas_LOS_train.append(coordenadas_LOS_train1[i][0])

    for i in range(len(coordenadas_NLOS_train1)):
        coordenadas_NLOS_train.append(coordenadas_NLOS_train1[i][0])

    for i in range(len(coordenadas_LOS_test1)):
        coordenadas_LOS_test.append(coordenadas_LOS_test1[i][0])

    for i in range(len(coordenadas_NLOS_test1)):
        coordenadas_NLOS_test.append(coordenadas_NLOS_test1[i][0])

    for i in range(len(train)):
        coordenadas_all_train.append(temp_train[i][0])

    for i in range(len(test)):
        coordenadas_all_test.append(temp_test[i][0])

    return coordenadas_all_train, coordenadas_all_test, coordenadas_LOS_train, coordenadas_NLOS_train, coordenadas_LOS_test, coordenadas_NLOS_test

def save_Qs_matrix(all_train_coord, all_test_coord, LOS_train_coord, LOS_test_coord, NLOS_train_coord, NLOS_test_coord):
    save_path = "../data/coordinates/processed/"

    # Todas as coordenadas em formato Qs
    np.savez(save_path + 'all_train_coord' + '.npz', coord_training=all_train_coord)
    np.savez(save_path + 'all_test_coord' + '.npz', coor_test=all_test_coord)


    # coordenadas LOS em formato Qs
    np.savez(save_path + 'LOS_train_coord' + '.npz', coord_LOS_training=LOS_train_coord)
    np.savez(save_path + 'LOS_test_coord' + '.npz', coor_LOS_test=LOS_test_coord)

    # coordenadas NLOS em formato Qs
    np.savez(save_path + 'NLOS_train_coord' + '.npz', coord_NLOS_training=NLOS_train_coord)
    np.savez(save_path + 'NLOS_test_coord' + '.npz', coor_NLOS_test=NLOS_test_coord)

def read_beams_raymobtime():
    data_path = '/Users/Joanna/git/beam_selection_wisard/data/beams/Ailton/beam_output/beams_output_8x32.npz'
    beams = np.load(data_path)['output_classification']
    num_antennas_tx = 32
    num_antennas_rx = 8

    best_beam_index = []
    for sample in range(beams.shape[0]):
        best_beam_index.append(np.argmax(beams[sample, :]))

    beam_index_rx = np.array(best_beam_index)

    tx_index = np.zeros((beams.shape[0]), dtype=int)
    rx_index = np.zeros((beams.shape[0]), dtype=int)

    for sample in range(len(beam_index_rx)):
        index_tx = best_beam_index[sample] // num_antennas_rx
        index_rx = best_beam_index[sample] % num_antennas_rx
        tx_index[sample] = index_tx
        rx_index[sample] = index_rx

    return tx_index, rx_index

def divide_beams_in_train_test(rx_beams, tx_beams, coord, save_data):


    all_data = np.column_stack((coord, rx_beams, tx_beams))

    limit_ep_train = 1564

    data_train = all_data[(all_data[:, 0] < limit_ep_train + 1)]
    data_test = all_data[(all_data[:, 0] > limit_ep_train)]

    all_beam_tx_train =data_train[:,-1].astype(int)
    all_beam_tx_test =data_test[:,-1].astype(int)

    all_beam_rx_train =data_train[:,-2].astype(int)
    all_beam_rx_test =data_test[:,-2].astype(int)

    if save_data:
        all_beams_tx_train = data_train[:, -1]
        all_beams_tx_test = data_test[:, -1]

        all_beams_rx_train = data_train[:, -2]
        all_beams_rx_test = data_test[:, -2]


        save_path = '/Users/Joanna/git/beam_selection_wisard/data/beams/all_index_beam/'
        np.savez(save_path + 'index_beams_tx_train' + '.npz', all_beams_tx_train=all_beams_tx_train)
        np.savez(save_path + 'index_beams_tx_test' + '.npz', all_beams_tx_test=all_beams_tx_test)

        np.savez(save_path + 'index_beams_rx_train' + '.npz', all_beams_rx_train=all_beams_rx_train)
        np.savez(save_path + 'index_beams_rx_test' + '.npz', all_beams_rx_test=all_beams_rx_test)


    return all_beam_tx_train, all_beam_tx_test, all_beam_rx_train, all_beam_rx_test

def divide_beams_and_coord_in_LOS_or_NLOS_connect(rx_beams, tx_beams, coord, save_data):
    '''Metodo que genera y guarda las coordenadas y beams para train y test'''

    all_data = np.column_stack((coord, rx_beams, tx_beams))
    all_info_LOS = []
    all_info_NLOS = []

    for sample in range(len(all_data)):
        if all_data[sample,4]=='LOS=1':
            all_info_LOS.append(all_data[sample])
        else:
            all_info_NLOS.append(all_data[sample])

    data_LOS = np.array(all_info_LOS)
    data_NLOS = np.array(all_info_NLOS)

    limit_ep_train = 1564

    data_train_LOS = data_LOS[(data_LOS[:, 0] < limit_ep_train + 1)]
    data_test_LOS = data_LOS[(data_LOS[:, 0] > limit_ep_train)]

    data_train_NLOS = data_NLOS[(data_NLOS[:, 0] < limit_ep_train + 1)]
    data_test_NLOS = data_NLOS[(data_NLOS[:, 0] > limit_ep_train)]

    beam_LOS_tx_train = data_train_LOS[:, -1].astype(int)
    beam_LOS_rx_train = data_train_LOS[:, -2].astype(int)

    beam_LOS_tx_test = data_test_LOS[:, -1].astype(int)
    beam_LOS_rx_test = data_test_LOS[:, -2].astype(int)

    beam_NLOS_tx_train = data_train_NLOS[:, -1].astype(int)
    beam_NLOS_rx_train = data_train_NLOS[:, -2].astype(int)

    beam_NLOS_tx_test = data_test_NLOS[:, -1].astype(int)
    beam_NLOS_rx_test = data_test_NLOS[:, -2].astype(int)


    if save_data:

        save_path = '/Users/Joanna/git/beam_selection_wisard/data/beams/LOS_index_beam/'
        np.savez(save_path + 'beam_LOS_tx_train' + '.npz', beam_LOS_tx_train=beam_LOS_tx_train)
        np.savez(save_path + 'beam_LOS_rx_train' + '.npz', beam_LOS_rx_train=beam_LOS_rx_train)

        np.savez(save_path + 'beam_LOS_tx_test' + '.npz', beam_LOS_tx_test=beam_LOS_tx_test)
        np.savez(save_path + 'beam_LOS_rx_test' + '.npz', beam_LOS_rx_test=beam_LOS_rx_test)

        save_path = '/Users/Joanna/git/beam_selection_wisard/data/beams/NLOS_index_beam/'
        np.savez(save_path + 'beam_NLOS_tx_train' + '.npz', beam_NLOS_tx_train=beam_NLOS_tx_train)
        np.savez(save_path + 'beam_NLOS_rx_train' + '.npz', beam_NLOS_rx_train=beam_NLOS_rx_train)

        np.savez(save_path + 'beam_NLOS_tx_test' + '.npz', beam_NLOS_tx_test=beam_NLOS_tx_test)
        np.savez(save_path + 'beam_NLOS_rx_test' + '.npz', beam_NLOS_rx_test=beam_NLOS_rx_test)


    return data_train_LOS, data_train_NLOS, data_test_LOS, data_test_NLOS

