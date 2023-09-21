import os

import numpy as np
import matplotlib.pyplot as plt
import csv
from math import modf


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
    #filename = '/Users/Joanna/git/Analise_de_dados/data/coordinates/CoordVehiclesRxPerScene_s008.csv'
    filename ='../data/coordinates/CoordVehiclesRxPerScene_s008.csv'
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

    return all_info_coord_val, coord_train, coord_test

def read_only_coord(all_info_coord_val):

    only_coord = all_info_coord_val[:,[1,2,3]]

    return only_coord

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

def pre_process_coordinates_in_lines(separed_coord_LOS):
    print_positions = True
    coordenadas = read_all_coordinates()
    ''''Este metodo recibe o obj coordenadas com a ordem dos seguintes dados
            (EpisodeID, coord_x, coord_y, coord_z, LOS, val)'''

    min_coord_x = np.min(coordenadas[:, 1])
    min_coord_y = np.min(coordenadas[:, 2])

    max_coord_x = np.max(coordenadas[:, 1])
    max_coord_y = np.max(coordenadas[:, 2])

    area_interese_mtrs = np.array([23, 250], dtype=int)
    #num_positions_of_internal_matrix = area_interese_mtrs[0] * area_interese_mtrs[1]
    area_interes_pre_process = np.array([2, 120], dtype=int)
    num_positions_of_internal_matrix = area_interes_pre_process[0] * area_interes_pre_process[1]

    coord_x_mtrs = np.array((coordenadas[:, 1] - min_coord_x) / max_coord_x * float(area_interese_mtrs[0]),dtype=float)
    coord_y_mtrs = np.array((coordenadas[:, 2] - min_coord_y) / max_coord_y * float(area_interese_mtrs[1]),dtype=float)
    all_x_coord_in_metros = np.round(coord_x_mtrs)
    all_y_coord_in_metros = np.round(coord_y_mtrs)

    average = np.average(all_x_coord_in_metros)
    sample_size = len(coordenadas)
    #x_line_position = np.zeros((sample_size, 1),   dtype=int)
    #y_line_position = np.zeros((sample_size, 120), dtype=int)
    x_all_position=[]
    all_coord_in_Qs_lines = []

    for i in range(sample_size):
        if all_x_coord_in_metros[i] < average:
             line = 0
        else:
            line = 1

        x_all_position.append(line)

    amostra = 0
    Qs_all = np.zeros(area_interes_pre_process, dtype=int)

    for i in range(2086):
        for a in range(10):
            x = x_all_position[amostra]
            y = int(all_y_coord_in_metros[amostra])
            Qs_all[x, y] = 1
            amostra += 1

        episodio = coordenadas[amostra-1,0]
        flag_valid_coord = coordenadas[amostra-1, 5]
        flag_LOS_or_NLOS = coordenadas[amostra-1, 4]

        all_coord_in_Qs_lines.append([episodio, flag_valid_coord, flag_LOS_or_NLOS, Qs_all.reshape(1, num_positions_of_internal_matrix)])
        Qs_all = np.zeros(area_interes_pre_process, dtype=int)

    coordenadas_validas = coordenadas[coordenadas[:, 5] == 'V']
    coord_val_x_mtrs = np.array((coordenadas_validas[:, 1] - min_coord_x) / max_coord_x * float(area_interese_mtrs[0]), dtype=float)
    coord_val_y_mtrs = np.array((coordenadas_validas[:, 2] - min_coord_y) / max_coord_y * float(area_interese_mtrs[1]), dtype=float)
    coord_val_x_mtrs = np.round(coord_val_x_mtrs)
    coord_val_y_mtrs = np.round(coord_val_y_mtrs)
    coord_val_Qs = []
    x_position_val= []

    for i in range(len(coordenadas_validas)):
        if coord_val_x_mtrs[i] < average:
             line = 0
        else:
            line = 1

        x_position_val.append(line)

    Qs_val = np.zeros(area_interes_pre_process, dtype=int)

    for i in range(len(coordenadas_validas)):
        x_val = int(x_position_val[i])
        y_val = int(coord_val_y_mtrs[i])
        Qs_val[x_val, y_val] = 1

        episodio_val = coordenadas_validas[i,0]
        flag_LOS_or_NLOS_val = coordenadas_validas[i,4]

        coord_val_Qs.append([episodio_val, flag_LOS_or_NLOS_val, Qs_val.reshape(1, num_positions_of_internal_matrix)])
        Qs_val = np.zeros(area_interes_pre_process, dtype=int)

    Qs = []
    for j in range(len(all_coord_in_Qs_lines )):
        for i in range(len(coord_val_Qs)):
            if (all_coord_in_Qs_lines[j][0] == coord_val_Qs[i][0]):
                z = np.concatenate([coord_val_Qs[i][2], all_coord_in_Qs_lines[j][3]], axis=1)
                Qs.append([coord_val_Qs[i][1], z])

    train = Qs[:9234]
    test = Qs[9234:]

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

    save_Qs_matrix_in_lines(coordenadas_all_train, coordenadas_all_test, coordenadas_LOS_train, coordenadas_LOS_test, coordenadas_NLOS_train, coordenadas_NLOS_test)



    # if print_positions==True: #Nao esta funcionando direito
    #     #all_x_coord = veicles_positions[:,0]
    #     #all_y_coord = np.array(veicles_positions[:,1:120])
    #     x = x_line_position[0:9]
    #     y = y_coordinate_in_metros[0:9]
    #     plt.scatter(x,y, label='All')
    #     plt.legend()
    #
    #     #plt.title("Coordenadas da amostra: " + str(sample) + " do episodio: " + str(id_episode))
    #     plt.title("posicao dos receptores")
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.xlim([-0.5, 2])
    #     plt.ylim([0, 122])
    #     plt.axvline(0.5, color='r', linestyle='dashed')
    #     #plt.text(20, mean_x + 3, str(mean_x), color='r')
    #     plt.grid(True)
    #     plt.show()

    return coordenadas_all_train, coordenadas_all_test, coordenadas_LOS_train, coordenadas_NLOS_train, coordenadas_LOS_test, coordenadas_NLOS_test

def save_Qs_matrix_in_lines(all_train_coord, all_test_coord, LOS_train_coord, LOS_test_coord, NLOS_train_coord, NLOS_test_coord):
    save_path = "../data/coordinates/processed/Qs_in_lines/"

    # Todas as coordenadas em formato Qs
    np.savez(save_path + 'all_train_coord_in_lines' + '.npz', coord_training=all_train_coord)
    np.savez(save_path + 'all_test_coord_in_lines' + '.npz', coor_test=all_test_coord)

    # coordenadas LOS em formato Qs
    np.savez(save_path + 'LOS_train_coord_in_lines' + '.npz', coord_LOS_training=LOS_train_coord)
    np.savez(save_path + 'LOS_test_coord_in_lines' + '.npz', coor_LOS_test=LOS_test_coord)

    # coordenadas NLOS em formato Qs
    np.savez(save_path + 'NLOS_train_coord_in_lines' + '.npz', coord_NLOS_training=NLOS_train_coord)
    np.savez(save_path + 'NLOS_test_coord_in_lines' + '.npz', coor_NLOS_test=NLOS_test_coord)

def read_beams_raymobtime(num_antennas_tx, num_antennas_rx):

    config_antenna = num_antennas_rx+'x'+num_antennas_tx
    #data_path = '/Users/Joanna/git/beam_selection_wisard/data/beams/Ailton/beam_output/beams_output_'+config_antenna+'.npz'
    data_path = '../data/beams/Ailton/beam_output/beams_output_'+config_antenna+'.npz'
    beams = np.load(data_path)['output_classification']

    best_beam_index = []
    for sample in range(beams.shape[0]):
        best_beam_index.append(np.argmax(beams[sample, :]))

    beam_index_rx = np.array(best_beam_index)

    tx_index = np.zeros((beams.shape[0]), dtype=int)
    rx_index = np.zeros((beams.shape[0]), dtype=int)

    for sample in range(len(beam_index_rx)):
        index_tx = best_beam_index[sample] // int(num_antennas_rx)
        index_rx = best_beam_index[sample] % int(num_antennas_rx)
        tx_index[sample] = index_tx
        rx_index[sample] = index_rx

    return tx_index, rx_index, best_beam_index

def divide_beams_in_train_test(rx_beams, tx_beams, combined_index, coord, antenna_config, save_data):


    all_data = np.column_stack((coord, rx_beams, tx_beams, combined_index))

    limit_ep_train = 1564

    data_train = all_data[(all_data[:, 0] < limit_ep_train + 1)]
    data_test = all_data[(all_data[:, 0] > limit_ep_train)]

    all_beam_combined_train = data_train[:,-1].astype(int)
    all_beam_combined_test = data_test[:,-1].astype(int)

    all_beam_tx_train =data_train[:,-2].astype(int)
    all_beam_tx_test =data_test[:,-2].astype(int)

    all_beam_rx_train =data_train[:,-3].astype(int)
    all_beam_rx_test =data_test[:,-3].astype(int)

    if save_data:

        all_beam_combined_train = data_train[:, -1]
        all_beam_combined_test = data_test[:, -1]

        all_beams_tx_train = data_train[:, -2]
        all_beams_tx_test = data_test[:, -2]

        all_beams_rx_train = data_train[:, -3]
        all_beams_rx_test = data_test[:, -3]


        #---------- esta parte nao esta funcionando ------------
        #save_path = '/Users/Joanna/git/beam_selection_wisard/data/beams/'+antenna_config+'/all_index_beam/'
        '''
        save_path = '../data/beam/'+antenna_config+'/'
        save_path1 = '../data/beam/'+antenna_config+'/all_index_beam/'
        try:
            os.mkdir(save_path)
            os.mkdir(save_path1)

        except OSError:
            print('Pasta %s NAO foi criada' %save_path)
            print('Pasta %s NAO foi criada' % save_path1)
        else:
            print('Pasta %s criada' %save_path)
            print('Pasta %s criada' % save_path1)

        #-------------------------------------------------------
        '''
        save_path = '../data/beams/' + antenna_config + '/all_index_beam/'
        np.savez(save_path + 'index_beams_tx_train' + '.npz', all_beams_tx_train=all_beams_tx_train)
        np.savez(save_path + 'index_beams_tx_test' + '.npz', all_beams_tx_test=all_beams_tx_test)

        np.savez(save_path + 'index_beams_rx_train' + '.npz', all_beams_rx_train=all_beams_rx_train)
        np.savez(save_path + 'index_beams_rx_test' + '.npz', all_beams_rx_test=all_beams_rx_test)

        np.savez(save_path + 'index_beams_combined_train' + '.npz', all_beam_combined_train=all_beam_combined_train)
        np.savez(save_path + 'index_beams_combined_test' + '.npz', all_beam_combined_test=all_beam_combined_test)



    return all_beam_tx_train, all_beam_tx_test, all_beam_rx_train, all_beam_rx_test

def divide_valid_coord_in_LOS_NLOS_connection(valid_coord):
    all_info_LOS = []
    all_info_NLOS = []
    for sample in range(len(valid_coord)):
        if valid_coord[sample,4] == 'LOS=1':
            all_info_LOS.append(valid_coord[sample])
        else:
            all_info_NLOS.append(valid_coord[sample])

    data_LOS = np.array(all_info_LOS)
    data_NLOS = np.array(all_info_NLOS)

    limit_ep_train = 1564

    data_train_LOS = data_LOS[(data_LOS[:, 0] < limit_ep_train + 1)]
    data_test_LOS = data_LOS[(data_LOS[:, 0] > limit_ep_train)]

    data_train_NLOS = data_NLOS[(data_NLOS[:, 0] < limit_ep_train + 1)]
    data_test_NLOS = data_NLOS[(data_NLOS[:, 0] > limit_ep_train)]

    return data_train_LOS, data_test_LOS, data_train_NLOS, data_test_NLOS

def divide_beams_and_coord_in_LOS_or_NLOS_connect(rx_beams, tx_beams, combined_index, coord, antenna_config, save_data):
    '''Metodo que genera y guarda las coordenadas y beams para train y test'''

    all_data = np.column_stack((coord, rx_beams, tx_beams, combined_index))
    all_info_LOS = []
    all_info_NLOS = []

    for sample in range(len(all_data)):
        if all_data[sample,4] == 'LOS=1':
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

    beam_LOS_combined_train = data_train_LOS[:, -1].astype(int)
    beam_LOS_tx_train = data_train_LOS[:, -2].astype(int)
    beam_LOS_rx_train = data_train_LOS[:, -3].astype(int)

    beam_LOS_combined_test = data_test_LOS[:, -1].astype(int)
    beam_LOS_tx_test = data_test_LOS[:, -2].astype(int)
    beam_LOS_rx_test = data_test_LOS[:, -3].astype(int)

    beam_NLOS_combined_train = data_train_NLOS[:, -1].astype(int)
    beam_NLOS_tx_train = data_train_NLOS[:, -2].astype(int)
    beam_NLOS_rx_train = data_train_NLOS[:, -3].astype(int)

    beam_NLOS_combined_test = data_test_NLOS[:, -1].astype(int)
    beam_NLOS_tx_test = data_test_NLOS[:, -2].astype(int)
    beam_NLOS_rx_test = data_test_NLOS[:, -3].astype(int)


    if save_data:

        #save_path = '/Users/Joanna/git/beam_selection_wisard/data/beams/'+antenna_config+'/LOS_index_beam/'
        save_path = '../data/beams/'+antenna_config+'/LOS_index_beam/'
        np.savez(save_path + 'beam_LOS_tx_train' + '.npz', beam_LOS_tx_train=beam_LOS_tx_train)
        np.savez(save_path + 'beam_LOS_rx_train' + '.npz', beam_LOS_rx_train=beam_LOS_rx_train)

        np.savez(save_path + 'beam_LOS_tx_test' + '.npz', beam_LOS_tx_test=beam_LOS_tx_test)
        np.savez(save_path + 'beam_LOS_rx_test' + '.npz', beam_LOS_rx_test=beam_LOS_rx_test)

        np.savez(save_path + 'beam_LOS_combined_train' + '.npz', beam_LOS_combined_train=beam_LOS_combined_train)
        np.savez(save_path + 'beam_LOS_combined_test' + '.npz', beam_LOS_combined_test=beam_LOS_combined_test)

        #save_path = '/Users/Joanna/git/beam_selection_wisard/data/beams/'+antenna_config+'/NLOS_index_beam/'
        save_path = '../data/beams/' + antenna_config + '/NLOS_index_beam/'
        np.savez(save_path + 'beam_NLOS_tx_train' + '.npz', beam_NLOS_tx_train=beam_NLOS_tx_train)
        np.savez(save_path + 'beam_NLOS_rx_train' + '.npz', beam_NLOS_rx_train=beam_NLOS_rx_train)

        np.savez(save_path + 'beam_NLOS_tx_test' + '.npz', beam_NLOS_tx_test=beam_NLOS_tx_test)
        np.savez(save_path + 'beam_NLOS_rx_test' + '.npz', beam_NLOS_rx_test=beam_NLOS_rx_test)

        np.savez(save_path + 'beam_NLOS_combined_train' + '.npz', beam_NLOS_combined_train=beam_NLOS_combined_train)
        np.savez(save_path + 'beam_NLOS_combined_test' + '.npz', beam_NLOS_combined_test=beam_NLOS_combined_test)

    return data_train_LOS, data_train_NLOS, data_test_LOS, data_test_NLOS,

def  Thermomether():
    #int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates()

    episodios = all_info_coord_val[:,0]

    all_x_coord_str = all_info_coord_val[:,1]
    all_x_coord = [int(x) for x in all_x_coord_str]
    all_y_coord_str = all_info_coord_val[:, 2]
    all_y_coord = [int(y) for y in all_y_coord_str]


    min_x_coord = np.min(all_x_coord)
    max_x_coord = np.max(all_x_coord)

    min_y_coord = np.min(all_y_coord)
    max_y_coord = np.max(all_y_coord)

    enconding = np.array([len(all_info_coord_val), 20], dtype=int)
    encoding_x = np.zeros(enconding, dtype=int)
    encoding_y = np.zeros(np.array([len(all_info_coord_val), 245], dtype=int), dtype=int)

    #encoding_x = np.zeros(np.array([10, 20], dtype='b'), dtype='b')
    #encoding_y = np.zeros(np.array([10, 245], dtype='b'), dtype='b')

    sample = 0
    result_x = 0
    for i in all_x_coord:
        result_x = i-min_x_coord
        for j in range(result_x):
            encoding_x[sample, j] = True

        sample = sample+1

    sample = 0
    result_y = 0
    for i in all_y_coord:
        result_y = i - min_y_coord
        for j in range(result_y):
            encoding_y[sample, j] = True

        sample = sample + 1


    encondig_coord = np.concatenate((encoding_x,encoding_y), axis=1)
    encoding_coord_and_episode = np.column_stack([episodios, encondig_coord])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] < limit_ep_train + 1)]
    encondign_coord_test = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] > limit_ep_train)]

    encondign_coord_train = encondign_coord_train[:,1:266]
    encondign_coord_test = encondign_coord_test[:,1:266]


    return encondign_coord_train, encondign_coord_test

def  Thermomether_dobro_resolucao():
    #int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates()

    episodios = all_info_coord_val[:,0]

    all_x_coord_str = all_info_coord_val[:,1]
    all_x_coord = [int(x) for x in all_x_coord_str]
    all_y_coord_str = all_info_coord_val[:, 2]
    all_y_coord = [int(y) for y in all_y_coord_str]


    min_x_coord = np.min(all_x_coord)
    max_x_coord = np.max(all_x_coord)

    min_y_coord = np.min(all_y_coord)
    max_y_coord = np.max(all_y_coord)

    enconding = np.array([len(all_info_coord_val), 160], dtype=int)
    encoding_x = np.zeros(enconding, dtype=int)
    encoding_y = np.zeros(np.array([len(all_info_coord_val), 1960], dtype=int), dtype=int)

    sample = 0
    result_x = 0
    for i in all_x_coord:
        result_x = i-min_x_coord
        for j in range(result_x*8):
            encoding_x[sample, j] = 1

        sample = sample+1

    sample = 0
    result_y = 0
    for i in all_y_coord:
        result_y = i - min_y_coord
        for j in range(result_y*8):
            encoding_y[sample, j] = 1

        sample = sample + 1


    encondig_coord = np.concatenate((encoding_x,encoding_y), axis=1)
    encoding_coord_and_episode = np.column_stack([episodios, encondig_coord])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] < limit_ep_train + 1)]
    encondign_coord_test = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] > limit_ep_train)]

    size_of_input = encondign_coord_train.shape
    encondign_coord_train = encondign_coord_train[:,1:size_of_input[1]]
    encondign_coord_test = encondign_coord_test[:,1:size_of_input[1]]


    return encondign_coord_train, encondign_coord_test

def  Thermomether_parte_inteira_parte_decimal():
    #int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates()

    episodios = all_info_coord_val[:,0]

    all_x_coord_str = all_info_coord_val[:,1]
    all_x_coord = [float(x) for x in all_x_coord_str]
    all_y_coord_str = all_info_coord_val[:, 2]
    all_y_coord = [float(y) for y in all_y_coord_str]


    min_x_coord = np.min(all_x_coord)
    max_x_coord = np.max(all_x_coord)

    min_y_coord = np.min(all_y_coord)
    max_y_coord = np.max(all_y_coord)


    diff_all_x_coord = np.array((all_x_coord-min_x_coord))
    diff_all_y_coord = np.array((all_y_coord-min_y_coord))

    enconding = np.array([len(all_info_coord_val), 20], dtype=int)
    encoding_x_whole_part = np.zeros(enconding, dtype=int)
    encoding_x_decimal_part = np.zeros(np.array([len(all_info_coord_val), 100], dtype=int), dtype=int)
    encoding_y_whole_part = np.zeros(np.array([len(all_info_coord_val), 245], dtype=int), dtype=int)
    encoding_y_decimal_part = np.zeros(np.array([len(all_info_coord_val), 100], dtype=int), dtype=int)

    sample = 0

    for i in diff_all_x_coord:
        whole_part = int(i)
        for j in range(whole_part):
            encoding_x_whole_part[sample, j] = 1
        sample = sample+1

    sample = 0
    for i in diff_all_x_coord:
        decimal_part = int(round(modf(i)[0],2)*100)
        for j in range(decimal_part):
            encoding_x_decimal_part[sample, j] = 1
        sample = sample+1

    encondig_x_coord = np.concatenate((encoding_x_whole_part, encoding_x_decimal_part), axis=1)

    sample = 0
    for i in diff_all_y_coord:
        whole_part = int(i)
        for j in range(whole_part):
            encoding_y_whole_part[sample, j] = 1
        sample = sample+1

    sample = 0
    for i in diff_all_y_coord:
        decimal_part = int(round(modf(i)[0], 2) * 100)
        for j in range(decimal_part):
            encoding_y_decimal_part[sample, j] = 1
        sample = sample + 1

    encondig_y_coord = np.concatenate((encoding_y_whole_part, encoding_y_decimal_part), axis=1)

    encondig_coord = np.concatenate((encondig_x_coord, encondig_y_coord), axis=1)
    encoding_coord_and_episode = np.column_stack([episodios, encondig_coord])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] < limit_ep_train + 1)]
    encondign_coord_test = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] > limit_ep_train)]

    size_of_input = encondign_coord_train.shape
    encondign_coord_train = encondign_coord_train[:,1:size_of_input[1]]
    encondign_coord_test = encondign_coord_test[:,1:size_of_input[1]]


    return encondign_coord_train, encondign_coord_test

def  Thermomether_parte_inteira_mais_parte_decimal():
    #int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates()

    episodios = all_info_coord_val[:,0]

    all_x_coord_str = all_info_coord_val[:,1]
    all_x_coord = [float(x) for x in all_x_coord_str]
    all_y_coord_str = all_info_coord_val[:, 2]
    all_y_coord = [float(y) for y in all_y_coord_str]


    min_x_coord = np.min(all_x_coord)
    max_x_coord = np.max(all_x_coord)

    min_y_coord = np.min(all_y_coord)
    max_y_coord = np.max(all_y_coord)


    diff_all_x_coord = np.array((all_x_coord-min_x_coord))
    diff_all_y_coord = np.array((all_y_coord-min_y_coord))

    enconding = np.array([len(all_info_coord_val), 20], dtype=int)
    encoding_x_whole_part = np.zeros(enconding, dtype=int)
    encoding_x = np.zeros(np.array([len(all_info_coord_val), 30], dtype=int), dtype=int)
    encoding_y = np.zeros(np.array([len(all_info_coord_val), 255], dtype=int), dtype=int)
    encoding_y_decimal_part = np.zeros(np.array([len(all_info_coord_val), 100], dtype=int), dtype=int)

    sample = 0

    for i in diff_all_x_coord:
        whole_part = int(i)
        decimal_part = int(round(modf(i)[0], 1) * 10)
        number = whole_part+decimal_part


        for j in range(number):
            encoding_x[sample, j] = 1
        sample = sample+1



    sample = 0
    for i in diff_all_y_coord:
        whole_part = int(i)
        decimal_part = int(round(modf(i)[0], 1) * 10)
        number = whole_part + decimal_part
        for j in range(number):
            encoding_y[sample, j] = 1
        sample = sample+1



    encondig_coord = np.concatenate((encoding_x, encoding_y), axis=1)
    encoding_coord_and_episode = np.column_stack([episodios, encondig_coord])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] < limit_ep_train + 1)]
    encondign_coord_test = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] > limit_ep_train)]

    size_of_input = encondign_coord_train.shape
    encondign_coord_train = encondign_coord_train[:,1:size_of_input[1]]
    encondign_coord_test = encondign_coord_test[:,1:size_of_input[1]]


    return encondign_coord_train, encondign_coord_test

def  Thermomether_parte_inteira_mais_parte_decimal_1():
    #int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates()

    episodios = all_info_coord_val[:,0]

    all_x_coord_str = all_info_coord_val[:,1]
    all_x_coord = [float(x) for x in all_x_coord_str]
    all_y_coord_str = all_info_coord_val[:, 2]
    all_y_coord = [float(y) for y in all_y_coord_str]


    min_x_coord = np.min(all_x_coord)
    max_x_coord = np.max(all_x_coord)

    min_y_coord = np.min(all_y_coord)
    max_y_coord = np.max(all_y_coord)


    diff_all_x_coord = np.array((all_x_coord-min_x_coord))
    diff_all_y_coord = np.array((all_y_coord-min_y_coord))

    n = 0

    enconding = np.array([len(all_info_coord_val), 20+n], dtype=int)
    #new_enconding = np.array([len(all_info_coord_val), 20+n], dtype='i')
    encoding_x_whole_part = np.zeros(enconding, dtype=int)
    encoding_x = np.zeros(np.array([len(all_info_coord_val), 20+n], dtype=int), dtype=int)
    encoding_y = np.zeros(np.array([len(all_info_coord_val), 245+n], dtype=int), dtype=int)
    encoding_y_decimal_part = np.zeros(np.array([len(all_info_coord_val), 100], dtype=int), dtype=int)

    sample = 0


    for i in diff_all_x_coord:
        whole_part = int(i)
        decimal_part = int(round(modf(i)[0], 1) * 10)

        if decimal_part >= 5:
            number = whole_part+n
        else:
            number = whole_part

        for j in range(number):
            encoding_x[sample, j] = 1
        sample = sample+1



    sample = 0
    for i in diff_all_y_coord:
        whole_part = int(i)
        decimal_part = int(round(modf(i)[0], 1) * 10)
        if decimal_part >= 5:
            number = whole_part + n
        else:
            number = whole_part
        for j in range(number):
            encoding_y[sample, j] = 1
        sample = sample+1



    encondig_coord = np.concatenate((encoding_x, encoding_y), axis=1)
    encoding_coord_and_episode = np.column_stack([episodios, encondig_coord])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] < limit_ep_train + 1)]
    encondign_coord_test = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] > limit_ep_train)]

    size_of_input = encondign_coord_train.shape
    encondign_coord_train = encondign_coord_train[:,1:size_of_input[1]]
    encondign_coord_test = encondign_coord_test[:,1:size_of_input[1]]


    return encondign_coord_train, encondign_coord_test

def  Thermomether_parte_inteira_mais_parte_decimal_2():
    #int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates()

    episodios = all_info_coord_val[:,0]

    all_x_coord_str = all_info_coord_val[:,1]
    all_x_coord = [float(x) for x in all_x_coord_str]
    all_y_coord_str = all_info_coord_val[:, 2]
    all_y_coord = [float(y) for y in all_y_coord_str]


    min_x_coord = np.min(all_x_coord)
    max_x_coord = np.max(all_x_coord)

    min_y_coord = np.min(all_y_coord)
    max_y_coord = np.max(all_y_coord)


    diff_all_x_coord = np.array((all_x_coord-min_x_coord))
    diff_all_y_coord = np.array((all_y_coord-min_y_coord))

    n = 0

    enconding = np.array([len(all_info_coord_val), 20+n], dtype=int)
    #new_enconding = np.array([len(all_info_coord_val), 20+n], dtype='i')
    encoding_x_whole_part = np.zeros(enconding, dtype=int)
    encoding_x = np.zeros(np.array([len(all_info_coord_val), 207], dtype=int), dtype=int)
    encoding_y = np.zeros(np.array([len(all_info_coord_val), 2457], dtype=int), dtype=int)
    encoding_y_decimal_part = np.zeros(np.array([len(all_info_coord_val), 100], dtype=int), dtype=int)

    sample = 0


    for i in diff_all_x_coord:

        #print(i)
        #print(round(i, 1)*10)
        number = int(round(i, 1)*10)

        for j in range(number):
            encoding_x[sample, j] = 1
        sample = sample+1



    sample = 0
    for i in diff_all_y_coord:
        number = int(round(i, 1) * 10)
        for j in range(number):
            encoding_y[sample, j] = 1
        sample = sample+1



    encondig_coord = np.concatenate((encoding_x, encoding_y), axis=1)
    encoding_coord_and_episode = np.column_stack([episodios, encondig_coord])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] < limit_ep_train + 1)]
    encondign_coord_test = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] > limit_ep_train)]

    size_of_input = encondign_coord_train.shape
    encondign_coord_train = encondign_coord_train[:,1:size_of_input[1]]
    encondign_coord_test = encondign_coord_test[:,1:size_of_input[1]]


    return encondign_coord_train, encondign_coord_test


def Thermomether_match_coor_x_and_y():
    # int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates()

    episodios = all_info_coord_val[:, 0]

    all_x_coord_str = all_info_coord_val[:, 1]
    all_x_coord = [float(x) for x in all_x_coord_str]
    all_y_coord_str = all_info_coord_val[:, 2]
    all_y_coord = [float(y) for y in all_y_coord_str]

    min_x_coord = np.min(all_x_coord)
    max_x_coord = np.max(all_x_coord)

    min_y_coord = np.min(all_y_coord)
    max_y_coord = np.max(all_y_coord)

    diff_all_x_coord = np.array((all_x_coord - min_x_coord))
    diff_all_y_coord = np.array((all_y_coord - min_y_coord))


    escala = 8
    size_of_data_x = 960*escala
    size_of_data_y = 980*escala
    enconding_x = np.array([len(all_info_coord_val), size_of_data_x], dtype=int)
    enconding_y = np.array([len(all_info_coord_val), size_of_data_y], dtype=int)

    encoding_x_whole_part = np.zeros(enconding_x, dtype=int)
    encoding_x_decimal_part = np.zeros(np.array([len(all_info_coord_val), 100], dtype=int), dtype=int)

    encoding_y_whole_part = np.zeros(enconding_y, dtype=int)
    encoding_y_decimal_part = np.zeros(np.array([len(all_info_coord_val), 100], dtype=int), dtype=int)

    sample = 0
    n_x = 12*escala
    for i in diff_all_x_coord:
        whole_part = int(i)
        for j in range(whole_part + n_x):
            encoding_x_whole_part[sample, j] = 1
        sample = sample + 1

    '''
    sample = 0
    for i in diff_all_x_coord:
        decimal_part = int(round(modf(i)[0], 2) * 100)
        for j in range(decimal_part):
            encoding_x_decimal_part[sample, j] = 1
        sample = sample + 1

    encondig_x_coord = np.concatenate((encoding_x_whole_part, encoding_x_decimal_part), axis=1)
    '''

    sample = 0
    n_y = 4*escala
    for i in diff_all_y_coord:
        whole_part = int(i)
        for j in range(whole_part + n_y):
            encoding_y_whole_part[sample, j] = 1
        sample = sample + 1

    '''
    sample = 0
    for i in diff_all_y_coord:
        decimal_part = int(round(modf(i)[0], 2) * 100)
        for j in range(decimal_part):
            encoding_y_decimal_part[sample, j] = 1
        sample = sample + 1

    encondig_y_coord = np.concatenate((encoding_y_whole_part, encoding_y_decimal_part), axis=1)
    '''

    encondig_coord = np.concatenate((encoding_x_whole_part, encoding_y_whole_part), axis=1)
    encoding_coord_and_episode = np.column_stack([episodios, encondig_coord])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] < limit_ep_train + 1)]
    encondign_coord_test = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] > limit_ep_train)]

    size_of_input = encondign_coord_train.shape
    encondign_coord_train = encondign_coord_train[:, 1:size_of_input[1]]
    encondign_coord_test = encondign_coord_test[:, 1:size_of_input[1]]

    return encondign_coord_train, encondign_coord_test

