import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

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

def position_just_rx(coordenadas):

    ''''Este metodo recibe as coordenadas (EpisodeID, coord_x, coord_y, coord_z, LOS, val)'''

    valid_conection = 5
    separated_dataset_LOS_NLOS = True

    min_coord_x = np.min(coordenadas[:,1])
    min_coord_y = np.min(coordenadas[:,2])

    max_coord_x = np.max(coordenadas[:,1])
    max_coord_y = np.max(coordenadas[:,2])

    numOfSamples = len(coordenadas)
    num_of_valid_samples = len(coordenadas[coordenadas[:, 5] == 'V'])
    area_interese_mtrs = np.array([23, 250], dtype=int)
    num_positions_of_internal_matrix = area_interese_mtrs[0] * area_interese_mtrs[1]

    coordenadas_x_mtrs = np.array((coordenadas[:, 1] - min_coord_x) / max_coord_x * float(area_interese_mtrs[0]), dtype=float)
    coordenadas_y_mtrs = np.array((coordenadas[:, 2] - min_coord_y) / max_coord_y * float(area_interese_mtrs[1]), dtype=float)
    coordenadas_x_mtrs = np.round(coordenadas_x_mtrs)
    coordenadas_y_mtrs = np.round(coordenadas_y_mtrs)

    Qs_all = np.zeros(area_interese_mtrs, dtype=int)  # Matriz que representa a posicao de TODOS os veiculos num episodio
    Qs_valid = np.zeros(area_interese_mtrs, dtype=int)  # Matriz que representa a posicao de TODOS os veiculos com conexao valida num episodio

    coord_qs_all            = np.zeros((num_of_valid_samples, num_positions_of_internal_matrix), dtype=int)
    coord_qs_valid          = np.zeros((num_of_valid_samples, num_positions_of_internal_matrix), dtype=int)
    coord_qs_valid_unique   = np.zeros((num_of_valid_samples, num_positions_of_internal_matrix), dtype=int)

    data_for_train = coordenadas[coordenadas[:,0]<1565]
    data_for_test  = coordenadas[coordenadas[:,0]>1564]

    num_of_samples_train = len(data_for_train)

    num_of_valid_communications_in_actual_episode = 0
    total_valid_communications = 0
    Qs_unico_array = []
    actual_episode =0

    id_episodes_flagLOS = np.zeros((num_of_valid_samples,2), dtype=object)

    for cont_samples in range(numOfSamples):

        Qs_unique = np.zeros(area_interese_mtrs)  # Matriz que representa a posicao de um veiculo

        x = int(coordenadas_x_mtrs[cont_samples])
        y = int(coordenadas_y_mtrs[cont_samples])

        sample_episode = int(coordenadas[cont_samples, 0])

        if sample_episode == actual_episode:
            Qs_all[x, y] = 1

            if coordenadas[cont_samples, 5] == 'V':

                num_of_valid_communications_in_actual_episode += 1


                Qs_unique[x, y] = 1
                Qs_valid[x, y] = 1

                Qs_unico_array.append(Qs_unique)
        else:
            vector_qs = Qs_all.reshape(1, num_positions_of_internal_matrix)
            vector_qs_valid = Qs_valid.reshape(1, num_positions_of_internal_matrix)


            for i in range(num_of_valid_communications_in_actual_episode):
                vector_qs_valid_unique = Qs_unico_array[i].reshape(1, num_positions_of_internal_matrix)
                id_episodes_flagLOS[total_valid_communications+i] = actual_episode, coordenadas[cont_samples, 4]
                coord_qs_all[total_valid_communications + i] = vector_qs
                coord_qs_valid[total_valid_communications + i] = vector_qs_valid
                coord_qs_valid_unique[total_valid_communications + i] = vector_qs_valid_unique

            total_valid_communications += num_of_valid_communications_in_actual_episode
            num_of_valid_communications_in_actual_episode = 0
            actual_episode = actual_episode + 1

            Qs_unico_array = []
            Qs_all = np.zeros(area_interese_mtrs)  # Matriz que representa a posicao de TODOS os veiculos num episodio
            Qs_valid = np.zeros(area_interese_mtrs)  # Matriz que representa a posicao de TODOS os veiculos com conexao valida num episodio

            Qs_all[x, y] = 1
            if coordenadas[cont_samples, 1] == 'V':

                num_of_valid_communications_in_actual_episode += 1

                Qs_unique[x, y] = 1
                Qs_valid[x, y] = 1

                Qs_unico_array.append(Qs_unique)

    if separated_dataset_LOS_NLOS:
        all_dataset = np.concatenate((id_episodes_flagLOS, coord_qs_all, coord_qs_valid_unique), axis=1)
    # output = coord_qs_all
    # output = coord_qs_valid
    # output = coord_qs_valid_unique
    # output = np.concatenate((coord_qs_all, coord_qs_valid), 1)
    output = np.concatenate((coord_qs_all, coord_qs_valid_unique), axis=1)
    #output = np.concatenate((coord_qs_valid, coord_qs_valid_unique), 1)



    #-------------------------------------------------------------

    '''Cria as matrizes Qs de TODO o banco de daos'''
    for cont_samples in range(numOfSamples):

        Qs_unique = np.zeros(area_interese_mtrs)  # Matriz que representa a posicao de um veiculo

        x = int(coordenadas_x_mtrs[cont_samples])
        y = int(coordenadas_y_mtrs[cont_samples])

        sample_episode = int(coordenadas[cont_samples, 0])

        if sample_episode == actual_episode:
            Qs_all[x, y] = 1

            if coordenadas[cont_samples, 5] == 'V':
                num_of_valid_communications_in_actual_episode += 1

                Qs_unique[x, y] = 1
                Qs_valid[x, y] = 1

                Qs_unico_array.append(Qs_unique)
        else:
            vector_qs = Qs_all.reshape(1, num_positions_of_internal_matrix)
            vector_qs_valid = Qs_valid.reshape(1, num_positions_of_internal_matrix)

            for i in range(num_of_valid_communications_in_actual_episode):
                vector_qs_valid_unique = Qs_unico_array[i].reshape(1, num_positions_of_internal_matrix)

                coord_qs_all[total_valid_communications + i] = vector_qs
                coord_qs_valid[total_valid_communications + i] = vector_qs_valid
                coord_qs_valid_unique[total_valid_communications + i] = vector_qs_valid_unique

            total_valid_communications += num_of_valid_communications_in_actual_episode
            num_of_valid_communications_in_actual_episode = 0
            actual_episode = actual_episode + 1

            Qs_unico_array = []
            Qs_all = np.zeros(area_interese_mtrs)  # Matriz que representa a posicao de TODOS os veiculos num episodio
            Qs_valid = np.zeros(area_interese_mtrs)  # Matriz que representa a posicao de TODOS os veiculos com conexao valida num episodio

            Qs_all[x, y] = 1
            if coordenadas[cont_samples, 1] == 'V':
                num_of_valid_communications_in_actual_episode += 1

                Qs_unique[x, y] = 1
                Qs_valid[x, y] = 1

                Qs_unico_array.append(Qs_unique)

    # output = coord_qs_all
    # output = coord_qs_valid
    # output = coord_qs_valid_unique
    # output = np.concatenate((coord_qs_all, coord_qs_valid), 1)
    output = np.concatenate((coord_qs_all, coord_qs_valid_unique), axis=1)
    #output = np.concatenate((coord_qs_valid, coord_qs_valid_unique), 1)

    return output


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
    coordenadas_NLOS_train = []
    coordenadas_LOS_test = []
    coordenadas_NLOS_test = []
    if separed_coord_LOS:
        for i in range(len(train)):
            if train[i][0] == 'LOS=1':
                coordenadas_LOS_train.append(train[i][1])
            else:
                coordenadas_NLOS_train.append(train[i][1])
        for i in range(len(test)):
            if test[i][0] == 'LOS=1':
                coordenadas_LOS_test.append(test[i][1])
            else:
                coordenadas_NLOS_test.append(test[i][1])




    return train, test, coordenadas_LOS_train, coordenadas_NLOS_train, coordenadas_LOS_test, coordenadas_NLOS_test






