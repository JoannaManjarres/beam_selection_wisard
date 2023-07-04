
import preprocesamento as preprocess
import read_data as read
import beam_selection_wisard as obj
import pre_process_lidar as obj_lidar
import analyse_data as obj_analyse
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌃' to toggle the breakpoint.


def preprocess_method(flag_pre_process, num_antennas_tx, num_antennas_rx, antenna_config):

    if flag_pre_process == 'S':
        flag_preprocess_coord = input("\t Realizar pre processamento das coordenadas? [S/N] \n").upper()
        flag_preprocess_beams_index = input("\t Realizar pre processamento dos indices dos Beams? [S/N] \n").upper()
        flag_preprocess_LiDAR = input("\t Realizar pre processamento dos dados LiDAR? [S/N] \n").upper()
        do_preprocess(flag_preprocess_coord,
                      flag_preprocess_beams_index,
                      flag_preprocess_LiDAR,
                      num_antennas_tx=num_antennas_tx,
                      num_antennas_rx=num_antennas_rx,
                      antenna_config=antenna_config)


def do_preprocess(flag_preprocess_coord,
                  flag_preprocess_beams_index,
                  flag_preprocess_LiDAR,
                  num_antennas_tx,
                  num_antennas_rx,
                  antenna_config):

    if flag_preprocess_beams_index =='S':
        valid_coord, coord_train, coord_test = preprocess.read_valid_coordinates()
        tx_index, rx_index, combined_index = preprocess.read_beams_raymobtime(num_antennas_tx, num_antennas_rx)
        preprocess.divide_beams_in_train_test(rx_index, tx_index, combined_index, valid_coord, antenna_config, save_data=True)
        preprocess.divide_beams_and_coord_in_LOS_or_NLOS_connect(rx_index, tx_index, combined_index, valid_coord, antenna_config, save_data=True)

    if flag_preprocess_coord =='S':

        print("pre processar em:")
        flag_matriz_type = input("\t [1] matriz Qs ou \n " \
                                 "\t [2] matriz Qs em linhas ")
        if (flag_matriz_type == '1'):
            coord = preprocess.read_all_coordinates()
            train, test, coordenadas_LOS_train, coordenadas_NLOS_train, coordenadas_LOS_test, coordenadas_NLOS_test = preprocess.coord_to_Qs_matrix(coord)
            preprocess.save_Qs_matrix(train, test, coordenadas_LOS_train, coordenadas_LOS_test, coordenadas_NLOS_train, coordenadas_NLOS_test)
            print("Coordenadas pre processadas em matriz Qs [23X250] e guardadas")

        if (flag_matriz_type == '2'):
            preprocess.pre_process_coordinates_in_lines(separed_coord_LOS=True)
            print("Coordenadas pre processadas em matriz Qs em linhas [2X120] e guardadas")

    if flag_preprocess_LiDAR == 'S':
        obj_lidar.process_data(enable_plot=False, num_scene_to_plot=0)
        obj_lidar.separed_data_lidar_LOS_NLOS()


def beam_selection(antenna_config, type_of_connection, type_of_input, flag_rx_or_tx):

    #flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
    #                      "\t Transmissor [T] ou Receptor [R]\n").upper()
    print("entre en BeamSelection")

    if type_of_connection == 'ALL':
        print("entre en all")
        index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test, index_beam_combined_train, index_beam_combined_test = read.read_all_beams(antenna_config)
        if flag_rx_or_tx == 'R':
            label_train = index_beam_rx_train
            label_test = index_beam_rx_test
        if flag_rx_or_tx == 'T':
            label_train = index_beam_tx_train
            label_test = index_beam_tx_test
        if flag_rx_or_tx == 'C':
            label_train = index_beam_combined_train
            label_test = index_beam_combined_test

        if type_of_input == 'coord_in_Qs':
            all_coord_train, all_coord_test = read.read_all_Qs_matrix()
            input_train = all_coord_train
            input_test = all_coord_test
        if type_of_input == 'coord_in_Qs_lines':
            all_coord_in_Qs_line_train, all_coord_in_Qs_line_test = read.read_all_Qs_matrix_in_lines()
            input_train = all_coord_in_Qs_line_train
            input_test = all_coord_in_Qs_line_test
        if type_of_input == 'lidar':
            data_lidar_train, data_lidar_validation = obj_lidar.read_all_LiDAR_data()
            input_train = data_lidar_train
            input_test = data_lidar_validation
        if type_of_input == 'coord_in_Qs_+_Lidar':
            data_lidar_train, data_lidar_validation = obj_lidar.read_all_LiDAR_data()
            all_coord_train, all_coord_test = read.read_all_Qs_matrix()
            input_train = np.column_stack((all_coord_train, data_lidar_train))
            input_test = np.column_stack((all_coord_test, data_lidar_validation))
        if type_of_input == 'coord_in_Qs_lines_+_Lidar':
            data_lidar_train, data_lidar_validation = obj_lidar.read_all_LiDAR_data()
            all_coord_in_Qs_line_train, all_coord_in_Qs_line_test = read.read_all_Qs_matrix_in_lines()
            input_train = np.column_stack((all_coord_in_Qs_line_train, data_lidar_train))
            input_test = np.column_stack((all_coord_in_Qs_line_test, data_lidar_validation))

    if type_of_connection == 'LOS':
        print('entre en LOS')
        index_beam_rx_LOS_train, index_beam_rx_LOS_test, index_beam_tx_LOS_train, index_beam_tx_LOS_test, index_beam_combined_LOS_train, index_beam_combined_LOS_test = read.read_LOS_beams(antenna_config)
        if flag_rx_or_tx == 'R':
            label_train = index_beam_rx_LOS_train
            label_test = index_beam_rx_LOS_test

        if flag_rx_or_tx == 'T':
            label_train = index_beam_tx_LOS_train
            label_test = index_beam_tx_LOS_test

        if flag_rx_or_tx == 'C':
            label_train = index_beam_combined_LOS_train
            label_test = index_beam_combined_LOS_test

        if type_of_input == 'coord_in_Qs':
            LOS_coord_train, LOS_coord_test = read.read_Qs_matrix_with_LOS_data()
            input_train = LOS_coord_train
            input_test = LOS_coord_test
        if type_of_input == 'coord_in_Qs_lines':
            LOS_coord_in_Qs_line_train, LOS_coord_in_Qs_line_test = read.read_Qs_matrix_in_lines_with_LOS_data()
            input_train = LOS_coord_in_Qs_line_train
            input_test = LOS_coord_in_Qs_line_test

        if type_of_input == 'lidar':
            data_lidar_LOS_train, data_lidar_LOS_test = obj_lidar.read_LiDAR_LOS_data()
            input_train = data_lidar_LOS_train
            input_test = data_lidar_LOS_test

        if type_of_input == 'coord_in_Qs_+_Lidar':
            LiDAR_LOS_train, LiDAR_LOS_test = obj_lidar.read_LiDAR_LOS_data()
            coord_LOS_train, coord_LOS_test = read.read_Qs_matrix_with_LOS_data()
            input_train = np.column_stack((coord_LOS_train, LiDAR_LOS_train))
            input_test = np.column_stack((coord_LOS_test, LiDAR_LOS_test))

        if type_of_input == 'coord_in_Qs_lines_+_Lidar':
            data_lidar_train, data_lidar_validation = obj_lidar.read_LiDAR_LOS_data()
            all_coord_in_Qs_line_train, all_coord_in_Qs_line_test = read.read_Qs_matrix_in_lines_with_LOS_data()
            input_train = np.column_stack((all_coord_in_Qs_line_train, data_lidar_train))
            input_test = np.column_stack((all_coord_in_Qs_line_test, data_lidar_validation))
    if type_of_connection == 'NLOS':
        print("entre en NLOS")
        index_beam_rx_NLOS_train, index_beam_rx_NLOS_test, index_beam_tx_NLOS_train, index_beam_tx_NLOS_test, index_beam_combined_NLOS_train, index_beam_combined_NLOS_test  = read.read_NLOS_beams(antenna_config)
        if flag_rx_or_tx == 'R':
            label_train = index_beam_rx_NLOS_train
            label_test = index_beam_rx_NLOS_test
        if flag_rx_or_tx == 'T':
            label_train = index_beam_tx_NLOS_train
            label_test = index_beam_tx_NLOS_test

        if flag_rx_or_tx == 'C':
            label_train = index_beam_combined_NLOS_train
            label_test = index_beam_combined_NLOS_test


        if type_of_input == 'coord_in_Qs':
            NLOS_coord_train, NLOS_coord_test = read.read_Qs_matrix_with_NLOS_data()
            input_train = NLOS_coord_train
            input_test = NLOS_coord_test

        if type_of_input == 'coord_in_Qs_lines':
            NLOS_coord_in_Qs_line_train, NLOS_coord_in_Qs_line_test = read.read_Qs_matrix_in_lines_with_NLOS_data()
            input_train = NLOS_coord_in_Qs_line_train
            input_test = NLOS_coord_in_Qs_line_test

        if type_of_input == 'lidar':
            data_lidar_NLOS_train, data_lidar_NLOS_test = obj_lidar.read_LiDAR_NLOS_data()
            input_train = data_lidar_NLOS_train
            input_test = data_lidar_NLOS_test

        if type_of_input == 'coord_in_Qs_+_Lidar':
            LiDAR_NLOS_train, LiDAR_NLOS_test = obj_lidar.read_LiDAR_NLOS_data()
            coord_NLOS_train, coord_NLOS_test = read.read_Qs_matrix_with_NLOS_data()
            input_train = np.column_stack((coord_NLOS_train, LiDAR_NLOS_train))
            input_test = np.column_stack((coord_NLOS_test, LiDAR_NLOS_test))

        if type_of_input == 'coord_in_Qs_lines_+_Lidar':
            LiDAR_NLOS_train, LiDAR_NLOS_test = obj_lidar.read_LiDAR_NLOS_data()
            coord_NLOS_train, coord_NLOS_test = read.read_Qs_matrix_in_lines_with_NLOS_data()
            input_train = np.column_stack((coord_NLOS_train, LiDAR_NLOS_train))
            input_test = np.column_stack((coord_NLOS_test, LiDAR_NLOS_test))


    user = ''
    if flag_rx_or_tx == 'T':
        user = 'Tx'
    elif flag_rx_or_tx == 'R':
        user = 'Rx'
    elif flag_rx_or_tx == 'C':
        user = 'Combined'


    title_figure = 'Seleção de Beam do ' + user + ' com antena em config ['+antenna_config+'] \n apartir de dados [' +type_of_input + '] com conexao ' + type_of_connection
    name_figure = 'beam_selection_'+user+'_['+antenna_config+']_'+type_of_input+'_'+type_of_connection
    obj.select_best_beam(input_train,
                         input_test,
                         label_train,
                         label_test,
                         figure_name = name_figure,
                         antenna_config=antenna_config,
                         type_of_input=type_of_input,
                         titulo_figura=title_figure,
                         user=user,
                         enableDebug=False,
                         plot_confusion_matrix_enable=False)

def beam_analysis(flag_beam_analysis, antenna_config,connection):

    if flag_beam_analysis == 'S':
        user = input('Escolha sobre qual usuario deseja realizar a análise do beam \n'
              '\t 1. Rx \n'
              '\t 2. Tx \n'
              '\t 3. Combinados \n')
        obj_analyse.beam_analyses(antenna_config, connection, user)


def run_simulation():
    print("Escolha a configuracao das antenas Rx-Tx")
    antennas_num_rx = input("Digite a quantidade de antenas no \n"
                            "\t Rx: 1, 2, 4, 8, 16 ou 32\n")
    antennas_num_tx = input("Digite a quantidade de antenas no \n"
                            "\t Tx: 2, 4, 8, 16 ou 32\n")
    antenna_config = antennas_num_rx + 'X' + antennas_num_tx

    print('--------------------------------------------')

    flag_type_of_connetion = input('Escolha o tipo de conexao que deseja analisar \n'
                                   '\t 1. ALL \n'
                                   '\t 2. LOS \n'
                                   '\t 3. NLOS\n')
    if flag_type_of_connetion == '1':
        connection = 'ALL'
    if flag_type_of_connetion == '2':
        connection = 'LOS'
    if flag_type_of_connetion == '3':
        connection = 'NLOS'

    print('--------------------------------------------')

    flag_pre_process = input("Realizar pre processamento? [S/N]\n").upper()
    preprocess_method(flag_pre_process,
                      num_antennas_tx=antennas_num_tx,
                      num_antennas_rx=antennas_num_rx,
                      antenna_config=antenna_config)

    print('--------------------------------------------')

    flag_beam_analysis = input("Realizar análise dos beams? [S/N] \n").upper()
    beam_analysis(flag_beam_analysis, antenna_config, connection)

    print('--------------------------------------------')

    temporal = input("Realizar a selecao de feixe? [S/N] \n").upper()
    if temporal == 'S':
        flag_rx_or_tx = input('Escolha sobre qual usuario deseja realizar a selecao do beam \n'
                              '\t 1. Rx \n'
                              '\t 2. Tx \n'
                              '\t 3. Combinados \n')

        if flag_rx_or_tx == '1':
            flag_rx_or_tx_or_C = 'R'
        if flag_rx_or_tx == '2':
            flag_rx_or_tx_or_C = 'T'
        if flag_rx_or_tx == '3':
            flag_rx_or_tx_or_C = 'C'

        flag_input_beam_selection = input("Com qual tipo de entreda deseja realizar a selecao de feixe com \n "
                                          "\t [1] Coordenadas? \n "
                                          "\t [2] LiDAR? \n "
                                          "\t [3] Coord + LiDAR? \n ")
    else:
        exit()

    # SELECIONA BEAMS COM COORD
    if flag_input_beam_selection == '1':
        a = input("selecionar o beam com as coordenadas pre-processadas em: \n"
                  "\t [1] matriz Qs [23 X 250] \n"
                  "\t [2] Matriz Qs em linhas [2 X 120] \n")
        if a == '1':
            type_of_input = 'coord_in_Qs'

        if a == '2':
            type_of_input = 'coord_in_Qs_lines'

    # SELECIONA BEAMS COM LIDAR
    if flag_input_beam_selection == '2':
        type_of_input = 'lidar'

    # SELECIONA BEAMS COM COORD + LIDAR
    if flag_input_beam_selection == '3':

        b = input("selecionar o Beam com as coordenadas pre-processadas em: \n" \
                  "\t [1] matriz Qs [23 X 250] \n" \
                  "\t [2] Matriz Qs em linhas [2 X 120] \n")
        if b == '1':
            type_of_input = 'coord_in_Qs_+_Lidar'

        if b == '2':
            type_of_input = 'coord_in_Qs_lines_+_Lidar'

    print('type_of_input: ', type_of_input, 'type_of_connection', connection, 'antenna_config', antenna_config,
          'flag_rx_or_tx', flag_rx_or_tx)
    beam_selection(type_of_input=type_of_input, type_of_connection=connection, antenna_config=antenna_config,
                   flag_rx_or_tx=flag_rx_or_tx_or_C)

############# ## MAIN ###############


    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    run_simulation()






####################-  SIMULACAO STAND ALONE PARAMETRIZADA -#################################################
'''
    flag_rx_or_tx_vector = ['T','R']
    type_of_input_vector = ['coord_in_Qs','coord_in_Qs_lines','lidar','coord_in_Qs_+_Lidar','coord_in_Qs_lines_+_Lidar']
    antenna_config_vector = ['2X2','2X32','4X4','4X32','8X8']
    connection_vector = ['ALL','LOS','NLOS']

    for i in range(len(antenna_config_vector)):
        for input in range(len(type_of_input_vector)):
            for con in range(len(connection_vector)):
                print('----------------------------------------------------')
                print('Beam selection do Tx:' +
                        '\n config da antena' + antenna_config_vector[i] +
                        '\n entrada: '+type_of_input_vector[input] +
                        '\n tipo de conexao' +connection_vector[con])
                beam_selection(type_of_input=type_of_input_vector[input],
                               type_of_connection=connection_vector[con],
                               antenna_config=antenna_config_vector[i],
                               flag_rx_or_tx=flag_rx_or_tx_vector[0])
                print('----------------------------------------------------')

                print('Beam selection do Rx:' +
                      '\n config da antena' + antenna_config_vector[i] +
                      '\n entrada: ' + type_of_input_vector[input] +
                      '\n tipo de conexao' + connection_vector[con])
                beam_selection(type_of_input=type_of_input_vector[input],
                               type_of_connection=connection_vector[con],
                               antenna_config=antenna_config_vector[i],
                               flag_rx_or_tx=flag_rx_or_tx_vector[1])
                print('----------------------------------------------------')
                print('----------------------------------------------------')
    '''

















