
import preprocesamento as preprocess
import read_data as read
import beam_selection_wisard as obj
import pre_process_lidar as obj_lidar
import beam_selection_wisard_top_k as obj_selection_top_k
import analyse_data as obj_analyse
import numpy as np
import connection_detection_LOS_NLOS as obj_connect_detection
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
        print("Pre processar os dados LiDAR em:")
        flag_type_of_pre_process_lidar=input("\t [1] Quantizados [Dataset] + Posicao Rx \n " \
                                             "\t [2] Separar os dados LiDAR quantizados em LOS e NLOS \n"
                                             "\t [3] Quantizados[Dataset] sem Rx \n"
                                             "\t [4] Receptor com um cubo + Quantizados[Dataset] \n"
                                             "\t [5] Todos os dados como um cubo incluindo o Rx\n"
                                             )

        if flag_type_of_pre_process_lidar == '1':
            obj_lidar.process_data(enable_plot=False, num_scene_to_plot=0)
        if flag_type_of_pre_process_lidar == '2':
            obj_lidar.separed_data_lidar_LOS_NLOS()
        if flag_type_of_pre_process_lidar == '3':
            obj_lidar.process_data_without_rx(enable_plot=False, num_scene_to_plot=0)
        if flag_type_of_pre_process_lidar == '4':
            obj_lidar.process_data_rx_like_cube()
        if flag_type_of_pre_process_lidar == '5':
            obj_lidar.process_all_data_like_cube()


def beam_selection(antenna_config, type_of_connection, type_of_input, flag_rx_or_tx, type_of_selection):

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
        if type_of_input == 'lidar_LOS_NLOS':
            data_lidar_LOS_train, data_lidar_LOS_validation = obj_lidar.read_LiDAR_LOS_data()
            data_lidar_NLOS_train, data_lidar_NLOS_validation = obj_lidar.read_LiDAR_NLOS_data()

        if type_of_input == 'Lidar_sem_rx':
            data_lidar_sem_rx_train, data_lidar_sem_rx_validation = obj_lidar.read_all_LiDAR_without_rx()
            input_train = data_lidar_sem_rx_train
            input_test = data_lidar_sem_rx_validation
        if type_of_input == 'rx_cubo_+_Lidar':
            print(type_of_input)
            data_lidar_rx_like_cube_train, data_lidar_rx_like_cube_validation = obj_lidar.read_LiDAR_with_rx_like_cube()
            input_train = data_lidar_rx_like_cube_train
            input_test = data_lidar_rx_like_cube_validation

        if type_of_input == 'all_data_like_cube':
            print('Input: ',type_of_input)
            data_lidar_like_cube_train, data_lidar_like_cube_validation = obj_lidar.read_LiDAR_all_data_like_cube()
            input_train = data_lidar_like_cube_train
            input_test = data_lidar_like_cube_validation

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

        if type_of_input == 'coord_in_termometro':
            encondign_coord_train, encondign_coord_test = preprocess.Thermomether_parte_inteira_mais_parte_decimal_2()
            input_train = encondign_coord_train
            input_test = encondign_coord_test

        if type_of_input == 'coord_in_termometro_iguais':
            encondign_coord_train, encondign_coord_test = preprocess.Thermomether_match_coor_x_and_y_include_decimal()
            input_train = encondign_coord_train
            input_test = encondign_coord_test

        if type_of_input == 'coord_in_termometro_+_Lidar':
            encondign_coord_train, encondign_coord_test = preprocess.Thermomether_dobro_resolucao()
            data_lidar_train, data_lidar_validation = obj_lidar.read_all_LiDAR_data()
            input_train = np.column_stack((encondign_coord_train, data_lidar_train))
            input_test = np.column_stack((encondign_coord_test, data_lidar_validation))

        if type_of_input == 'coord_in_termometro_+_Lidar_sem_rx':
            encondign_coord_train, encondign_coord_test = preprocess.Thermomether_dobro_resolucao()
            data_lidar_sem_rx_train, data_lidar_sem_rx_validation = obj_lidar.read_all_LiDAR_without_rx()
            input_train = np.column_stack((encondign_coord_train, data_lidar_sem_rx_train))
            input_test = np.column_stack((encondign_coord_test, data_lidar_sem_rx_validation))

        if type_of_input == 'coord_in_termometro_int_decimal_+_rx_cubo_+_Lidar':
            print(type_of_input)
            data_lidar_rx_like_cube_train, data_lidar_rx_like_cube_validation = obj_lidar.read_LiDAR_with_rx_like_cube()
            encondign_coord_train, encondign_coord_test = preprocess.Thermomether_match_coor_x_and_y_include_decimal()
            input_train = np.column_stack((encondign_coord_train, data_lidar_rx_like_cube_train))
            input_test = np.column_stack((encondign_coord_test, data_lidar_rx_like_cube_validation))



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

    if type_of_selection == '1':

        print("input: ", np.shape(input_train))
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

    elif type_of_selection =='2':
        obj_selection_top_k.beam_selection_top_k_wisard(x_train=input_train,
                                                        x_test=input_test,
                                                        y_train=label_train,
                                                        y_test=label_test,
                                                        data_input=type_of_input)


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
        top_k = input("1 - Normal \n"
                      "2 - Top-k \n").upper()

        if top_k == '1':

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
            else:
                exit()

        elif top_k == '2':
            flag_rx_or_tx = '3'
            flag_rx_or_tx_or_C = 'C'
        else:
            exit()
    else:
        exit()

    flag_input_beam_selection = input("Com qual tipo de entrada deseja realizar a selecao de feixe com \n "
                                              "\t [1] Coordenadas? \n "
                                              "\t [2] LiDAR? \n "
                                              "\t [3] Coord + LiDAR? \n ")



    # SELECIONA BEAMS COM COORD
    if flag_input_beam_selection == '1':
        a = input("selecionar o beam com as coordenadas pre-processadas em: \n"
                  "\t [1] matriz Qs [23 X 250] \n"
                  "\t [2] Matriz Qs em linhas [2 X 120] \n"
                  "\t [3] Termometro [256 bits] \n"
                  "\t [4] Termometro coord x e y 'iguais' [960X980] \n")

        if a == '1':
            type_of_input = 'coord_in_Qs'

        if a == '2':
            type_of_input = 'coord_in_Qs_lines'

        if a =='3':
            type_of_input = 'coord_in_termometro'

        if a=='4':
            type_of_input = 'coord_in_termometro_iguais'

    # SELECIONA BEAMS COM LIDAR
    if flag_input_beam_selection == '2':
        c = input("selecionar o Beam com dados LiDAR pre-processadas em: \n" \
                  "\t [1] Quantizados [Dataset] + Posicao do Rx ? \n" \
                  "\t [2] Quantizados [Dataset] SEM a Posicao do Rx ? \n" \
                  "\t [3] Quantizados [Dataset] + Rx como um cubo? \n"
                  "\t [4] Todos os dados como um cubo + Rx como um cubo? \n")

        if c =='1':
            type_of_input = 'lidar'
        if c =='2':
            type_of_input = 'Lidar_sem_rx'
        if c =='3':
            type_of_input = 'rx_cubo_+_Lidar'
        if c =='4':
            type_of_input = 'all_data_like_cube'

    # SELECIONA BEAMS COM COORD + LIDAR
    if flag_input_beam_selection == '3':

        b = input("selecionar o Beam com as coordenadas pre-processadas em: \n" \
                  "\t [1] matriz Qs [23 X 250] \n" \
                  "\t [2] Matriz Qs em linhas [2 X 120] \n"\
                  "\t [3] Termometro \n"
                  "\t [4] Termometro e LiDAR sem rx \n"
                  "\t [5] Termometro (int, decimal) + Rx como um cubo + LiDAR\n")


        if b == '1':
            type_of_input = 'coord_in_Qs_+_Lidar'

        if b == '2':
            type_of_input = 'coord_in_Qs_lines_+_Lidar'

        if b == '3':
            type_of_input = 'coord_in_termometro_+_Lidar'

        if b =='4':
            type_of_input = 'coord_in_termometro_+_Lidar_sem_rx'
        if b=='5':
            type_of_input = 'coord_in_termometro_int_decimal_+_rx_cubo_+_Lidar'

    print('type_of_input: ', type_of_input, 'type_of_connection', connection, 'antenna_config', antenna_config,
          'flag_rx_or_tx', flag_rx_or_tx)

    beam_selection(type_of_input=type_of_input, type_of_connection=connection, antenna_config=antenna_config,
                   flag_rx_or_tx=flag_rx_or_tx_or_C, type_of_selection=top_k)



############# ## MAIN ###############


    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')




    #obj_connect_detection.do_LOS_NLOS_detection()
    run_simulation()
    #encondign_coord_train, encondign_coord_test= preprocess.Thermomether_parte_inteira_mais_parte_decimal_2()
    #a=0
    #label_train, label_test, label_train_in_str, label_test_in_str, coord_train, coord_test = obj_connect_detection.analyses_data()
    '''
    wisard_accuracy = obj_connect_detection.LOS_NLOS_WISARD_classification(input_train=encondign_coord_train,
                                                     input_validation=input_train,
                                                     input_validation=encondign_coord_test,
                                                     label_train=label_train_in_str,
                                                     label_validation=label_test_in_str,
                                                     figure_name='LOS_NLOS_detection_encondign_coord',
                                                     antenna_config='8X32',
                                                     type_of_input='lidar',
                                                     titulo_figura='Deteção LOS/NLOS Com coord codificadas via Termometro',
                                                     user='All',
                                                     enableDebug=False,
                                                     )

'''



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

















