
import preprocesamento as preprocess
import read_data as read
import beam_selection_wisard as obj
import pre_process_lidar as obj_lidar
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌃' to toggle the breakpoint.

def do_preprocess(flag_preprocess_coord, flag_preprocess_beams_index, flag_preprocess_LiDAR):

    if flag_preprocess_beams_index =='S':
        coord = preprocess.read_valid_coordinates()
        tx_index, rx_index = preprocess.read_beams_raymobtime()
        preprocess.divide_beams_in_train_test(rx_index, tx_index, coord, save_data=True)
        preprocess.divide_beams_and_coord_in_LOS_or_NLOS_connect(rx_index, tx_index, coord, save_data=True)


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

def beam_selection(flag_beam_selection):
    print("A seleção dos indices dos feixes sera realizada  ")
    if flag_beam_selection == '1':    #TODOS os dados do dataset
        print(" a partir das posicoes de TODOS os veiculos")
        all_coord_train, all_coord_test = read.read_all_Qs_matrix()
        index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test = read.read_all_beams()

        flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
                              "\t Transmissor [T] ou Receptor [R]\n").upper()

        if(flag_rx_or_tx=='R'):
            obj.select_best_beam(all_coord_train,
                             all_coord_test,
                             index_beam_rx_train,
                             index_beam_rx_test,
                             "TODOS os dados das coordenadas dos veiculos",
                             enableDebug = False,
                             plot_confusion_matrix_enable = False)
        else:
            obj.select_best_beam(all_coord_train,
                                 all_coord_test,
                                 index_beam_tx_train,
                                 index_beam_tx_test,
                                 "Selecao_do_BEAM_Tx_com_TODOS_os_dados_das_coord_dos_veiculos",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)

    if flag_beam_selection == '2':    # COORD [LOS]
        print("a partir das posicoes dos veiculos que apenas possuam conexão LOS")
        LOS_coord_train, LOS_coord_test = read.read_Qs_matrix_with_LOS_data()
        index_beam_rx_LOS_train, index_beam_rx_LOS_test, index_beam_tx_LOS_train, index_beam_tx_LOS_test = read.read_LOS_beams()

        flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
                              "\t Transmissor [T] ou Receptor [R]\n").upper()

        if (flag_rx_or_tx == 'R'):
            obj.select_best_beam(LOS_coord_train,
                             LOS_coord_test,
                             index_beam_rx_LOS_train,
                             index_beam_rx_LOS_test,
                             "Selecao_do_BEAM_Rx_com_coordenadas com conexão LOS",
                             enableDebug = False,
                             plot_confusion_matrix_enable = False)
        else:
            obj.select_best_beam(LOS_coord_train,
                                 LOS_coord_test,
                                 index_beam_tx_LOS_train,
                                 index_beam_tx_LOS_test,
                                 "Selecao_do_BEAM_Tx_com_Coordenadas_com_conexao_LOS",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)

    if flag_beam_selection == '3':    # COORD [NLOS]
        print("a partir das posicoes dos veiculos que apenas possuam NLOS")
        NLOS_coord_train, NLOS_coord_test = read.read_Qs_matrix_with_NLOS_data()
        index_beam_rx_NLOS_train, index_beam_rx_NLOS_test, index_beam_tx_NLOS_train, index_beam_tx_NLOS_test = read.read_NLOS_beams()

        flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
                              "\t Transmissor [T] ou Receptor [R]\n").upper()

        if (flag_rx_or_tx == 'R'):

            obj.select_best_beam(NLOS_coord_train,
                                NLOS_coord_test,
                                index_beam_rx_NLOS_train,
                                index_beam_rx_NLOS_test,
                                "Selecao_do_BEAM_Rx_com_coordenadas com conexão NLOS",
                                enableDebug = False,
                                plot_confusion_matrix_enable = False)
        else:
            obj.select_best_beam(NLOS_coord_train,
                                 NLOS_coord_test,
                                 index_beam_tx_NLOS_train,
                                 index_beam_tx_NLOS_test,
                                 "Selecao_do_BEAM_Tx_com_Coordenadas_com_conexão_NLOS")

    if flag_beam_selection == '4': # LiDAR [ALL]
        print("a partir de dados LiDAR com TODOS os dados do dataset")
        data_lidar_train, data_lidar_validation = obj_lidar.read_all_LiDAR_data()
        index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test = read.read_all_beams()

        flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
                              "\t Transmissor [T] ou Receptor [R]\n").upper()

        if (flag_rx_or_tx == 'R'):

            obj.select_best_beam(data_lidar_train,
                                data_lidar_validation,
                                index_beam_rx_train,
                                index_beam_rx_test,
                                "Selecao_do_BEAM_Rx_com_TODOS_os_dados_LIDAR ",
                                enableDebug=False,
                                plot_confusion_matrix_enable=False)

        else:
            obj.select_best_beam(data_lidar_train,
                                data_lidar_validation,
                                 index_beam_tx_train,
                                 index_beam_tx_test,
                                 "Selecao_do_BEAM_Tx_com_TODOS_os_dados_LIDAR",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)

    if flag_beam_selection =='5' : # LiDAR [LOS]
        print("a partir de dados LiDAR com veiculos com conexao LOS  do dataset")
        data_lidar_LOS_train, data_lidar_LOS_test = obj_lidar.read_LiDAR_LOS_data()
        index_beam_rx_LOS_train, index_beam_rx_LOS_test, index_beam_tx_LOS_train, index_beam_tx_LOS_test = read.read_LOS_beams()

        flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
                              "\t Transmissor [T] ou Receptor [R]\n").upper()

        if (flag_rx_or_tx == 'R'):

            obj.select_best_beam(data_lidar_LOS_train,
                             data_lidar_LOS_test,
                             index_beam_rx_LOS_train,
                             index_beam_rx_LOS_test,
                             "Selecao_do_BEAM_Rx_LIDAR_com_dados_LOS",
                             enableDebug=False,
                             plot_confusion_matrix_enable=False)

        else:
            obj.select_best_beam(data_lidar_LOS_train,
                                 data_lidar_LOS_test,
                                 index_beam_tx_LOS_train,
                                 index_beam_tx_LOS_test,
                                 "Selecao_do_BEAM_Tx_LIDAR_com_dados_LOS",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)


    if flag_beam_selection =='6' : # LiDAR [NLOS]
        print("a partir de dados LiDAR com veiculos com conexao NLOS  do dataset")
        data_lidar_NLOS_train, data_lidar_NLOS_test =obj_lidar.read_LiDAR_NLOS_data()
        index_beam_rx_NLOS_train, index_beam_rx_NLOS_test, index_beam_tx_NLOS_train, index_beam_tx_NLOS_test = read.read_NLOS_beams()

        flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
                              "\t Transmissor [T] ou Receptor [R]\n").upper()

        if (flag_rx_or_tx == 'R'):

            obj.select_best_beam(data_lidar_NLOS_train,
                             data_lidar_NLOS_test,
                             index_beam_rx_NLOS_train,
                             index_beam_rx_NLOS_test,
                             "Selecao_do_BEAM_Rx_LIDAR_com_dados_NLOS",
                             enableDebug=False,
                             plot_confusion_matrix_enable=False)

        else:
            obj.select_best_beam(data_lidar_NLOS_train,
                                 data_lidar_NLOS_test,
                                 index_beam_tx_NLOS_train,
                                 index_beam_tx_NLOS_test,
                                 "Selecao_do_BEAM_Tx_LIDAR_com_dados_NLOS",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)

    if flag_beam_selection =='7':
        print("a partir de dados LiDAR + Coordenadas com TODOS os dados do dataset")
        index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test = read.read_all_beams()
        data_lidar_train, data_lidar_validation = obj_lidar.process_data(enable_plot=False, num_scene_to_plot=0)
        all_coord_train, all_coord_test         = read.read_all_Qs_matrix()

        coord_LiDAR_ALL_train = np.column_stack((all_coord_train, data_lidar_train))
        coord_LiDAR_ALL_test = np.column_stack((all_coord_test, data_lidar_validation))

        flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
                              "\t Transmissor [T] ou Receptor [R]\n").upper()

        if (flag_rx_or_tx == 'R'):

            obj.select_best_beam(coord_LiDAR_ALL_train,
                                 coord_LiDAR_ALL_test,
                                 index_beam_rx_train,
                                 index_beam_rx_test,
                                 "Selecao_do_BEAM_Rx_LIDAR_+_Coord_com_TODOS_os_dados",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)
        else:
            obj.select_best_beam(coord_LiDAR_ALL_train,
                                 coord_LiDAR_ALL_test,
                                 index_beam_tx_train,
                                 index_beam_tx_test,
                                 "Selecao_do_BEAM_Tx_LIDAR_+_Coord_com_TODOS_os_dados",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False
                                 )

    if flag_beam_selection == '8':
        print("a partir de dados LiDAR + Coordenadas com dados LOS do dataset")

        LiDAR_LOS_train, LiDAR_LOS_test = obj_lidar.read_LiDAR_LOS_data()
        coord_LOS_train, coord_LOS_test = read.read_Qs_matrix_with_LOS_data()
        index_beam_rx_LOS_train, index_beam_rx_LOS_test, index_beam_tx_LOS_train, index_beam_tx_LOS_test = read.read_LOS_beams()

        coord_LiDAR_LOS_train = np.column_stack((coord_LOS_train, LiDAR_LOS_train))
        coord_LiDAR_LOS_test = np.column_stack((coord_LOS_test, LiDAR_LOS_test))

        flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
                              "\t Transmissor [T] ou Receptor [R]\n").upper()

        if (flag_rx_or_tx == 'R'):

            obj.select_best_beam(coord_LiDAR_LOS_train,
                                 coord_LiDAR_LOS_test,
                                 index_beam_rx_LOS_train,
                                 index_beam_rx_LOS_test,
                                 "Selecao_do_BEAM_Rx_LIDAR_+_Coord_com_dados_LOS",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)
        else:
            obj.select_best_beam(coord_LiDAR_LOS_train,
                                 coord_LiDAR_LOS_test,
                                 index_beam_tx_LOS_train,
                                 index_beam_tx_LOS_test,
                                 "Selecao_do_BEAM_Tx_LIDAR_+_Coord_com_dados_LOS",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)


    if flag_beam_selection == '9':
        print("a partir de dados LiDAR + Coordenadas com dados NLOS do dataset")
        LiDAR_NLOS_train, LiDAR_NLOS_test = obj_lidar.read_LiDAR_NLOS_data()
        coord_NLOS_train, coord_NLOS_test = read.read_Qs_matrix_with_NLOS_data()
        index_beam_rx_NLOS_train, index_beam_rx_NLOS_test, index_beam_tx_NLOS_train, index_beam_tx_NLOS_test = read.read_NLOS_beams()

        coord_LiDAR_NLOS_train = np.column_stack((coord_NLOS_train, LiDAR_NLOS_train))
        coord_LiDAR_NLOS_test = np.column_stack((coord_NLOS_test, LiDAR_NLOS_test))

        flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
                              "\t Transmissor [T] ou Receptor [R]\n").upper()

        if (flag_rx_or_tx == 'R'):

            obj.select_best_beam(coord_LiDAR_NLOS_train,
                             coord_LiDAR_NLOS_test,
                             index_beam_rx_NLOS_train,
                             index_beam_rx_NLOS_test,
                             "Selecao_do_BEAM_Rx_LiDAR_+_Coord_com_dados_NLOS",
                             enableDebug=False,
                             plot_confusion_matrix_enable=False)

        else:
            obj.select_best_beam(coord_LiDAR_NLOS_train,
                                 coord_LiDAR_NLOS_test,
                                 index_beam_tx_NLOS_train,
                                 index_beam_tx_NLOS_test,
                                 "Selecao_do_BEAM_Tx_LiDAR_+_Coord_com_dados_NLOS",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)

    if flag_beam_selection == '10':
        all_coord_in_Qs_line_train, all_coord_in_Qs_line_test = read.read_all_Qs_matrix_in_lines()
        index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test = read.read_all_beams()

        flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
                              "\t Transmissor [T] ou Receptor [R]\n").upper()

        if (flag_rx_or_tx == 'R'):
            obj.select_best_beam(all_coord_in_Qs_line_train,
                                 all_coord_in_Qs_line_test,
                                 index_beam_rx_train,
                                 index_beam_rx_test,
                                 "Selecao_do_BEAM_Rx_with_Coord_em_Qs_Line",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)
        else:
            obj.select_best_beam(all_coord_in_Qs_line_train,
                                 all_coord_in_Qs_line_test,
                                 index_beam_tx_train,
                                 index_beam_tx_test,
                                 "Selecao_do_BEAM_Tx_with_Coord_em_Qs_Line",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)

    if flag_beam_selection == '11':
        LOS_coord_in_Qs_line_train, LOS_coord_in_Qs_line_test = read.read_Qs_matrix_in_lines_with_LOS_data()
        index_beam_rx_LOS_train, index_beam_rx_LOS_test, index_beam_tx_LOS_train, index_beam_tx_LOS_test = read.read_LOS_beams()

        flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
                              "\t Transmissor [T] ou Receptor [R]\n").upper()

        if (flag_rx_or_tx == 'R'):
            obj.select_best_beam(LOS_coord_in_Qs_line_train,
                                 LOS_coord_in_Qs_line_test,
                                 index_beam_rx_LOS_train,
                                 index_beam_rx_LOS_test,
                                 "Selecao_do_BEAM_Rx_with_LOS_Coord_em_Qs_Line",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)
        else:
            obj.select_best_beam(LOS_coord_in_Qs_line_train,
                                 LOS_coord_in_Qs_line_test,
                                 index_beam_tx_LOS_train,
                                 index_beam_tx_LOS_test,
                                 "Selecao_do_BEAM_Tx_with_LOS_Coord_em_Qs_Line",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)

    if flag_beam_selection == '12':
        NLOS_coord_in_Qs_line_train, NLOS_coord_in_Qs_line_test = read.read_Qs_matrix_in_lines_with_NLOS_data()
        index_beam_rx_NLOS_train, index_beam_rx_NLOS_test, index_beam_tx_NLOS_train, index_beam_tx_NLOS_test = read.read_NLOS_beams()


        flag_rx_or_tx = input("Realizar a selecao de feixe do: \n " \
                              "\t Transmissor [T] ou Receptor [R]\n").upper()

        if (flag_rx_or_tx == 'R'):
            obj.select_best_beam(NLOS_coord_in_Qs_line_train,
                                 NLOS_coord_in_Qs_line_test,
                                 index_beam_rx_NLOS_train,
                                 index_beam_rx_NLOS_test,
                                 "Selecao_do_BEAM_Rx_with_NLOS_Coord_em_Qs_Line",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)
        else:
            obj.select_best_beam(NLOS_coord_in_Qs_line_train,
                                 NLOS_coord_in_Qs_line_test,
                                 index_beam_tx_NLOS_train,
                                 index_beam_tx_NLOS_test,
                                 "Selecao_do_BEAM_Tx_with_NLOS_Coord_em_Qs_Line",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)



    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    preprocess.pre_process_coordinates_in_lines(separed_coord_LOS=True)


    flag_preprocess_coord = input("Realizar pre processamento das coordenadas? [S/N] \n").upper()
    flag_preprocess_beams_index = input("Realizar pre processamento dos indices dos Beams? [S/N] \n").upper()
    flag_preprocess_LiDAR = input("Realizar pre processamento dos dados LiDAR? [S/N] \n").upper()
    do_preprocess(flag_preprocess_coord, flag_preprocess_beams_index, flag_preprocess_LiDAR)

    flag_input_beam_selection = input("Realizar a selecao de feixe com \n " \
        "\t [1] Coordenadas? \n " \
        "\t [2] LiDAR? \n " 
        "\t [3] Coord + LiDAR? \n " )

    if flag_input_beam_selection == '1':
        a =  input("selecionar o beam com as coordenadas pre-processadas em: \n"\
                   "\t [1] matriz Qs [23 X 250] \n"\
                   "\t [2] Matriz Qs em linhas [2 X 120] \n")
        if a=='1':
            flag_beam_selection = input("Realizar a selecao de feixe com  Qs [23X250] \n " \
                                        "\t 1. [Coord] TODOS os dados do dataset? \n " \
                                        "\t 2. [Coord] Apenas os dados Com Linha de Visado [LOS]? \n " \
                                        "\t 3. [Coord] Apenas os dados Sem Linha de Visado [NLOS]?\n" )
        if a=='2':
            flag_beam_selection = input("Realizar a selecao de feixe com Qs [2X120] \n " \
                                        "\t 10. [Coord] TODOS os dados do dataset? \n " \
                                        "\t 11. [Coord] Apenas os dados Com Linha de Visado [LOS]? \n " \
                                        "\t 12. [Coord] Apenas os dados Sem Linha de Visado [NLOS]?\n")

    if flag_input_beam_selection == '2':
        flag_beam_selection = input("Realizar a selecao de feixe com \n " \
                                    "\t 4. [LiDAR] TODOS os dados do dataset? \n " \
                                    "\t 5. [LiDAR] com dados LOS? \n " \
                                    "\t 6. [LiDAR] com dados NLOS? \n" )

    if flag_input_beam_selection == '3':
        flag_beam_selection = input("Realizar a selecao de feixe com \n " \
                                     "\t 7. [Coord + LiDAR] com TODOS os dados do dataset?\n"
                                     "\t 8. [Coord + LiDAR] com dados LOS? \n"
                                     "\t 9. [Coord + LiDAR] com dados NLOS? \n")


    beam_selection(flag_beam_selection)













