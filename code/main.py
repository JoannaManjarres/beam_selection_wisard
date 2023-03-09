
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
        coord = preprocess.read_all_coordinates()
        train, test, coordenadas_LOS_train, coordenadas_NLOS_train, coordenadas_LOS_test, coordenadas_NLOS_test = preprocess.coord_to_Qs_matrix(coord)
        preprocess.save_Qs_matrix(train, test, coordenadas_LOS_train, coordenadas_LOS_test, coordenadas_NLOS_train, coordenadas_NLOS_test)

    if flag_preprocess_LiDAR == 'S':
        obj_lidar.process_data(enable_plot=False, num_scene_to_plot=0)
        obj_lidar.separed_data_lidar_LOS_NLOS()

def beam_selection(flag_beam_selection):
    print("A seleção dos indices dos feixes sera realizada  ")
    if flag_beam_selection == '1':    #TODOS os dados do dataset
        print(" a partir das posicoes de TODOS os veiculos")
        all_coord_train, all_coord_test = read.read_all_Qs_matrix()
        index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test = read.read_all_beams()
        obj.select_best_beam(all_coord_train,
                             all_coord_test,
                             index_beam_rx_train,
                             index_beam_rx_test,
                             "TODOS os dados das coordenadas dos veiculos",
                             enableDebug = False,
                             plot_confusion_matrix_enable = False)

    if flag_beam_selection == '2':    # COORD [LOS]
        print("a partir das posicoes dos veiculos que apenas possuam conexão LOS")
        LOS_coord_train, LOS_coord_test = read.read_Qs_matrix_with_LOS_data()
        index_beam_rx_LOS_train, index_beam_rx_LOS_test, index_beam_tx_LOS_train, index_beam_tx_LOS_test = read.read_LOS_beams()
        obj.select_best_beam(LOS_coord_train,
                             LOS_coord_test,
                             index_beam_rx_LOS_train,
                             index_beam_rx_LOS_test,
                             "coordenadas dos veiculos com conexão LOS",
                             enableDebug = False,
                             plot_confusion_matrix_enable = False)

    if flag_beam_selection == '3':    # COORD [NLOS]
        print("a partir das posicoes dos veiculos que apenas possuam NLOS")
        NLOS_coord_train, NLOS_coord_test = read.read_Qs_matrix_with_NLOS_data()
        index_beam_rx_NLOS_train, index_beam_rx_NLOS_test, index_beam_tx_NLOS_train, index_beam_tx_NLOS_test = read.read_NLOS_beams()
        obj.select_best_beam(NLOS_coord_train,
                             NLOS_coord_test,
                             index_beam_rx_NLOS_train,
                             index_beam_rx_NLOS_test,
                             "coordenadas dos veiculos com conexão NLOS",
                             enableDebug = False,
                             plot_confusion_matrix_enable = False)

    if flag_beam_selection == '4': # LiDAR [ALL]
        print("a partir de dados LiDAR com TODOS os dados do dataset")
        data_lidar_train, data_lidar_validation = obj_lidar.read_all_LiDAR_data()
        index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test = read.read_all_beams()
        obj.select_best_beam(data_lidar_train,
                             data_lidar_validation,
                             index_beam_rx_train,
                             index_beam_rx_test,
                             "LIDAR com TODOS os dados",
                             enableDebug=False,
                             plot_confusion_matrix_enable=False)

    if flag_beam_selection =='5' : # LiDAR [LOS]
        print("a partir de dados LiDAR com veiculos com conexao LOS  do dataset")
        data_lidar_LOS_train, data_lidar_LOS_test = obj_lidar.read_LiDAR_LOS_data()
        index_beam_rx_LOS_train, index_beam_rx_LOS_test, index_beam_tx_LOS_train, index_beam_tx_LOS_test = read.read_LOS_beams()
        obj.select_best_beam(data_lidar_LOS_train,
                             data_lidar_LOS_test,
                             index_beam_rx_LOS_train,
                             index_beam_rx_LOS_test,
                             "LIDAR com dados LOS",
                             enableDebug=False,
                             plot_confusion_matrix_enable=False)


    if flag_beam_selection =='6' : # LiDAR [NLOS]
        print("a partir de dados LiDAR com veiculos com conexao NLOS  do dataset")
        data_lidar_NLOS_train, data_lidar_NLOS_test =obj_lidar.read_LiDAR_NLOS_data()
        index_beam_rx_NLOS_train, index_beam_rx_NLOS_test, index_beam_tx_NLOS_train, index_beam_tx_NLOS_test = read.read_NLOS_beams()
        obj.select_best_beam(data_lidar_NLOS_train,
                             data_lidar_NLOS_test,
                             index_beam_rx_NLOS_train,
                             index_beam_rx_NLOS_test,
                             "LIDAR com dados NLOS",
                             enableDebug=False,
                             plot_confusion_matrix_enable=False)

    if flag_beam_selection =='7':
        print("a partir de dados LiDAR + Coordenadas com TODOS os dados do dataset")
        index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test = read.read_all_beams()
        data_lidar_train, data_lidar_validation = obj_lidar.process_data(enable_plot=False, num_scene_to_plot=0)
        all_coord_train, all_coord_test         = read.read_all_Qs_matrix()

        number_of_samples_train     = data_lidar_train.shape[0]
        dimension_of_coordenadas    = all_coord_train.shape[1]
        dimension_of_lidar          = data_lidar_train.shape[1]
        dimension_of_input          = dimension_of_lidar + dimension_of_coordenadas

        number_of_samples_test = data_lidar_validation.shape[0]

        input_train = np.zeros([number_of_samples_train, dimension_of_input], dtype=np.int8)
        input_test = np.zeros([number_of_samples_test, dimension_of_input], dtype=np.int8)

        for i in range(len(data_lidar_train)):
            input_train[i] = np.concatenate((all_coord_train[i], data_lidar_train[i]))

        for i in range(len(data_lidar_validation)):
            input_test[i] = np.concatenate((all_coord_test[i], data_lidar_validation[i]))

        obj.select_best_beam(input_train,
                             input_test,
                             index_beam_rx_train,
                             index_beam_rx_test,
                             "LIDAR + Coord com TODOS os dados",
                             enableDebug=False,
                             plot_confusion_matrix_enable=False)

    if flag_beam_selection == '8':
        print("a partir de dados LiDAR + Coordenadas com dados LOS do dataset")

        LiDAR_LOS_train, LiDAR_LOS_test = obj_lidar.read_LiDAR_LOS_data()
        coord_LOS_train, coord_LOS_test = read.read_Qs_matrix_with_LOS_data()
        index_beam_rx_LOS_train, index_beam_rx_LOS_test, index_beam_tx_LOS_train, index_beam_tx_LOS_test = read.read_LOS_beams()

        coord_LiDAR_LOS_train = np.column_stack((coord_LOS_train, LiDAR_LOS_train))
        coord_LiDAR_LOS_test = np.column_stack((coord_LOS_test, LiDAR_LOS_test))

        obj.select_best_beam(coord_LiDAR_LOS_train,
                                 coord_LiDAR_LOS_test,
                                 index_beam_rx_LOS_train,
                                 index_beam_rx_LOS_test,
                                 "LIDAR_+_Coord_com_dados_LOS",
                                 enableDebug=False,
                                 plot_confusion_matrix_enable=False)

    if flag_beam_selection == '9':
        print("a partir de dados LiDAR + Coordenadas com dados NLOS do dataset")
        LiDAR_NLOS_train, LiDAR_NLOS_test = obj_lidar.read_LiDAR_NLOS_data()
        coord_NLOS_train, coord_NLOS_test = read.read_Qs_matrix_with_NLOS_data()
        index_beam_rx_NLOS_train, index_beam_rx_NLOS_test, index_beam_tx_NLOS_train, index_beam_tx_NLOS_test = read.read_NLOS_beams()

        coord_LiDAR_NLOS_train = np.column_stack((coord_NLOS_train, LiDAR_NLOS_train))
        coord_LiDAR_NLOS_test = np.column_stack((coord_NLOS_test, LiDAR_NLOS_test))

        obj.select_best_beam(coord_LiDAR_NLOS_train,
                             coord_LiDAR_NLOS_test,
                             index_beam_rx_NLOS_train,
                             index_beam_rx_NLOS_test,
                             "LiDAR_+_Coord_com_dados_NLOS",
                             enableDebug=False,
                             plot_confusion_matrix_enable=False)


    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')



    flag_preprocess_coord = input("Realizar pre processamento das coordenadas? [S/N] \n").upper()
    flag_preprocess_beams_index = input("Realizar pre processamento dos indices dos Beams? [S/N] \n").upper()
    flag_preprocess_LiDAR = input("Realizar pre processamento dos dados LiDAR? [S/N] \n").upper()
    do_preprocess(flag_preprocess_coord, flag_preprocess_beams_index, flag_preprocess_LiDAR)

    flag_beam_selection = input ("Realizar a selecao de feixe com \n " \
    "\t 1. [Coord] TODOS os dados do dataset? \n " \
    "\t 2. [Coord] Apenas os dados Com Linha de Visado [LOS]? \n " \
    "\t 3. [Coord] Apenas os dados Sem Linha de Visado [NLOS]?\n" \
    "\t 4. [LiDAR] TODOS os dados do dataset? \n "\
    "\t 5. [LiDAR] com dados LOS? \n "\
    "\t 6. [LiDAR] com dados NLOS? \n"\
    "\t 7. [Coord + LiDAR] com TODOS os dados do dataset?\n"
    "\t 8. [Coord + LiDAR] com dados LOS? \n"
    "\t 9. [Coord + LiDAR] com dados NLOS? \n")


    beam_selection(flag_beam_selection)













