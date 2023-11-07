import wisardpkg as wp
from operator import itemgetter
import pandas as pd
from sklearn.metrics import accuracy_score

import preprocesamento as obj_coord
import pre_process_lidar as obj_lidar
import read_data as read_labels
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import statistics

def read_coordinates():
    escala = 8
    all_coord_in_Thermomether_x_y_unbalanced_train, all_coord_in_Thermomether_x_y_unbalanced_test = obj_coord.read_coord_in_Thermomether_x_y_unbalanced(escala)
    input_train = all_coord_in_Thermomether_x_y_unbalanced_train
    input_test = all_coord_in_Thermomether_x_y_unbalanced_test

    return input_train, input_test

def read_labels():
    antenna_config = '8X32'
    path = '../data/beams/' + antenna_config + '/all_index_beam/'
    input_cache_file = np.load(path + "index_beams_combined_train.npz", allow_pickle=True)
    index_beam_combined_train = input_cache_file["all_beam_combined_train"].astype(str)

    input_cache_file = np.load(path + "index_beams_combined_test.npz", allow_pickle=True)
    index_beam_combined_test = input_cache_file["all_beam_combined_test"].astype(str)

    return index_beam_combined_train, index_beam_combined_test

def read_Lidar():
    all_lidar_train, all_lidar_test = obj_lidar.read_LiDAR_with_rx_like_cube()

    return all_lidar_train, all_lidar_test

def beam_selection_commite(x_train, x_test, y_train, y_test, addressSize):
    data_set = 'all'

    # addressSize = 44
    #addressSize = 64
    ignoreZero = False
    verbose = False
    var = True
    wsd = wp.Wisard(addressSize,
                    ignoreZero=ignoreZero,
                    verbose=verbose,
                    returnConfidence=var,
                    returnActivationDegree=var,
                    returnClassesDegrees=var)
    wsd.train(x_train, y_train)

    # the output is a list of string, this represent the classes attributed to each input
    out = wsd.classify(x_test)


    all_out_wisard_order = np.empty(len(out), dtype=object)
    a = []

    for amostra_a_avaliar in range(len(out)):
        lista_das_classes = out[amostra_a_avaliar]['classesDegrees']
        dict_com_classes_na_ordem = sorted(lista_das_classes, key=itemgetter('degree'), reverse=True)
        all_out_wisard_order[amostra_a_avaliar] = dict_com_classes_na_ordem
        a.append(dict_com_classes_na_ordem)

    return a


def committe_majority_vote():
    index_beam_combined_train, index_beam_combined_test = read_labels()

    coord_train, coord_test = read_coordinates()
    print('selecionando feixes a partir das coordenadas')
    print('   coord train: ', coord_train.shape)
    beam_selection_for_coord = beam_selection_commite(x_train=coord_train,
                                                      x_test=coord_test,
                                                      y_train=index_beam_combined_train,
                                                      y_test=index_beam_combined_test,
                                                      addressSize=44)

    lidar_train, lidar_test = read_Lidar()
    print('selecionando feixes a partir do Lidar')
    print('   Lidar train: ', lidar_train.shape)
    beam_selection_for_lidar = beam_selection_commite(x_train=lidar_train,
                                                      x_test=lidar_test,
                                                      y_train=index_beam_combined_train,
                                                      y_test=index_beam_combined_test,
                                                      addressSize=64)

    acuraccy = committe_difference_between_scores(beam_selection_for_coord, beam_selection_for_lidar, label_validation=index_beam_combined_test)
    print('acuracia: ', acuraccy)
def committe_using_wisard():

    coord_train, coord_test = read_coordinates()
    x_train =coord_train
    x_test = coord_test

    index_beam_combined_train, index_beam_combined_test = read_labels()
    y_train = index_beam_combined_train
    y_test = index_beam_combined_test

    print('selecionando feixes a partir das coordenadas')
    print('   coord train: ', coord_train.shape)
    beam_selection_for_coord = beam_selection_commite(x_train=coord_train,
                                                      x_test=coord_test,
                                                      y_train=index_beam_combined_train,
                                                      y_test=index_beam_combined_test,
                                                      addressSize=44)

    b = []
    #b=np.zeros([1960,2300], dtype=int)
    all_beams_in_thermometer_vector = []
    for sample in range(len(beam_selection_for_coord)):                             #len(beam_selection_for_coord) = 1960
        all_beam_for_sample = beam_selection_for_coord[sample]                      #len(beam_selection_for_coord[sample] ) = 143
        list_of_class = np.array([int(d['class']) for d in all_beam_for_sample if 'class' in d])
        list_of_degree = np.array([(d['degree']) for d in all_beam_for_sample if 'degree' in d])
        data_for_thermometer = np.rint(np.multiply(list_of_class, list_of_degree)).astype(int)  #len(data_for_thermometer) = 143

        size_of_thermometer = np.array([len(data_for_thermometer), max(data_for_thermometer)], dtype=int)
        all_beams_in_thermomether = np.zeros(size_of_thermometer, dtype=int)

        sample_in_sample = 0
        for i in data_for_thermometer:
            for j in range(i):
                all_beams_in_thermomether[sample_in_sample, j] = 1
            sample_in_sample = sample_in_sample + 1

        a = all_beams_in_thermomether.reshape(1, (len(data_for_thermometer) * max(data_for_thermometer)))

        all_beams_in_thermometer_vector.append(a)

    for i in range(len(all_beams_in_thermometer_vector)):
        #b[i] = all_beams_in_thermometer_vector[i][0]
        b.append(all_beams_in_thermometer_vector[i][0])


        #coord_val_Qs.append([episodio_val, flag_LOS_or_NLOS_val, Qs_val.reshape(1, num_positions_of_internal_matrix)])

        #for i in range(len(data_for_thermometer)):
            #all_beams_in_thermomether[i][0:data_for_thermometer[i]] = 1
         #   all_beams_in_thermomether[i, 0:data_for_thermometer[i]] = 1


        #all_beams_in_thermometer_vector.append(all_beams_in_thermomether.reshape(1, (len(data_for_thermometer)*max(data_for_thermometer))))

    #test = all_beams_in_thermometer_vector.tolist()
    all_beams_in_thermometer_vector_train = b[0:1000]
    all_beams_in_thermometer_vector_test = b[1000:1960]

    new_labels_train = y_test[0:1000]
    new_labels_test = y_test[1000:1960]

    x_train_1 = all_beams_in_thermometer_vector_train
    x_test_1 =all_beams_in_thermometer_vector_test

    y_train_1 = new_labels_train
    y_test_1 = new_labels_test

    ignoreZero = False
    addressSize = 64
    verbose = True
    var = True
    wsd = wp.Wisard(addressSize,
                    ignoreZero=ignoreZero,
                    verbose=verbose,
                    returnConfidence=var,
                    returnActivationDegree=var,
                    returnClassesDegrees=var)
    wsd.train(x_train_1, y_train_1)

    # the output is a list of string, this represent the classes attributed to each input
    out = wsd.classify(x_test_1)

    acuracia = accuracy_score(y_test_1, out)
    print(acuracia)


    #all_beams_in_thermomether.reshape(1, 2002)


    lidar_train, lidar_test = read_Lidar()
    print('selecionando feixes a partir do Lidar')
    print('   Lidar train: ', lidar_train.shape)
    beam_selection_for_lidar = beam_selection_commite(x_train=lidar_train,
                                                      x_test=lidar_test,
                                                      y_train=index_beam_combined_train,
                                                      y_test=index_beam_combined_test,
                                                      addressSize=64)


    a=0


def committe():
    index_beam_combined_train, index_beam_combined_test = read_labels()

    coord_train, coord_test = read_coordinates()
    print('selecionando feixes a partir das coordenadas')
    print('   coord train: ', coord_train.shape)
    beam_selection_for_coord = beam_selection_commite(x_train=coord_train,
                                                      x_test=coord_test,
                                                      y_train=index_beam_combined_train,
                                                      y_test=index_beam_combined_test,
                                                      addressSize=44)

    lidar_train, lidar_test = read_Lidar()
    print('selecionando feixes a partir do Lidar')
    print('   Lidar train: ', lidar_train.shape)
    beam_selection_for_lidar = beam_selection_commite(x_train=lidar_train,
                                                      x_test=lidar_test,
                                                      y_train=index_beam_combined_train,
                                                      y_test=index_beam_combined_test,
                                                      addressSize=64)

    #acuracia_score_mean, name_committe_score = committe_score_mean(beam_selection_for_coord, beam_selection_for_lidar, label_validation=index_beam_combined_test)
    #acuracia_position_mean, name_committe_pos = committe_position_mean(beam_selection_for_coord, beam_selection_for_lidar, label_validation=index_beam_combined_test)
    #acuracia_score_geometric_mean, name_committe_score_geometric_mean = committe_score_geometric_mean(beam_selection_for_coord, beam_selection_for_lidar, label_validation=index_beam_combined_test)
    #acuracy_difference_between_scores, name_difference_between_scores = committe_difference_between_scores(beam_selection_for_coord, beam_selection_for_lidar, label_validation=index_beam_combined_test)
    #accuracy_add_score, name_add_score = committe_add_of_score(beam_selection_for_coord, beam_selection_for_lidar, label_validation=index_beam_combined_test)
    accuracy_dot_score, name_dot_score = committe_dot_of_score(beam_selection_for_coord, beam_selection_for_lidar, label_validation=index_beam_combined_test)



    #plot_results_each_committe(acuracy_difference_between_scores, name_difference_between_scores)

    #plot_results(x_data_1=acuracia_score_mean,
    #             name_committe_1=name_committe_score,
    #             x_data_2=acuracia_position_mean,
    #             name_committe_2=name_committe_pos)
def committe_score_mean(beam_selection_for_coord, beam_selection_for_lidar, label_validation):

    # number_of_samples = beam_selection_for_coord.shape[0]
    number_of_samples = len(beam_selection_for_coord)


    beam_selection = []

    for i in range(number_of_samples):
        beam_coord = beam_selection_for_coord[i]
        beam_lidar = beam_selection_for_lidar[i]

        #max_degree_coord = max(beam_coord, key=itemgetter('degree'))
        max_degree_coord = max([d['degree'] for d in beam_coord if 'degree' in d])
        max_degree_lidar = max([d['degree'] for d in beam_lidar if 'degree' in d])

        coord_degree_norm = [d['degree']/max_degree_coord for d in beam_coord if 'degree' in d]
        lidar_degree_norm = [d['degree']/max_degree_lidar for d in beam_lidar if 'degree' in d]

        # Adiciona a ponderacao no dicionario
        for j in range(len(beam_coord)):
            beam_coord[j]['norm'] = coord_degree_norm[j]
            beam_lidar[j]['norm'] = lidar_degree_norm[j]


        # extrai todas as classes que estao no dicionario
        #list_of_class_of_beam_coord = [d['class'] for d in beam_coord if 'class' in d]
        list_of_class_of_beam_coord = [int(d['class']) for d in beam_coord if 'class' in d]
        list_of_class_of_beam_lidar = [int(d['class']) for d in beam_lidar if 'class' in d]


        list_of_classes = []
        list_of_classes_degree = []
        list_of_classes_degree_1=[]
        #for i in range(len(beam_lidar)):
        for i in range(len(list_of_class_of_beam_lidar)):
            for index, value in enumerate(list_of_class_of_beam_coord):
                #if beam_lidar[i]['class'] == value:
                if int(beam_lidar[i]['class']) == value:
                    mean_of_degree_norm_per_class = (beam_coord[index]['norm'] + beam_lidar[i]['norm'])/2
                    #classe = beam_lidar[i]['class']
                    classe = int(beam_lidar[i]['class'])
                    list_of_classes_degree.append([classe, mean_of_degree_norm_per_class])
                    list_of_classes_degree_1.append((classe, mean_of_degree_norm_per_class))

        order_list_of_classes = sorted(list_of_classes_degree, key=itemgetter(1), reverse=True)
        #order_list_of_classes = sorted(list_of_classes, key=itemgetter(1), reverse=True)


        list_of_classes = np.array(order_list_of_classes)[:,0]

        beam_selection.append(list_of_classes)

    beam_selection_by_comitte = np.array(beam_selection)[:,0]
    #pro top-k
    #beam_selection_by_comitte = np.array(beam_selection)[:,0:5]

    #beam_selection_int = [int(i) for i in beam_selection[:][0]]
    label_validation_int = [int(i) for i in label_validation]
    acuracia_1 = accuracy_score(label_validation_int, beam_selection_by_comitte)

    top_k = [1, 5, 10, 20, 30, 40, 50]

    acuracia = []

    for i in range(len(top_k)):
        acerto = 0
        nao_acerto = 0
        for amostra_a_avaliar in range(len(beam_selection)):
            group = np.array(beam_selection)[:,0:top_k[i]]
            if (label_validation_int[amostra_a_avaliar] in group[amostra_a_avaliar]):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append(acerto / len(beam_selection))

    name_committe = 'media do Score'
    df_acuracia_comite_top_k = pd.DataFrame(acuracia)
    path = '../results/accuracy/8X32/committe/'
    df_acuracia_comite_top_k.to_csv(path + 'acuracia_comite_' + name_committe + '_top_k.csv')

    #plot_results_each_committe(x_data=top_k, y_data=acuracia, name_committe='media do Score')

    return acuracia, name_committe
def committe_position_mean(beam_selection_for_coord, beam_selection_for_lidar, label_validation):

    # number_of_samples = beam_selection_for_coord.shape[0]
    number_of_samples = len(beam_selection_for_coord)


    beam_selection = []

    for i in range(number_of_samples):
        beam_coord = beam_selection_for_coord[i]
        beam_lidar = beam_selection_for_lidar[i]

        #max_degree_coord = max(beam_coord, key=itemgetter('degree'))
        max_degree_coord = max([d['degree'] for d in beam_coord if 'degree' in d])
        max_degree_lidar = max([d['degree'] for d in beam_lidar if 'degree' in d])

        coord_degree_norm = [d['degree']/max_degree_coord for d in beam_coord if 'degree' in d]
        lidar_degree_norm = [d['degree']/max_degree_lidar for d in beam_lidar if 'degree' in d]

        # Adiciona a ponderacao no dicionario
        for j in range(len(beam_coord)):
            beam_coord[j]['norm'] = coord_degree_norm[j]
            beam_lidar[j]['norm'] = lidar_degree_norm[j]


        # extrai todas as classes que estao no dicionario
        #list_of_class_of_beam_coord = [d['class'] for d in beam_coord if 'class' in d]
        list_of_class_of_beam_coord = [int(d['class']) for d in beam_coord if 'class' in d]
        list_of_class_of_beam_lidar = [int(d['class']) for d in beam_lidar if 'class' in d]


        list_of_classes = []
        list_of_classes_postion = []
        list_of_classes_degree_1=[]
        #for i in range(len(beam_lidar)):

        for index, value in enumerate(list_of_class_of_beam_coord):
            if list_of_class_of_beam_lidar[index] in list_of_class_of_beam_coord:
                index_of_class_in_coord = list_of_class_of_beam_coord.index(list_of_class_of_beam_lidar[index])
                index_of_class_in_lidar = index
                mean_of_position_per_class = (index_of_class_in_coord + index_of_class_in_lidar)/2
                classe = int(beam_lidar[index]['class'])
                list_of_classes_postion.append([classe, mean_of_position_per_class])

        order_list_of_classes = sorted(list_of_classes_postion, key=itemgetter(1), reverse=False)

        list_of_classes = np.array(order_list_of_classes)[:,0]

        beam_selection.append(list_of_classes)

    beam_selection_by_comitte = np.array(beam_selection)[:,0]
    #pro top-k
    #beam_selection_by_comitte = np.array(beam_selection)[:,0:5]

    #beam_selection_int = [int(i) for i in beam_selection[:][0]]
    label_validation_int = [int(i) for i in label_validation]
    acuracia_1 = accuracy_score(label_validation_int, beam_selection_by_comitte)

    top_k = [1, 5, 10, 20, 30, 40, 50]

    acuracia = []

    for i in range(len(top_k)):
        acerto = 0
        nao_acerto = 0
        for amostra_a_avaliar in range(len(beam_selection)):
            group = np.array(beam_selection)[:,0:top_k[i]]
            if (label_validation_int[amostra_a_avaliar] in group[amostra_a_avaliar]):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append(acerto / len(beam_selection))

    name_committe = 'media da posicao'
    df_acuracia_comite_top_k = pd.DataFrame(acuracia)
    path='../results/accuracy/8X32/committe/'
    df_acuracia_comite_top_k.to_csv(path + 'acuracia_comite_' + name_committe + '_top_k.csv')

    #plot_results_each_committe(x_data=top_k, y_data=acuracia, name_committe='media da posicao')

    return acuracia, name_committe
def committe_score_geometric_mean(beam_selection_for_coord, beam_selection_for_lidar, label_validation):

    # number_of_samples = beam_selection_for_coord.shape[0]
    number_of_samples = len(beam_selection_for_coord)


    beam_selection = []

    for i in range(number_of_samples):
        beam_coord = beam_selection_for_coord[i]
        beam_lidar = beam_selection_for_lidar[i]

        #max_degree_coord = max(beam_coord, key=itemgetter('degree'))
        max_degree_coord = max([d['degree'] for d in beam_coord if 'degree' in d])
        max_degree_lidar = max([d['degree'] for d in beam_lidar if 'degree' in d])

        coord_degree_norm = [d['degree']/max_degree_coord for d in beam_coord if 'degree' in d]
        lidar_degree_norm = [d['degree']/max_degree_lidar for d in beam_lidar if 'degree' in d]

        # Adiciona a ponderacao no dicionario
        for j in range(len(beam_coord)):
            beam_coord[j]['norm'] = coord_degree_norm[j]
            beam_lidar[j]['norm'] = lidar_degree_norm[j]


        # extrai todas as classes que estao no dicionario
        #list_of_class_of_beam_coord = [d['class'] for d in beam_coord if 'class' in d]
        list_of_class_of_beam_coord = [int(d['class']) for d in beam_coord if 'class' in d]
        list_of_class_of_beam_lidar = [int(d['class']) for d in beam_lidar if 'class' in d]


        list_of_classes = []
        list_of_classes_degree = []
        list_of_classes_degree_1=[]
        #for i in range(len(beam_lidar)):
        for i in range(len(list_of_class_of_beam_lidar)):
            for index, value in enumerate(list_of_class_of_beam_coord):
                #if beam_lidar[i]['class'] == value:
                if int(beam_lidar[i]['class']) == value:
                    geometric_mean_of_degree_norm_per_class_1 = np.sqrt(beam_coord[index]['norm'] * beam_lidar[i]['norm'])
                    #geometric_mean_of_degree_norm_per_class = statistics.geometric_mean([beam_coord[index]['norm'], beam_lidar[i]['norm']])
                    #mean_of_degree_norm_per_class = (beam_coord[index]['norm'] + beam_lidar[i]['norm'])/2
                    #classe = beam_lidar[i]['class']
                    classe = int(beam_lidar[i]['class'])
                    list_of_classes_degree.append([classe, geometric_mean_of_degree_norm_per_class_1])
                    list_of_classes_degree_1.append((classe, geometric_mean_of_degree_norm_per_class_1))

        order_list_of_classes = sorted(list_of_classes_degree, key=itemgetter(1), reverse=True)
        #order_list_of_classes = sorted(list_of_classes, key=itemgetter(1), reverse=True)


        list_of_classes = np.array(order_list_of_classes)[:,0]

        beam_selection.append(list_of_classes)

    beam_selection_by_comitte = np.array(beam_selection)[:,0]
    #pro top-k
    #beam_selection_by_comitte = np.array(beam_selection)[:,0:5]

    #beam_selection_int = [int(i) for i in beam_selection[:][0]]
    label_validation_int = [int(i) for i in label_validation]
    acuracia_1 = accuracy_score(label_validation_int, beam_selection_by_comitte)

    top_k = [1, 5, 10, 20, 30, 40, 50]

    acuracia = []

    for i in range(len(top_k)):
        acerto = 0
        nao_acerto = 0
        for amostra_a_avaliar in range(len(beam_selection)):
            group = np.array(beam_selection)[:,0:top_k[i]]
            if (label_validation_int[amostra_a_avaliar] in group[amostra_a_avaliar]):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append(acerto / len(beam_selection))

    name_committe = 'media geometrica do Score'
    df_acuracia_comite_top_k = pd.DataFrame(acuracia)
    path = '../results/accuracy/8X32/committe/'
    df_acuracia_comite_top_k.to_csv(path + 'acuracia_comite_' + name_committe + '_top_k.csv')

    #plot_results_each_committe(x_data=top_k, y_data=acuracia, name_committe='media do Score')

    return acuracia, name_committe
def committe_difference_between_scores(beam_selection_for_coord, beam_selection_for_lidar, label_validation):

    #number_of_samples = beam_selection_for_coord.shape[0]
    number_of_samples = len(beam_selection_for_coord)
    #number_of_classes_to_evaluated = 12

    beam_selection =[]

    for i in range(number_of_samples):
        beam_coord = beam_selection_for_coord[i]
        #beam_coord = beam_coord[0:number_of_classes_to_evaluated]

        beam_lidar = beam_selection_for_lidar[i]
        #beam_lidar = beam_lidar[0:number_of_classes_to_evaluated]

        #Calcula a ponderacao dos degree em cada vetor de feixes selecionados
        max_degree_coord = max([d['degree'] for d in beam_coord if 'degree' in d])
        max_degree_lidar = max([d['degree'] for d in beam_lidar if 'degree' in d])

        coord_degree_norm = [d['degree']/max_degree_coord for d in beam_coord if 'degree' in d]
        lidar_degree_norm = [d['degree']/max_degree_lidar for d in beam_lidar if 'degree' in d]

        #Adiciona a ponderacao no dicionario
        for j in range(len(beam_coord)):
            beam_coord[j]['norm'] = coord_degree_norm[j]
            beam_lidar[j]['norm'] = lidar_degree_norm[j]

        # extrai todas as classes que estao no dicionario
        list_of_class_of_beam_coord = [int(d['class']) for d in beam_coord if 'class' in d]
        list_of_class_of_beam_lidar = [int(d['class']) for d in beam_lidar if 'class' in d]

        list_of_classes = []
        list_of_classes_degree = []
        list_of_classes_degree_1 = []
        for i in range(len(list_of_class_of_beam_lidar)):
            for index, value in enumerate(list_of_class_of_beam_coord):
                # if beam_lidar[i]['class'] == value:
                if int(beam_lidar[i]['class']) == value:
                    diferenca = abs(beam_coord[index]['norm'] - beam_lidar[i]['norm'])
                    list_of_classes.append([value, diferenca])
                    # list_of_classes.append([beam_lidar[i]['class'], diferenca])
        order_list_of_classes = sorted(list_of_classes, key=itemgetter(1), reverse=False)

        list_of_classes = np.array(order_list_of_classes)[:, 0]

        beam_selection.append(list_of_classes)

    # pro top-k
    # beam_selection_by_comitte = np.array(beam_selection)[:,0:5]

    # beam_selection_int = [int(i) for i in beam_selection[:][0]]
    beam_selection_by_comitte = np.array(beam_selection)[:, 0]
    label_validation_int = [int(i) for i in label_validation]
    acuracia_1 = accuracy_score(label_validation_int, beam_selection_by_comitte)

    top_k = [1, 5, 10, 20, 30, 40, 50]

    acuracia = []

    for i in range(len(top_k)):
        acerto = 0
        nao_acerto = 0
        for amostra_a_avaliar in range(len(beam_selection)):
            group = np.array(beam_selection)[:, 0:top_k[i]]
            if (label_validation_int[amostra_a_avaliar] in group[amostra_a_avaliar]):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append(acerto / len(beam_selection))

    name_committe = 'diferenca_do_score'
    df_acuracia_comite_top_k = pd.DataFrame(acuracia)
    path = '../results/accuracy/8X32/committe/'
    df_acuracia_comite_top_k.to_csv(path + 'acuracia_comite_' + name_committe + '_top_k.csv')

    return acuracia, name_committe


def committe_add_of_score(beam_selection_for_coord, beam_selection_for_lidar, label_validation):

    # number_of_samples = beam_selection_for_coord.shape[0]
    number_of_samples = len(beam_selection_for_coord)


    beam_selection = []

    for i in range(number_of_samples):
        beam_coord = beam_selection_for_coord[i]
        beam_lidar = beam_selection_for_lidar[i]

        #max_degree_coord = max(beam_coord, key=itemgetter('degree'))
        max_degree_coord = max([d['degree'] for d in beam_coord if 'degree' in d])
        max_degree_lidar = max([d['degree'] for d in beam_lidar if 'degree' in d])

        coord_degree_norm = [d['degree']/max_degree_coord for d in beam_coord if 'degree' in d]
        lidar_degree_norm = [d['degree']/max_degree_lidar for d in beam_lidar if 'degree' in d]

        # Adiciona a ponderacao no dicionario
        for j in range(len(beam_coord)):
            beam_coord[j]['norm'] = coord_degree_norm[j]
            beam_lidar[j]['norm'] = lidar_degree_norm[j]


        # extrai todas as classes que estao no dicionario
        #list_of_class_of_beam_coord = [d['class'] for d in beam_coord if 'class' in d]
        list_of_class_of_beam_coord = [int(d['class']) for d in beam_coord if 'class' in d]
        list_of_class_of_beam_lidar = [int(d['class']) for d in beam_lidar if 'class' in d]


        list_of_classes = []
        list_of_classes_degree = []
        list_of_classes_degree_1=[]
        #for i in range(len(beam_lidar)):
        for i in range(len(list_of_class_of_beam_lidar)):
            for index, value in enumerate(list_of_class_of_beam_coord):
                #if beam_lidar[i]['class'] == value:
                if int(beam_lidar[i]['class']) == value:
                    add_of_degree_norm_per_class_1 = beam_coord[index]['norm'] + beam_lidar[i]['norm']
                    #geometric_mean_of_degree_norm_per_class = statistics.geometric_mean([beam_coord[index]['norm'], beam_lidar[i]['norm']])
                    #mean_of_degree_norm_per_class = (beam_coord[index]['norm'] + beam_lidar[i]['norm'])/2
                    #classe = beam_lidar[i]['class']
                    classe = int(beam_lidar[i]['class'])
                    list_of_classes_degree.append([classe, add_of_degree_norm_per_class_1])
                    list_of_classes_degree_1.append((classe, add_of_degree_norm_per_class_1))

        order_list_of_classes = sorted(list_of_classes_degree, key=itemgetter(1), reverse=True)
        #order_list_of_classes = sorted(list_of_classes, key=itemgetter(1), reverse=True)


        list_of_classes = np.array(order_list_of_classes)[:,0]

        beam_selection.append(list_of_classes)

    beam_selection_by_comitte = np.array(beam_selection)[:,0]
    #pro top-k
    #beam_selection_by_comitte = np.array(beam_selection)[:,0:5]

    #beam_selection_int = [int(i) for i in beam_selection[:][0]]
    label_validation_int = [int(i) for i in label_validation]
    acuracia_1 = accuracy_score(label_validation_int, beam_selection_by_comitte)

    top_k = [1, 5, 10, 20, 30, 40, 50]

    acuracia = []

    for i in range(len(top_k)):
        acerto = 0
        nao_acerto = 0
        for amostra_a_avaliar in range(len(beam_selection)):
            group = np.array(beam_selection)[:,0:top_k[i]]
            if (label_validation_int[amostra_a_avaliar] in group[amostra_a_avaliar]):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append(acerto / len(beam_selection))

    name_committe = 'soma_do_score'
    df_acuracia_comite_top_k = pd.DataFrame(acuracia)
    path = '../results/accuracy/8X32/committe/'
    df_acuracia_comite_top_k.to_csv(path + 'acuracia_comite_' + name_committe + '_top_k.csv')

    #plot_results_each_committe(x_data=top_k, y_data=acuracia, name_committe='media do Score')

    return acuracia, name_committe

def committe_dot_of_score(beam_selection_for_coord, beam_selection_for_lidar, label_validation):

    # number_of_samples = beam_selection_for_coord.shape[0]
    number_of_samples = len(beam_selection_for_coord)


    beam_selection = []

    for i in range(number_of_samples):
        beam_coord = beam_selection_for_coord[i]
        beam_lidar = beam_selection_for_lidar[i]

        #max_degree_coord = max(beam_coord, key=itemgetter('degree'))
        max_degree_coord = max([d['degree'] for d in beam_coord if 'degree' in d])
        max_degree_lidar = max([d['degree'] for d in beam_lidar if 'degree' in d])

        coord_degree_norm = [d['degree']/max_degree_coord for d in beam_coord if 'degree' in d]
        lidar_degree_norm = [d['degree']/max_degree_lidar for d in beam_lidar if 'degree' in d]

        # Adiciona a ponderacao no dicionario
        for j in range(len(beam_coord)):
            beam_coord[j]['norm'] = coord_degree_norm[j]
            beam_lidar[j]['norm'] = lidar_degree_norm[j]


        # extrai todas as classes que estao no dicionario
        #list_of_class_of_beam_coord = [d['class'] for d in beam_coord if 'class' in d]
        list_of_class_of_beam_coord = [int(d['class']) for d in beam_coord if 'class' in d]
        list_of_class_of_beam_lidar = [int(d['class']) for d in beam_lidar if 'class' in d]


        list_of_classes = []
        list_of_classes_degree = []
        list_of_classes_degree_1=[]
        #for i in range(len(beam_lidar)):
        for i in range(len(list_of_class_of_beam_lidar)):
            for index, value in enumerate(list_of_class_of_beam_coord):
                #if beam_lidar[i]['class'] == value:
                if int(beam_lidar[i]['class']) == value:
                    add_of_degree_norm_per_class_1 = beam_coord[index]['norm'] * beam_lidar[i]['norm']
                    #geometric_mean_of_degree_norm_per_class = statistics.geometric_mean([beam_coord[index]['norm'], beam_lidar[i]['norm']])
                    #mean_of_degree_norm_per_class = (beam_coord[index]['norm'] + beam_lidar[i]['norm'])/2
                    #classe = beam_lidar[i]['class']
                    classe = int(beam_lidar[i]['class'])
                    list_of_classes_degree.append([classe, add_of_degree_norm_per_class_1])
                    list_of_classes_degree_1.append((classe, add_of_degree_norm_per_class_1))

        order_list_of_classes = sorted(list_of_classes_degree, key=itemgetter(1), reverse=True)
        #order_list_of_classes = sorted(list_of_classes, key=itemgetter(1), reverse=True)


        list_of_classes = np.array(order_list_of_classes)[:,0]

        beam_selection.append(list_of_classes)

    beam_selection_by_comitte = np.array(beam_selection)[:,0]
    #pro top-k
    #beam_selection_by_comitte = np.array(beam_selection)[:,0:5]

    #beam_selection_int = [int(i) for i in beam_selection[:][0]]
    label_validation_int = [int(i) for i in label_validation]
    acuracia_1 = accuracy_score(label_validation_int, beam_selection_by_comitte)

    top_k = [1, 5, 10, 20, 30, 40, 50]

    acuracia = []

    for i in range(len(top_k)):
        acerto = 0
        nao_acerto = 0
        for amostra_a_avaliar in range(len(beam_selection)):
            group = np.array(beam_selection)[:,0:top_k[i]]
            if (label_validation_int[amostra_a_avaliar] in group[amostra_a_avaliar]):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append(acerto / len(beam_selection))

    name_committe = 'produto_do_score'
    df_acuracia_comite_top_k = pd.DataFrame(acuracia)
    path = '../results/accuracy/8X32/committe/'
    df_acuracia_comite_top_k.to_csv(path + 'acuracia_comite_' + name_committe + '_top_k.csv')

    #plot_results_each_committe(x_data=top_k, y_data=acuracia, name_committe='media do Score')

    return acuracia, name_committe

def plot_results_each_committe( y_data, name_committe):

    sns.set()

    path='../results/accuracy/8X32/committe/'
    title_figure = "Desempenho do comite " + name_committe

    top_k = [1, 5, 10, 20, 30, 40, 50]
    x_data = top_k
    plt.plot(x_data, y_data, color='g', linestyle='dashed')#, label=label_1)  # linestyle = 'solid'
    plt.plot(x_data, y_data, 'go')


    plt.xlabel('top-k')
    plt.ylabel('Acuracia')
    # plt.text(x_pos_tex, y_pos_tex, text)
    # plt.ylim(min_y_lim, max_y_lim)
    plt.title(title_figure, fontsize=11)
    plt.legend()
    plt.grid(True)
    plt.savefig(path + name_committe, dpi=300, bbox_inches='tight')
    plt.show()
def plot_results(x_data_1, name_committe_1,
                 x_data_2, name_committe_2):
    top_k = [1, 5, 10, 20, 30, 40, 50]
    sns.set()

    path='../results/accuracy/8X32/committe/'
    title_figure = "comparacao de Desempenho de comites "

    plt.plot(top_k, x_data_1, color='g', linestyle='dashed', label=name_committe_1)  # linestyle = 'solid'
    plt.plot(top_k, x_data_1, 'go')
    plt.plot(top_k, x_data_2, color='b', linestyle='dashed', label=name_committe_2)  # linestyle = 'solid'
    plt.plot(top_k, x_data_2, 'bo')


    plt.xlabel('top-k')
    plt.ylabel('Acuracia')
    # plt.text(x_pos_tex, y_pos_tex, text)
    # plt.ylim(min_y_lim, max_y_lim)
    plt.title(title_figure, fontsize=11)
    plt.legend()
    plt.grid(True)
    plt.savefig(path + 'comparacao_comites', dpi=300, bbox_inches='tight')
    plt.show()
def plot_results_from_csv():


    top_k = [1, 5, 10, 20, 30, 40, 50]

    # Read files
    path='../results/accuracy/8X32/committe/'

    file_postion = 'acuracia_comite_media da posicao_top_k'
    file = path + file_postion + '.csv'
    data = pd.read_csv(file, usecols=[1])
    data_of_postion_committe = data.to_numpy()
    name_committe_1 = 'media da posicao'

    file_score = 'acuracia_comite_media do Score_top_k'
    file = path + file_score + '.csv'
    data = pd.read_csv(file, usecols=[1])
    data_of_score_committe = data.to_numpy()
    name_committe_2 = 'Score: média'

    file_score_geometric_mean = 'acuracia_comite_media geometrica do Score_top_k'
    file = path + file_score_geometric_mean + '.csv'
    data = pd.read_csv(file, usecols=[1])
    data_of_score_media_geometrica = data.to_numpy()
    name_committe_media_geometrica_score = 'Score: media geometrica'

    file_score_difference = 'acuracia_comite_diferenca_do_score_top_k'
    file = path + file_score_difference + '.csv'
    data = pd.read_csv(file, usecols=[1])
    data_of_score_difference = data.to_numpy()
    name_committe_difference_score = 'Score: Diferença'

    file_score_add = 'acuracia_comite_soma_do_score_top_k'
    file = path + file_score_add + '.csv'
    data = pd.read_csv(file, usecols=[1])
    data_of_score_add = data.to_numpy()
    name_committe_add_score = 'Score: soma'

    file_score_dot = 'acuracia_comite_produto_do_score_top_k'
    file = path + file_score_dot + '.csv'
    data = pd.read_csv(file, usecols=[1])
    data_of_score_dot = data.to_numpy()
    name_committe_dot_score = 'Score: produto'


    sns.set()

    path = '../results/accuracy/8X32/committe/'


    title_figure = "comparacao de Desempenho de comites "

    plt.plot(top_k, data_of_postion_committe, color='g', linestyle='solid',
             label=name_committe_1)  # linestyle = 'dashed' 'dotted'
    plt.plot(top_k, data_of_postion_committe, 'go')

    plt.plot(top_k, data_of_score_committe, color='darkgrey', linestyle='dashed',
             label=name_committe_2, linewidth=1)#, alpha=.8)  # linestyle = 'solid'
    plt.plot(top_k, data_of_score_committe, 'bo')

    plt.plot(top_k, data_of_score_media_geometrica, color='r', linestyle='dashed',
             label=name_committe_media_geometrica_score, linewidth=1)#, alpha=.5)  # linestyle = 'solid'
    plt.plot(top_k, data_of_score_media_geometrica, 'ro')


    plt.plot(top_k, data_of_score_difference, color='y', linestyle='dashed',
             label=name_committe_difference_score)  # linestyle = 'solid'
    plt.plot(top_k, data_of_score_difference, 'yo')

    plt.plot(top_k, data_of_score_add, color='black', linestyle='dashed',
             label=name_committe_add_score)  # linestyle = 'solid'
    plt.plot(top_k, data_of_score_add, 'b-')

    plt.plot(top_k, data_of_score_dot, color='m', linestyle='dashed',
             label=name_committe_dot_score)  # linestyle = 'solid'
    plt.plot(top_k, data_of_score_dot, 'mo')

    for x, y in zip(top_k, data_of_postion_committe):
        label = "{:.2f}".format(y[0])

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, -15),  # distance from text to points (x,y)
                     ha='left',# horizontal alignment can be left, right or center
                     color='g',
                     fontsize=8)

    for x, y in zip(top_k, data_of_score_committe):
        label = "{:.2f}".format(y[0])

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 7),  # distance from text to points (x,y)
                     ha='right',# horizontal alignment can be left, right or center
                     color='b',
                     fontsize=8)

    for x, y in zip(top_k, data_of_score_difference):
        label = "{:.2f}".format(y[0])

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 7),  # distance from text to points (x,y)
                     ha='right',# horizontal alignment can be left, right or center
                     color='y',
                     fontsize=8)



    plt.xlabel('top-k')
    plt.ylabel('Acurácia')
    # plt.text(x_pos_tex, y_pos_tex, text)
    # plt.ylim(min_y_lim, max_y_lim)
    plt.title(title_figure, fontsize=11)
    plt.legend()
    plt.grid(True)
    plt.savefig(path + 'comparacao_comites', dpi=300, bbox_inches='tight')
    plt.show()

def plot_results_in_bars_from_csv():
    top_k = [1, 5, 10, 20, 30, 40, 50]

    # Read files
    path = '../results/accuracy/8X32/committe/'

    file_postion = 'acuracia_comite_media da posicao_top_k'
    file = path + file_postion + '.csv'
    data_position = pd.read_csv(file, usecols=[1])
    data_position_1 = data_position.to_numpy()
    name_committe_data_postion = 'media da posicao'
    data_of_postion_committe = []
    for i in range(len(data_position_1)):
        data_of_postion_committe.append(data_position_1[i][0])


    file_score = 'acuracia_comite_media do Score_top_k'
    file = path + file_score + '.csv'
    data = pd.read_csv(file, usecols=[1])
    data_of_media_score = data.to_numpy()
    name_committe_media_score = 'Score: média'
    data_of_score_committe=[]
    for i in range(len(data_of_media_score)):
        data_of_score_committe.append(data_of_media_score[i][0])



    file_score_geometric_mean = 'acuracia_comite_media geometrica do Score_top_k'
    file = path + file_score_geometric_mean + '.csv'
    data = pd.read_csv(file, usecols=[1])
    data_media_geometrica = data.to_numpy()
    name_committe_media_geometrica_score = 'Score: media geometrica'
    data_of_score_media_geometrica=[]
    for i in range(len(data_media_geometrica)):
        data_of_score_media_geometrica.append(data_media_geometrica[i][0])




    file_score_add = 'acuracia_comite_soma_do_score_top_k'
    name_committe_add_score = 'Score: soma'
    file = path + file_score_add + '.csv'
    data = pd.read_csv(file, usecols=[1])
    data_add = data.to_numpy()
    data_of_score_add = []
    for i in range(len(data_add)):
        data_of_score_add.append(data_add[i][0])


    file_score_difference = 'acuracia_comite_diferenca_do_score_top_k'
    name_committe_difference_score = 'Score: Diferença'
    file = path + file_score_difference + '.csv'
    data_difference = pd.read_csv(file, usecols=[1])
    data_difference = data_difference.to_numpy()
    data_of_score_difference = []
    for i in range(len(data_difference)):
        data_of_score_difference.append(data_difference[i][0])


    file_score_dot = 'acuracia_comite_produto_do_score_top_k'
    name_committe_dot_score = 'Score: produto'
    file = path + file_score_dot + '.csv'
    data = pd.read_csv(file, usecols=[1])
    data_np = data.to_numpy()
    data_of_score_dot = []
    for i in range(len(data_np)):
        data_of_score_dot.append(data_np[i][0])


    test = np.arange(1, 71, 10)

    #sns.set()

    path = '../results/accuracy/8X32/committe/'

    title_figure = "comparacao de Desempenho de comites "

    width = 1
    size_font=6

    #fig = plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(right=0.856, left= 0.074)

    # creating the bar plot
    bars_position = ax.bar(test, data_of_postion_committe, color='maroon', width=width, label=name_committe_data_postion)
    for bar in bars_position:
        height = bar.get_height()
        ax.annotate('{:.3}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height-0.07),
                    fontsize=size_font, rotation=90, ha='center', va='bottom', color='w')

    bars_score_dot = ax.bar([i + 1 for i in test], data_of_score_dot, color='lightyellow', width=width, label=name_committe_dot_score)
    for bar in bars_score_dot:
        height = bar.get_height()
        ax.annotate('{:.3}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height-0.07),
                    ha='center', va='bottom', rotation=90, fontsize=size_font) #textcoords="offset points", xytext=(0, 3),


    bars_score_difference = ax.bar([i-1 for i in test], data_of_score_difference, color='lightgray', width=width, label=name_committe_difference_score)
    for bar in bars_score_difference:
        height = bar.get_height()
        ax.annotate('{:.3}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height-0.07),
                    ha='center', va='bottom', rotation=90, fontsize=size_font)

    bars_score_add = ax.bar([i+2 for i in test], data_of_score_add, color='c', width=width, label=name_committe_add_score)
    for bar in bars_score_add:
        height = bar.get_height()
        ax.annotate('{:.3}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height-0.07),
                    ha='center', va='bottom', rotation=90, fontsize=size_font)

    bar_score_media_geometrica = ax.bar([i+3 for i in test], data_of_score_media_geometrica, color='lightsalmon', width=width, label=name_committe_media_geometrica_score)
    for bar in bar_score_media_geometrica:
        height = bar.get_height()
        ax.annotate('{:.3}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height-0.07),
                    ha='center', va='bottom', rotation=90, fontsize=size_font)

    bars_score_media = ax.bar([i+4 for i in test], data_of_score_committe, color='darkcyan', width=width, label=name_committe_media_score)
    for bar in bars_score_media:
        height = bar.get_height()
        ax.annotate('{:.3}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height-0.07),
                    ha='center', va='bottom', rotation=90, fontsize=size_font, color='w')

    ax.set_ylabel('Acurácia')
    ax.set_xlabel('top-k')
    ax.set_title('Comparacao entre comites')
    ax.set_xticks(test, top_k)
    #ax.set_yticks(fontsize=16) # This ensures we have one tick per year, otherwise we get fewer
    #ax.set_xticklabels(top_k)#, rotation='vertical')
    #ax.legend(frameon=False, fontsize=8, loc='upper right', bbox_to_anchor=(1.1, 1), fancybox=True, shadow=True, borderaxespad=0)#, ncol=3)

   # plt.bar(top_k, data_of_score_dot, color='maroon', width=0.4, label=name_committe_dot_score)
    #plt.bar(top_k, data_of_score_add, color='blue', width=0.4, label=name_committe_add_score)
    #plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),frameon=False, fontsize=8)
    #plt.xlabel("Top-k")
    #plt.ylabel("Acurácia")
    #plt.title("Comparacao entre comites")

    plt.show()
    a=0


#committe()
plot_results_in_bars_from_csv()







