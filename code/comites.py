import wisardpkg as wp
from operator import itemgetter
import pandas as pd
from sklearn.metrics import accuracy_score

import preprocesamento as obj_coord
import pre_process_lidar as obj_lidar
import read_data as read_labels
import numpy as np

def read_coordinates():
    all_coord_in_Thermomether_x_y_unbalanced_train, all_coord_in_Thermomether_x_y_unbalanced_test = obj_coord.read_coord_in_Thermomether_x_y_unbalanced()
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

def beam_selection_commite(x_train, x_test, y_train, y_test,addressSize):
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


def commite(beam_selection_for_coord, beam_selection_for_lidar, label_validation):

    #number_of_samples = beam_selection_for_coord.shape[0]
    number_of_samples = len(beam_selection_for_coord)
    number_of_classes_to_evaluated = 12

    beam_selection =[]

    for i in range(number_of_samples):
        beam_coord = beam_selection_for_coord[i]
        beam_coord = beam_coord[0:number_of_classes_to_evaluated]

        beam_lidar = beam_selection_for_lidar[i]
        beam_lidar = beam_lidar[0:number_of_classes_to_evaluated]

        if beam_coord[0]['class'] == beam_lidar[0]['class']:
            beam_selection.append(beam_coord[0]['class'])
        else:
            #max_a = max(a, key=itemgetter('degree'))
            #max_b = max(b, key=itemgetter('degree'))

            #Calcula a ponderacao dos degree em cada vetor de feixes selecionados
            sum_degree_beam_coord = sum([d['degree'] for d in beam_coord if 'degree' in d])
            sum_degree_beam_lidar = sum([d['degree'] for d in beam_lidar if 'degree' in d])

            coord_degree_norm = [d['degree']/sum_degree_beam_coord for d in beam_coord if 'degree' in d]
            lidar_degree_norm = [d['degree']/sum_degree_beam_lidar for d in beam_lidar if 'degree' in d]

            #Adiciona a ponderacao no dicionario
            for j in range(number_of_classes_to_evaluated):
                beam_coord[j]['norm'] = coord_degree_norm[j]
                beam_lidar[j]['norm'] = lidar_degree_norm[j]


            # Busca o feixe selecionado pelo lidar nos feixes selecionados pelas coordenadas
            #class_in_coord = [d for d in beam_coord if beam_lidar[0]['class'] in d.values()]  #faltou incluir o else para usar esta sentencia

            for d in beam_coord:
                if beam_lidar[0]['class'] in d.values():
                    class_for_lidar_in_coord = [d]
                else :
                    class_for_lidar_in_coord = [beam_lidar[0]]

            # Busca o feixe selecionado pelas coordenadas nos feixes selecionados pelo lidar
            #class_in_lidar = [d for d in beam_lidar if beam_coord[0]['class'] in d.values()] #faltou incluir o else para usar esta sentencia

            for d in beam_lidar:
                if beam_coord[0]['class'] in d.values():
                    class_for_coord_in_lidar = [d]
                else :
                    class_for_coord_in_lidar = [beam_coord[0]]


            if class_for_lidar_in_coord[0]['norm'] > class_for_coord_in_lidar[0]['norm']:
                #print(beam_lidar[0]['class'])
                #beam_selection[number_of_samples] = beam_lidar[0]['class']
                beam_selection.append(beam_lidar[0]['class'])
            else:
                beam_selection.append(beam_coord[0]['class'])

    acuracia = accuracy_score(label_validation, beam_selection)
    print('acuracia: ', acuracia)

    return beam_selection

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

commite(beam_selection_for_coord, beam_selection_for_lidar, label_validation=index_beam_combined_test)
