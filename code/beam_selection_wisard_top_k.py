import numpy as np
import wisardpkg as wp
from operator import itemgetter
import pandas as pd

import numpy as np

def beam_selection_top_k_wisard(x_train, x_test, y_train, y_test, data_input):
    #x_train, x_test = obj_read_input.read_all_Qs_matrix()
    #label_rx_LOS_train, label_rx_LOS_test, label_tx_LOS_train, label_tx_LOS_test, label_combined_LOS_train, label_combined_LOS_test = obj_read_label.read_LOS_beams(antenna_config='8X32')
    #index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test, index_beam_combined_train, index_beam_combined_test = obj_read_label.read_all_beams(antenna_config='8X32')
    #y_train = index_beam_combined_train
    #y_test = index_beam_combined_test


    #data_input = 'coord_in_termometro'
    #data_input ='coord_in_termometro_+_Lidar'
    data_set = 'all'

    #addressSize = 44
    addressSize = 64
    ignoreZero = False
    verbose = True
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

    content_index = 0
    ram_index = 0
    #print(wsd.getsizeof(ram_index,content_index))
    #print(wsd.json())
    #print(out)

    top_k=[1,5,10,20,30,40,50]
    #top_k=[5]
    acerto = 0
    nao_acerto = 0
    acuracia = []

    f = open('../results/accuracy/8X32/'+data_input+'/Acurcacia_top_k_wisard.txt', 'w')


    for i in range(len(top_k)):
        acerto = 0
        nao_acerto = 0
        for amostra_a_avaliar in range(len(out)):

            lista_das_classes =out[amostra_a_avaliar]['classesDegrees']
            dict_com_classes_na_ordem = sorted(lista_das_classes, key=itemgetter('degree'), reverse=True)
            f.write(str(dict_com_classes_na_ordem))



            classes_na_ordem_descendente = []
            for x in range(len(dict_com_classes_na_ordem)):
                classes_na_ordem_descendente.append(dict_com_classes_na_ordem[x]['class'])

            top_5 = classes_na_ordem_descendente[0:top_k[i]]

            if( y_test[amostra_a_avaliar] in top_5):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append(acerto/len(out))

    print("len(out):", len(out))
    print("TOP-K: ", top_k)
    print("Acuracia: ",acuracia)
    f.close()

    df_acuracia_wisard_top_k = pd.DataFrame(acuracia)
    ruta='../results/accuracy/8X32/'+data_input+'/'
    print(ruta+'acuracia_wisard_' + data_input + '_' + data_set + '_top_k.csv')
    df_acuracia_wisard_top_k.to_csv(ruta + 'acuracia_wisard_' + data_input + '_' + data_set + '_top_k.csv')

    return top_k, acuracia