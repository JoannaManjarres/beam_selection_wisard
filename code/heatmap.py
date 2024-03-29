import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv


def read_data_for_dilated_lidar(folder):
    path = '../results/accuracy/8X32/'+folder+'/Combined/'

    size_memory = [6,12,18,24,28,34,38,44,48,54,58,64]
    iterations = [1,2,3,4]

    all_accuracy = []
    for i in range(1, 5):
        file_name = 'acuracia_beam_selection_Combined_[8X32]_'+folder+'_ALL_' + str(i) + '_it.csv'

        print(file_name)
        filename = path + file_name
        name_1 = 'Tamanho_memoria_' + str(i)
        name_2 = 'Acuracia_' + str(i)
        name_3 = 'intervalo_conf_' + str(i)
        df = pd.read_csv(filename,
                         sep='\t',
                         names=[name_1, name_2, name_3])

        accuracy = df[name_2].to_numpy()
        all_accuracy.append(accuracy)


    return all_accuracy, size_memory, iterations, path


def plot_headmap(all_data,
                 x_axis_labels_vector,
                 y_axis_labels_vector,
                 x_label,
                 y_label,
                 path,
                 folder):


    y_axis_labels = y_axis_labels_vector
    x_axis_labels = x_axis_labels_vector

    # plotting the heatmap
    plt.figure(figsize=(6, 4), dpi=200)

    annot_kws = {'fontsize': 10,
                 'fontstyle': 'italic',
                 'color': "k"}

    hm = sns.heatmap(data=all_data,
                     annot=False,
                     cmap="YlGnBu",
                     xticklabels=x_axis_labels,
                     yticklabels=y_axis_labels,
                     )
    hm.set_xlabel(x_label, color='steelblue', size=14, fontweight='bold')
    hm.set_ylabel(y_label, color='steelblue', size=14, fontweight='bold')

    # displaying the plotted heatmap
    plt.subplots_adjust(right=1, left=0.09)
    plt.savefig(path+'1_headmap_'+folder)
    plt.show()
def read_data_for_coord_in_termometro():

    rota = '../results/accuracy/8X32/coord_in_termometro_iguais/Combined/'
    file_name = 'acuracia_beam_selection_Combined_[8X32]_coord_in_termometro_iguais_12_ALL.csv'
    filename = rota+file_name

    '''
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
    
        all_info = np.zeros([12, 2], dtype=object)
    
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            cont = 0
            for row in reader:
                all_info[cont] = int(row['tamanho_memoria']), float(row['Acuracia'])
                cont += 1
    '''


    df_1 = pd.read_csv(filename,
                     sep='\t',
                     names=['Tamanho_memoria', 'Acuracia', 'intervalo_conf'])

    rota = '../results/accuracy/8X32/coord_in_termometro_iguais/Combined/'

    j=12
    #all_info = [6, 12, 18,24,28,34,38,44,48,54,58,64]
    all_info = []
    resolucao = []
    for i in range(1,8):
        file_name = 'acuracia_beam_selection_Combined_[8X32]_coord_in_termometro_iguais_'+str(j)+'_ALL.csv'


        print(file_name)
        filename = rota+file_name
        name_1 = 'Tamanho_memoria_'+str(j)
        name_2 = 'Acuracia_'+str(j)
        name_3 = 'intervalo_conf_'+str(j)
        df = pd.read_csv(filename,
                     sep='\t',
                     names=[name_1, name_2, name_3])

        data = df[name_2].to_numpy()
        all_info.append(data)

        resolucao.append(j)
        j = j * 2
        #all_info = pd.concat([all_info, pd], axis=1)

    #print(df_1)
    #print(df_2)

    print(all_info)
    size_memory = [6, 12, 18, 24, 28, 34, 38, 44, 48, 54, 58, 64]
    #resolucao = [12, 24, 3, 4]

    rota = '../results/accuracy/article/'

    plot_headmap(all_info, size_memory, resolucao, 'Address Size', 'Resolution', rota, 'coord_in_termometro_iguais')
    #return all_accuracy, size_memory, resolucao, rota


def read_data_for_coord_in_termometro_iguais_com_decimal():
    rota = '../results/accuracy/8X32/coord_in_termometro_iguais_com_decimal/Combined/'
    file_name = 'acuracia_beam_selection_Combined_[8X32]_coord_in_termometro_iguais_com_decimal_1_ALL.csv'
    filename = rota + file_name

    '''
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)

        all_info = np.zeros([12, 2], dtype=object)

        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            cont = 0
            for row in reader:
                all_info[cont] = int(row['tamanho_memoria']), float(row['Acuracia'])
                cont += 1
    '''

    df_1 = pd.read_csv(filename,
                       sep='\t',
                       names=['Tamanho_memoria', 'Acuracia', 'intervalo_conf'])

    rota = '../results/accuracy/8X32/coord_in_termometro_iguais_com_decimal/Combined/'
    j = 1
    # all_info = [6, 12, 18,24,28,34,38,44,48,54,58,64]
    all_info = []
    resolucao = []
    for i in range(1, 6):
        file_name = 'acuracia_beam_selection_Combined_[8X32]_coord_in_termometro_iguais_com_decimal_' + str(j) + '_ALL.csv'

        print(file_name)
        filename = rota + file_name
        name_1 = 'Tamanho_memoria_' + str(j)
        name_2 = 'Acuracia_' + str(j)
        name_3 = 'intervalo_conf_' + str(j)
        df = pd.read_csv(filename,
                         sep='\t',
                         names=[name_1, name_2, name_3])

        data = df[name_2].to_numpy()
        all_info.append(data)

        resolucao.append(j)
        j = j * 2
        # all_info = pd.concat([all_info, pd], axis=1)

    # print(df_1)
    # print(df_2)

    print(all_info)
    size_memory = [6, 12, 18, 24, 28, 34, 38, 44, 48, 54, 58, 64]
    # resolucao = [12, 24, 3, 4]

    plot_headmap(all_info, size_memory, resolucao, 'Tamanho da memoria', 'Resolução', rota,
                 'coord_in_termometro_iguais')
    # return all_accuracy, size_memory, resolucao, rota

def read_coord_in_Thermomether_x_y_unbalanced():
    rota = '../results/accuracy/8X32/coord_in_Thermomether_x_y_unbalanced/Combined/'
    j = 1
    # all_info = [6, 12, 18,24,28,34,38,44,48,54,58,64]
    all_info = []
    resolucao = []
    for i in range(1,11):
        file_name = 'acuracia_beam_selection_Combined_[8X32]_coord_in_Thermomether_x_y_unbalanced_' + str(
            j) + '_ALL.csv'

        print(file_name)
        filename = rota + file_name
        name_1 = 'Tamanho_memoria_' + str(j)
        name_2 = 'Acuracia_' + str(j)
        name_3 = 'intervalo_conf_' + str(j)
        df = pd.read_csv(filename,
                         sep='\t',
                         names=[name_1, name_2, name_3])

        data = df[name_2].to_numpy()
        all_info.append(data)

        resolucao.append(j)
        j = j * 2
        # all_info = pd.concat([all_info, pd], axis=1)

    # print(df_1)
    # print(df_2)

    print(all_info)
    size_memory = [6, 12, 18, 24, 28, 34, 38, 44, 48, 54, 58, 64]
    # resolucao = [12, 24, 3, 4]

    rota = '../results/accuracy/article/'
    plot_headmap(all_info, size_memory, resolucao, 'Address size', 'Resolution', rota,
                 'read_coord_in_Thermomether_x_y_unbalanced')


def read_coord_in_Thermomether_x_y_unbalanced_with_decimal_part():
    rota = '../results/accuracy/8X32/coord_in_Thermomether_x_y_unbalanced_with_decimal_part/Combined/'
    j = 1
    # all_info = [6, 12, 18,24,28,34,38,44,48,54,58,64]
    all_info = []
    resolucao = []
    for i in range(1,7):
        file_name = 'acuracia_beam_selection_Combined_[8X32]_coord_in_Thermomether_x_y_unbalanced_with_decimal_part_' + str(
            j) + '_ALL.csv'

        print(file_name)
        filename = rota + file_name
        name_1 = 'Tamanho_memoria_' + str(j)
        name_2 = 'Acuracia_' + str(j)
        name_3 = 'intervalo_conf_' + str(j)
        df = pd.read_csv(filename,
                         sep='\t',
                         names=[name_1, name_2, name_3])

        data = df[name_2].to_numpy()
        all_info.append(data)

        resolucao.append(j)
        j = j * 2
        # all_info = pd.concat([all_info, pd], axis=1)

    # print(df_1)
    # print(df_2)

    print(all_info)
    size_memory = [6, 12, 18, 24, 28, 34, 38, 44, 48, 54, 58, 64]
    # resolucao = [12, 24, 3, 4]

    plot_headmap(all_info, size_memory, resolucao, 'Tamanho da memoria', 'Resolução', rota,
                 'coord_in_Thermomether_x_y_unbalanced_with_decimal_part')

#read_data_for_coord_in_termometro()
#read_data_for_coord_in_termometro()
#read_data_for_coord_in_termometro_iguais_com_decimal()
read_coord_in_Thermomether_x_y_unbalanced()
read_coord_in_Thermomether_x_y_unbalanced_with_decimal_part()

folder = 'all_data_dilated'
folder = 'all_data_dilated_+_rx_as_cube'
#folder = 'all_data_lidar_dilated_+_rx_like_cube_to_tx'

all_accuracy, size_memory, iterations, path = read_data_for_dilated_lidar(folder)
plot_headmap(all_accuracy, size_memory, iterations, 'Tamanho da memoria', 'Iteraçoes da dilatação', path, folder)
a=0



