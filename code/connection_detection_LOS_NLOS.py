import wisardpkg as wp
import timeit
import csv
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score
from sklearn import  svm, metrics
import preprocesamento as obj_pre_preprocessing
import pre_process_lidar as obj_read_lidar
import seaborn as sns
import pandas as pd
import numpy as np




def tic():
    global tic_s
    tic_s = timeit.default_timer()
def toc():
    global tic_s
    toc_s = timeit.default_timer()

    return (toc_s - tic_s)
def config_red_wisard(addressSize):
    # addressSize # number of addressing bits in the ram
    ignoreZero = False  # optional; causes the rams to ignore the address 0

    # False by default for performance reasons,
    # when True, WiSARD prints the progress of train() and classify()
    verbose = False

    wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)
    return wsd
def redWizard(data_train,
              label_train,
              data_validation,
              addressSize):
    # addressSize # number of addressing bits in the ram
    ignoreZero = False  # optional; causes the rams to ignore the address 0

    # False by default for performance reasons,
    # when True, WiSARD prints the progress of train() and classify()
    verbose = False

    bleachingActivated = False #this enable or disable the bleaching process on the classification.
    completeAddressing = True
    wsd = wp.Wisard(addressSize,
                    ignoreZero=ignoreZero,
                    verbose=verbose,
                    bleachingActivated=bleachingActivated,
                    completeAddressing=completeAddressing
                    )

    print('\n Training WISARD net ...')
    tic()
    # train using the input data
    wsd.train(data_train, label_train)

    tiempo_entrenamiento_ms = toc()

    print('\n classification of connection using WISARD network ...')
    tic()
    # classify some data
    salida_de_la_red = wsd.classify(data_validation)
    tiempo_test_ms = toc()

    return salida_de_la_red, tiempo_entrenamiento_ms, tiempo_test_ms

def calculoDesvioPadrao(input_vector):
    sumatoria = 0
    numero_de_elementos = len(input_vector)
    for i in range(numero_de_elementos):
        sumatoria = sumatoria + input_vector[i]

    media = sumatoria / numero_de_elementos
    sumatoria = 0
    for i in range(numero_de_elementos):
        sumatoria = + (input_vector[i] - media) ** 2
    desvio_padrao = math.sqrt(sumatoria / numero_de_elementos)

    return [media, desvio_padrao]

def plotarResultados(x_vector,
                     y_vector,
                     desvio_padrao_vector,
                     titulo,
                     nombre_curva,
                     x_label,
                     y_label,
                     ruta="figura.png"):
    plt.figure()
    plt.errorbar(x_vector, y_vector, yerr=desvio_padrao_vector, fmt='o', label=nombre_curva, capsize=5, ecolor='red')

    plt.legend(loc="best")
    plt.title(titulo)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
def LOS_NLOS_WISARD_classification(input_train,
                             input_validation,
                             label_train,
                             label_validation,
                             figure_name,
                             antenna_config,
                             type_of_input,
                             titulo_figura,
                             user,
                             enableDebug=True,
                             plot_confusion_matrix_enable=False):

    # config parameters
    if (enableDebug):
        address_size = [28]
        numero_experimentos = 2
    else:
        #address_size = [64]
        address_size = [6,12,18,24,28,34,38,44,48,54,58,64]
        numero_experimentos = 10

    vector_time_train_media = []
    vector_time_test_media = []
    vector_acuracia_media = []

    vector_acuracia_desvio_padrao = []
    vector_time_train_desvio_padrao = []
    vector_time_test_desvio_padrao = []

    path_result = "../results"

    for j in range(len(address_size)):  # For encargado de variar el tamano de la memoria

        vector_acuracia = []
        vector_time_test = []
        vector_time_train = []
        vector_matriz_confusion = []
        #matriz_confusion_sumatoria = np.zeros((numero_de_grupos, numero_de_grupos), dtype=float)

        print("Tamanho memoria: " + str(address_size[j]))

        for i in range(numero_experimentos):  # For encargado de ejecutar el numero de rodadas (experimentos)
            print("   Experimento: " + str(i))

            # -----------------USA LA RED WIZARD -------------------
            out_red, time_train, time_test = redWizard(input_train,
                                                       label_train,
                                                       input_validation,
                                                       address_size[j])

            vector_time_train.append(time_train)
            vector_time_test.append(time_test)

            # #----------------- CALCULA MATRIZ DE CONFUSION -----------------------
            titulo = "** MATRIZ DE CONFUSÃO " + str(i) + " **" + " \n Address Size " + str(address_size[j])

            #matrizdeconfusion = calcularMatrixDeConfusion(label_validation,
            #                                              out_red,
            #                                              titulo)
            #matriz_confusion_sumatoria = matriz_confusion_sumatoria + matrizdeconfusion

            print('\n Measuring output performance ...')
            acuracia = accuracy_score(label_validation, out_red)
            vector_acuracia.append(acuracia)

        # ----------------- CALCULA ESTADISTICAS -----------------------
        [acuracia_media, acuracia_desvio_padrao] = calculoDesvioPadrao(vector_acuracia)
        [time_train_media, time_train_desvio_padrao] = calculoDesvioPadrao(vector_time_train)
        [time_test_media, time_test_desvio_padrao] = calculoDesvioPadrao(vector_time_test)
        #matriz_confusion_media = matriz_confusion_sumatoria / numero_experimentos

        # ----------------- GUARDA VECTORES DE ESTADISTICAS -----------------------
        vector_acuracia_media.append(acuracia_media)
        vector_acuracia_desvio_padrao.append(acuracia_desvio_padrao)

        vector_time_train_media.append(time_train_media)
        vector_time_train_desvio_padrao.append(time_train_desvio_padrao)

        vector_time_test_media.append(time_test_media)
        vector_time_test_desvio_padrao.append(time_test_desvio_padrao)

        # np.savez( path_result+"metricas.npz",
        #          matriz_confusao = vector_matriz_confusion)

        # ----------------- IMPRIME MATRIZ DE CONFUSION MEDIA -----------------------
        #titulo_mc = "** MATRIZ DE CONFUSÃO MÉDIA ** \n Address Size " + str(address_size[j])
        #df_cm = pd.DataFrame(matriz_confusion_media, index=range(0, numero_de_grupos),
        #                     columns=range(0, numero_de_grupos))
        #path_confusion_matriz = path_result + 'confusionMatrix/' + titulo_mc + ".png"
        #if plot_confusion_matrix_enable:
        #    pretty.pretty_plot_confusion_matrix(df_cm, cmap='Blues', title=titulo_mc,
        #                                        nombreFigura=path_confusion_matriz)

    # ----------------- GUARDA EM CSV VECTORES DE ESTADISTICAS  -----------------------
    print('\n Saving results files ...')

    with open(path_result + '/accuracy/'+antenna_config+'/'+type_of_input+'/'+user+'/LOS_NLOS_detection/acuracia_' + figure_name + '.csv', 'w') as f:
        writer_acuracy = csv.writer(f, delimiter='\t')
        writer_acuracy.writerows(zip(address_size, vector_acuracia_media, vector_acuracia_desvio_padrao))

    with open(path_result + '/processingTime/'+antenna_config+'/'+type_of_input + '/' + user + '/LOS_NLOS_detection/time_train_' + figure_name + '.csv', 'w') as f:
        writer_time_train = csv.writer(f, delimiter='\t')
        writer_time_train.writerows(zip(address_size, vector_acuracia_media, vector_time_train_desvio_padrao))

    with open(path_result + '/processingTime/'+antenna_config+'/'+type_of_input + '/' + user +'/LOS_NLOS_detection/time_test_' + figure_name + '.csv', 'w') as f:
        writer_time_test = csv.writer(f, delimiter='\t')
        writer_time_test.writerows(zip(address_size, vector_time_test_media, vector_time_test_desvio_padrao))

    # ----------------- PLOT DE RESULTADOS  ------------------------------
    titulo = titulo_figura
    nombre_curva = "Dado com desvio padrão"

    plotarResultados(address_size,
                     vector_acuracia_media,
                     vector_acuracia_desvio_padrao,
                     titulo,
                     nombre_curva,
                     "Tamanho da memória",
                     "Acuracia Média (%)",
                     ruta=path_result + '/accuracy/'+antenna_config+'/'+type_of_input + '/' + user +'/LOS_NLOS_detection/acuracia_'+figure_name+'.png')

    plotarResultados(address_size,
                     vector_time_train_media,
                     vector_time_train_desvio_padrao,
                     titulo,
                     nombre_curva,
                     "Tamanho da memória",
                     "Tempo de treinamento Médio (s)",
                     ruta=path_result + '/processingTime/'+antenna_config+'/'+type_of_input + '/' + user +'/LOS_NLOS_detection/time_train_'+figure_name+'.png''')

    plotarResultados(address_size,
                     vector_time_test_media,
                     vector_time_test_desvio_padrao,
                     titulo,
                     nombre_curva,
                     "Tamanho da memória",
                     "Tempo de Teste Médio (s)",
                     ruta=path_result + '/processingTime/'+antenna_config+'/'+type_of_input + '/' + user +'/LOS_NLOS_detection/time_test_'+figure_name+'.png')

    return out_red#, df_cm


def LOS_NLOS_SVC_classification(X_train, y_train, X_test, y_test):

    clf = svm.SVC(gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    Acuracia = metrics.accuracy_score(y_test, y_pred)

    print("--- PARAMETROS DO MODELO SVC---")
    model_parameters = clf.get_params(deep=True)
    gamma = model_parameters.get('gamma')
    c = model_parameters.get('C')

    print("Gamma: ", gamma)
    if gamma == 'scale':
        print("[o valor de Gamma é] 1 / (n_features * X.var())]")
    elif gamma == 'auto':
        print("[o valor de Gamma é ]1 / n_features]")

    print("C: ", c)

    print("Acurácia: ", Acuracia)

    return Acuracia

def classificador_KNeighbors():

def read_labels_as_LOS_NLOS():
    all_info_coord_val, coord_train, coord_test = obj_pre_preprocessing.read_valid_coordinates()
    label_train_in_str = coord_train[:, 4]
    label_test_in_str = coord_test[:, 4]

    labels_train_in_binary = []
    for i in range(len(label_train_in_str)):
        if label_train_in_str[i] == 'LOS=0':
            labels_train_in_binary.append(0)
        else:
            labels_train_in_binary.append(1)

    labels_test_in_binary = []
    for i in range(len(label_test_in_str)):
        if label_test_in_str[i] == 'LOS=0':
            labels_test_in_binary.append(0)
        else:
            labels_test_in_binary.append(1)

    return label_train_in_str, label_test_in_str, labels_train_in_binary, labels_test_in_binary

def relation_coord_with_connection_type(all_data, set):
    data = pd.DataFrame(all_data, columns=['EpisodeID', 'x', 'y', 'z', 'LOS_str', 'LOS_binary'])
    #data = pd.DataFrame(all_data, columns=['EpisodeID', 'x', 'y', 'z', 'LOS', 'rxBeams', 'txBeams', 'combinedBeams'])
    sns.set(style='darkgrid')
    a=2
    if a==1:
        plot = sns.relplot(data=data,
                       x='x',
                       y='y',
                       kind='scatter',
                       hue='LOS_binary',
                       palette='dark',
                       style='LOS_binary',
                       col='LOS_binary',
                       size='LOS_binary',
                       legend=False)
        plot.fig.suptitle(
        'Relacao do tipo de conexao com a posicao do Rx \n no dataset de ['+set+']',
        #'Dist. dos ind. dos Beams do ' + user_type + ' [' + connection + '] em config ' + ' [' + config + ']\n relativo à posição usando dados ' + ' [' + set + ']',
        fontweight='bold')
        plot.fig.subplots_adjust(top=0.825, left=0.048)

        plot.fig.set_figwidth(15)
        plot.fig.set_figheight(6)
        #plt.savefig(name, transparent=False, dpi=300)

    elif a==2:
        plot = sns.relplot(data=data,
                           x='x',
                           y='y',
                           kind='scatter',
                           hue='LOS_binary',
                           size='LOS_binary',
                           alpha=0.5,
                           palette='deep')
        sns.set(rc={'figure.figsize': (60, 5)})

        plot.fig.suptitle(
            'Relação do tipo de conexao com a posicao do Rx \n no dataset de ['+set+']',
            fontweight='bold')
        plot.fig.subplots_adjust(top=0.90)
        plot.fig.set_figwidth(6)
        plot.fig.set_figheight(8)
        plt.show()

        #plot._legend.text[0].set_text("") #(labels=['LOS', 'NLOS'])


    elif a==3:
        sns.scatterplot(data=data, x='x', y='y', hue='LOS_binary', size='LOS_binary')
        plt.legend(labels=['LOS', 'NLOS'], title='Tipos de conexao')
        plt.show()

def plot_histogram_LOS_NLOS_connection(data, title):

    plt.rcParams['figure.figsize'] = [6, 4]
    plt.hist(data, bins=4)
    plt.title(title)
    plt.xticks((0, 1))
    plt.legend(loc='best',
               title='Amostras',
               labels=[str(len(data))])

    plt.show()
def analyses_data():

    label_train_in_str, label_test_in_str, labels_train_in_binary, labels_test_in_binary = read_labels_as_LOS_NLOS()
    all_info_coord_val, all_info_coord_train, all_info_coord_test = obj_pre_preprocessing.read_valid_coordinates()
    coord_train = obj_pre_preprocessing.read_only_coord(all_info_coord_train)
    coord_test = obj_pre_preprocessing.read_only_coord(all_info_coord_test)

    data_train = np.column_stack((all_info_coord_train, labels_train_in_binary))
    relation_coord_with_connection_type(all_data=data_train, set='Train')


    plot_histogram_LOS_NLOS_connection(data=labels_train_in_binary,
                                       title="Distribuição das conexões LOS/NLOS dos dados de [TRAIN] \n LOS=1 NLOS=0")
    plot_histogram_LOS_NLOS_connection(data=labels_test_in_binary,
                                       title="Distribuição das conexões LOS/NLOS dos dados de [TEST] \n LOS=1 NLOS=0")

    return labels_train_in_binary, labels_test_in_binary, coord_train, coord_test



def do_LOS_NLOS_detection():

    all_lidar_train, all_lidar_test = obj_read_lidar.read_all_LiDAR_data()
    label_train, label_test, coord_train, coord_test = analyses_data()
    #label_train, label_test = read_labels_as_LOS_NLOS()

    svc_accuracy = LOS_NLOS_SVC_classification(X_train=coord_train, y_train=label_train, X_test=coord_test, y_test=label_test)

    LOS_NLOS_WISARD_classification(input_train=all_lidar_train,
                            input_validation=all_lidar_test,
                            label_train = label_train,
                            label_validation = label_test,
                            figure_name = 'LOS_NLOS_detection',
                            antenna_config = '8X32',
                            type_of_input = 'lidar',
                            titulo_figura = 'Deteção LOS/NLOS',
                            user = 'All',
                            enableDebug=False,
                            )





    a=0
