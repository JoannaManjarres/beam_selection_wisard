import numpy as np
import csv
import wisardpkg as wp
import timeit
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


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

    wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)

    print('\n Training WISARD net ...')
    tic()
    # train using the input data
    wsd.train(data_train, label_train)

    tiempo_entrenamiento_ms = toc()

    print('\n Selecting Beams using WISARD network ...')
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

def select_best_beam(input_train,
                     input_validation,
                     label_train,
                     label_validation,
                     titulo_resultados,
                     enableDebug=False,
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

    path_result = "../results/"

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

    with open(path_result + 'accuracy/acuracia.csv', 'w') as f:
        writer_acuracy = csv.writer(f, delimiter='\t')
        writer_acuracy.writerows(zip(address_size, vector_acuracia_media, vector_acuracia_desvio_padrao))

    with open(path_result + 'processingTime/time_train.csv', 'w') as f:
        writer_time_train = csv.writer(f, delimiter='\t')
        writer_time_train.writerows(zip(address_size, vector_acuracia_media, vector_time_train_desvio_padrao))

    with open(path_result + 'processingTime/time_test.csv', 'w') as f:
        writer_time_test = csv.writer(f, delimiter='\t')
        writer_time_test.writerows(zip(address_size, vector_time_test_media, vector_time_test_desvio_padrao))

    # ----------------- PLOT DE RESULTADOS  ------------------------------
    titulo = titulo_resultados
    nombre_curva = "Dado com desvio padrão"

    plotarResultados(address_size,
                     vector_acuracia_media,
                     vector_acuracia_desvio_padrao,
                     titulo,
                     nombre_curva,
                     "Tamanho da memória",
                     "Acuracia Média (%)",
                     ruta=path_result + "/accuracy/acuracia_"+titulo+".png")

    plotarResultados(address_size,
                     vector_time_train_media,
                     vector_time_train_desvio_padrao,
                     titulo,
                     nombre_curva,
                     "Tamanho da memória",
                     "Tempo de treinamento Médio (s)",
                     ruta=path_result + "/processingTime/time_train_"+titulo+".png")

    plotarResultados(address_size,
                     vector_time_test_media,
                     vector_time_test_desvio_padrao,
                     titulo,
                     nombre_curva,
                     "Tamanho da memória",
                     "Tempo de Teste Médio (s)",
                     ruta=path_result + "/processingTime/time_test_"+titulo+".png")

    return out_red#, df_cm