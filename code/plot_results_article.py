import numpy as np
import seaborn as sns
import pandas as pd
import csv
import matplotlib.pyplot as plt

def read_results_data_for_coordinates():
    path = "../results/accuracy/8X32/coord_in_Thermomether_x_y_unbalanced/Combined/"
    filename = "_acuracia_beam_selection_Combined_[8X32]_coord_in_Thermomether_x_y_unbalanced_8_ALL.csv"
    file = path + filename
    print(file)

    usecols = ["size_memory", "accuracy", "std_deviation"]
    # Read the CSV file
    data = pd.read_csv(file, sep="\t", header=None, usecols=usecols,
                             names=["size_memory", "accuracy", "std_deviation"])


    size_memory = data['size_memory'].tolist()
    accuracy = data['accuracy'].tolist()
    std = data['std_deviation'].tolist()
    std = [0.0042433, 0.0053565, 0.000484, 0.003243, 0.0024847, 0.000500156, 0.00098417, 0.001468,0.00098417,0.0000045363, 0.00179088, 0.0010487]
    filename = '../results/accuracy/article/coordinates_accuracy_vs_size_of_memory.png'

    #plot_results(size_memory, accuracy, filename)
    plot_results_with_std_desviation(size_memory, accuracy, std, filename)

def read_results_data_for_lidar():
    path = "../results/accuracy/8X32/rx_cubo_+_Lidar/Combined/"
    filename = "acuracia_beam_selection_Combined_[8X32]_rx_cubo_+_Lidar_ALL.csv"
    file = path + filename
    print(file)

    usecols = ["size_memory", "accuracy", "std_deviation"]
    # Read the CSV file
    data = pd.read_csv(file, sep="\t", header=None, usecols=usecols,
                             names=["size_memory", "accuracy", "std_deviation"])
    size_memory = data ['size_memory'].tolist()
    accuracy = data ['accuracy'].tolist()
    std_desviation = data ['std_deviation'].tolist()
    filename = '../results/accuracy/article/lidar_accuracy_vs_size_of_memory.png'

    plot_results_with_std_desviation(size_memory, accuracy, std_desviation, filename)

    a=0

def read_results_data_for_lidar_e_coord():
    path = "../results/accuracy/8X32/coord_in_Thermomether_x_y_unbalanced_+_rx_cubo_+_Lidar/Combined/"
    filename = "acuracia_beam_selection_Combined_[8X32]_coord_in_Thermomether_x_y_unbalanced_+_rx_cubo_+_Lidar_ALL.csv"
    file = path + filename
    print (file)

    usecols = ["size_memory", "accuracy", "std_deviation"]
    # Read the CSV file
    data = pd.read_csv (file, sep="\t", header=None, usecols=usecols,
                        names=["size_memory", "accuracy", "std_deviation"])
    size_memory = data ['size_memory'].tolist ()
    accuracy = data ['accuracy'].tolist ()
    std_desviation = data ['std_deviation'].tolist ()
    filename = '../results/accuracy/article/lidar_and_coord_accuracy_vs_size_of_memory.png'
    plot_results_with_std_desviation(size_memory, accuracy, std_desviation, filename)
def read_results_top_k_coordinates():
    path = '../results/accuracy/article/data_top-k/coord/ALL/'
    filename_decision_tree = 'acuracia_decision_tree_coord_ALL_top_k.csv'
    filename_knn = 'acuracia_KNN_coord_ALL_top_k.csv'
    filename_random_forest = 'acuracia_random_forest_coord_ALL_top_k.csv'
    filename_wisard = 'acuracia_wisard_coord_in_termometro_ALL_top_k.csv'
    filename_svc = 'acuracia_SVC_coord_ALL_top_k.csv'

    usecols = ["id", "accuracy"]
    top_k = [1, 5, 10, 20, 30, 40, 50]

    file = path + filename_decision_tree
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_decision_tree = data['accuracy'].tolist()
    accuracy_decision_tree = accuracy_decision_tree[1:]

    file = path + filename_knn
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_knn = data['accuracy'].tolist()
    accuracy_knn = accuracy_knn[1:]

    file = path + filename_random_forest
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_random_forest = data['accuracy'].tolist()
    accuracy_random_forest = accuracy_random_forest[1:]

    file = path + filename_wisard
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_wisard = data['accuracy'].tolist()
    accuracy_wisard = accuracy_wisard[1:]

    file = path + filename_svc
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_svc = data['accuracy'].tolist()
    accuracy_svc = accuracy_svc[1:]

    filename = '../results/accuracy/article/coordinates_accuracy_top_k.png'
    plot_results_top_k(top_k, accuracy_decision_tree, accuracy_wisard, accuracy_knn, accuracy_random_forest, accuracy_svc, data_coord=True, filename=filename)

def read_results_top_k_lidar():
    path = '../results/accuracy/article/data_top-k/LiDAR/ALL/'
    filename_decision_tree = 'acuracia_decision_tree_LiDAR_ALL_top_k.csv'
    filename_knn = 'acuracia_KNN_LiDAR_ALL_top_k.csv'
    filename_random_forest = 'acuracia_random_forest_LiDAR_ALL_top_k.csv'
    filename_wisard = 'acuracia_wisard_rx_cubo_+_Lidar_all_top_k.csv'

    usecols = ["id","accuracy"]
    top_k = [1, 5, 10, 20, 30, 40, 50]

    file = path + filename_decision_tree
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_decision_tree = data['accuracy'].tolist()
    accuracy_decision_tree = accuracy_decision_tree[1:]

    file = path + filename_knn
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_knn = data['accuracy'].tolist()
    accuracy_knn = accuracy_knn[1:]

    file = path + filename_random_forest
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_random_forest = data['accuracy'].tolist()
    accuracy_random_forest = accuracy_random_forest[1:]

    file = path + filename_wisard
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_wisard = data['accuracy'].tolist()
    accuracy_wisard = accuracy_wisard[1:]

    filename = '../results/accuracy/article/lidar_accuracy_top_k.png'
    accuracy_svc = [0,0,0,0,0,0,0]
    plot_results_top_k(top_k, accuracy_decision_tree, accuracy_wisard, accuracy_knn, accuracy_random_forest, accuracy_svc, data_coord=False, filename=filename)

def read_results_top_k_lidar_e_coord():
    path = '../results/accuracy/article/data_top-k/LiDAR/ALL/'
    filename_decision_tree = 'acuracia_decision_tree_LiDAR_ALL_top_k.csv'
    filename_knn = 'acuracia_KNN_LiDAR_ALL_top_k.csv'
    filename_random_forest = 'acuracia_random_forest_LiDAR_ALL_top_k.csv'
    filename_wisard = '../results/accuracy/article/data_top_k_lidar_coord/acuracia_wisard_coord_in_Thermomether_x_y_unbalanced_+_rx_cubo_+_Lidar_all_top_k.csv'
    filename_wisard_gps = '../results/accuracy/article/data_top-k/coord/ALL/acuracia_wisard_coord_in_termometro_ALL_top_k.csv'
                           #'coord/ALL/acuracia_wisard_coord_in_termometro_ALL_top_k.csv')
    filename_wisard_lidar ='../results/accuracy/article/data_top-k/LiDAR/ALL/acuracia_wisard_rx_cubo_+_Lidar_all_top_k.csv'
                            #'data_top_k/LiDAR/ALL/acuracia_wisard_rx_cubo_+_Lidar_all_top_k.csv')

    usecols = ["id","accuracy"]
    top_k = [1, 5, 10, 20, 30, 40, 50]

    file = path + filename_decision_tree
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_decision_tree = data['accuracy'].tolist()
    accuracy_decision_tree = accuracy_decision_tree[1:]

    file = path + filename_knn
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_knn = data['accuracy'].tolist()
    accuracy_knn = accuracy_knn[1:]

    file = path + filename_random_forest
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_random_forest = data['accuracy'].tolist()
    accuracy_random_forest = accuracy_random_forest[1:]

    file = filename_wisard
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_wisard = data['accuracy'].tolist()
    accuracy_wisard = accuracy_wisard[1:]

    file = filename_wisard_gps
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_wisard_gps = data['accuracy'].tolist()
    accuracy_wisard_gps = accuracy_wisard_gps[1:]

    file = filename_wisard_lidar
    data = pd.read_csv(file,  header=None, usecols=usecols, names=["id","accuracy"])
    accuracy_wisard_lidar = data['accuracy'].tolist()
    accuracy_wisard_lidar = accuracy_wisard_lidar[1:]


    filename = '../results/accuracy/article/lidar_e_coord_accuracy_top_k.png'
    accuracy_svc = [0,0,0,0,0,0,0]
    plot_results_top_k(top_k,
                       accuracy_decision_tree,
                       accuracy_wisard,
                       accuracy_knn,
                       accuracy_random_forest,
                       accuracy_svc,
                       accuracy_wisard_gps,
                       accuracy_wisard_lidar,
                       data_coord=False,
                       filename=filename)

def plot_results_top_k(top_k,
                       accuracy_decision_tree,
                       accuracy_wisard,
                       accuracy_knn,
                       accuracy_random_forest,
                       accuracy_svc,
                       accuracy_wisard_gps,
                       accuracy_wisard_lidar,
                       data_coord,
                       filename):
    style_of_line = 'dashed'#'solid'#
    type_of_marker = 'x'
    size_of_marker = 3
    width_of_line = 0.7
    #sns.set()
    plt.figure()

    plt.plot(top_k, accuracy_decision_tree, color='darkviolet', marker=type_of_marker, linestyle=style_of_line,
             linewidth=width_of_line, markersize=size_of_marker, label='Decision Tree')
    plt.plot(top_k, accuracy_knn, color='seagreen', marker=type_of_marker, linestyle=style_of_line,
             linewidth=width_of_line, markersize=size_of_marker, label='KNN')
    plt.plot(top_k, accuracy_random_forest, color='darkgoldenrod', marker=type_of_marker, linestyle=style_of_line,
             linewidth=width_of_line, markersize=size_of_marker, label='Random Forest')
    plt.plot(top_k, accuracy_wisard_gps, color= 'steelblue', marker=type_of_marker, linestyle='dashed',
             linewidth=width_of_line, markersize=size_of_marker, label='WiSARD (GPS)')
    plt.plot(top_k, accuracy_wisard_lidar, color='darkorange', marker=type_of_marker, linestyle='dashed',
             linewidth=width_of_line, markersize=size_of_marker, label='WiSARD (LiDAR)')
    plt.plot (top_k, accuracy_wisard, color='red', marker=type_of_marker, linestyle='solid',
              linewidth=1, markersize=size_of_marker, label='WiSARD (GPS + LiDAR')

    if data_coord:
        plt.plot(top_k, accuracy_svc, color='darkorange', marker=type_of_marker, linestyle=style_of_line, linewidth=width_of_line, markersize=size_of_marker, label='SVC')

    plt.xticks(top_k)
    plt.xlabel('Top-k', color='steelblue', size=14, fontweight='bold')
    plt.ylabel('Accuracy', color='steelblue', size=14, fontweight='bold')
    plt.legend ()
    plt.grid()
    plt.savefig (filename, dpi=300, bbox_inches='tight')
    plt.show ()





def plot_results_with_std_desviation(x, y, std_desviation, filename):


    subtracted = [element1 - element2 for (element1, element2) in zip (y, std_desviation)]
    add = [element1 + element2 for (element1, element2) in zip (y, std_desviation)]
    sns.set()

    plt.figure()
    plt.plot(x, y, color='steelblue', marker='o', linestyle='-', linewidth=0.5, markersize=6)
    plt.fill_between(x, subtracted, add, color='steelblue', alpha=0.15)

    #plt.plot(x, y, color='steelblue', markersize=6)
    #plt.errorbar(x, y, yerr=std_desviation,  color='steelblue', linestyle='None',marker='^', markersize=3)#, alpha=0.2), marker='o', ,
    plt.xticks(x)
    plt.xlabel('Address size', color='steelblue', size=14, fontweight='bold')
    plt.ylabel('Accuracy', color='steelblue', size=14, fontweight='bold')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    a=0

def plot_results(x, y, filename):
    sns.set()

    plt.figure()
    plt.plot(x, y, color='steelblue', marker='o', linestyle='-', linewidth=0.5, markersize=6)
    plt.xticks(x)
    plt.xlabel('Address size', color='steelblue', size=14, fontweight='bold')
    plt.ylabel('Accuracy', color='steelblue', size=14, fontweight='bold')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    a=0


read_results_top_k_lidar_e_coord()
read_results_data_for_lidar_e_coord()
read_results_data_for_coordinates()
read_results_top_k_lidar_e_coord()
read_results_top_k_coordinates()
read_results_top_k_lidar()
read_results_data_for_lidar()