import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import csv
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data, color


def process_data(enable_plot, num_scene_to_plot):

    data_path = "../data/lidar/lidar_train_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = read_data(data_path)
    data_lidar_train = pre_process_lidar(data_lidar_process_all, data_position_rx)

    saveInputPath = "../data/lidar/ALL/"
    np.savez(saveInputPath + 'all_data_lidar_train' + '.npz', lidar_train=data_lidar_train)

    if enable_plot:
        print_scene(data_lidar_process_all[num_scene_to_plot], data_position_rx[num_scene_to_plot], data_position_tx[num_scene_to_plot])

    data_path = "../data/lidar/lidar_validation_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = read_data(data_path)
    data_lidar_validation = pre_process_lidar(data_lidar_process_all, data_position_rx)

    saveInputPath = "../data/lidar/ALL/"
    np.savez(saveInputPath + 'all_data_lidar_test' + '.npz', lidar_test=data_lidar_validation)

    return data_lidar_train, data_lidar_validation

def process_data_rx_like_cube():
    data_path = "../data/lidar/lidar_train_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = read_data(data_path)

    #a = []
    #for i, v in enumerate(data_position_rx[0, 8, :, :]):
    #    a.extend(v)



    b = 0

    x_dimension = len(data_position_rx[0, :, 0, 0])
    y_dimension = len(data_position_rx[0, 0, :, 0])
    z_dimension = len(data_position_rx[0, 0, 0, :])
    dimension_of_coordenadas = x_dimension * y_dimension * z_dimension
    number_of_samples = data_position_rx.shape[0]
    lidar_data_vector = np.zeros([number_of_samples, dimension_of_coordenadas])
    position_of_rx_vector = np.zeros([number_of_samples, dimension_of_coordenadas])

    data = data_position_rx.copy()
    data_1 = data*0

    amostra, x_rx, y_rx, z_rx = np.unravel_index(data_position_rx.argmax(), data_position_rx.shape)
    data_1[:, 0:x_rx, 0:y_rx, 0:z_rx]=1


    '''
    for amostra in range(1):
        for plano in range(x_dimension):
            for i, v in enumerate(data_position_rx[amostra,plano,:,:]):
                    if v.any() == 1:  #v é um vetor de 10 posicoes
                        for j, val in enumerate(v):
                            if v[j] == 1:
                                print('Amostra',amostra,'y=',i,' z=', j, val)
                                for a in range(plano+1):
                                    if a <= plano:
                                        for k in range(y_dimension):
                                            if k <= i:
                                                for t in range(z_dimension):
                                                    if t <= j:
                                                        data_1[amostra,a,k,t] = 1
                                                    else:
                                                        break
                                            else:
                                                break
                                    else:
                                        break
                                        
    '''
    nc = data_1[amostra,:,:,:]
    scenario_complet = data_lidar_process_all[0,:,:,:]
    #one = np.ones((nc + 1, nc + 1, nc + 1))
    fig = plt.figure()

    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.voxels(nc, alpha=0.12, edgecolor=None, shade=True)  # Voxel visualization
    ax.set_title('Receptor')
    ax.set_xlabel('x', labelpad=10)
    ax.set_ylabel('y', labelpad=10)
    ax.set_zlabel('z', labelpad=10)
    plt.tight_layout()

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.voxels(scenario_complet, alpha=0.12, edgecolor=None, shade=True, antialiased = False)  # Voxel visualization
    ax.set_title('Cenario completo')
    ax.set_xlabel('x', labelpad=10)
    ax.set_ylabel('y', labelpad=10)
    ax.set_zlabel('z', labelpad=10)
    plt.tight_layout()

    m=0






def process_data_without_rx(enable_plot, num_scene_to_plot):

    data_path = "../data/lidar/lidar_train_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = read_data(data_path)
    data_lidar_train = pre_process_lidar_without_rx_transposto(data_lidar_process_all)
    #data_lidar_train = pre_process_lidar_without_rx(data_lidar_process_all)

    saveInputPath = "../data/lidar/sem_rx/"
    np.savez(saveInputPath + 'all_data_lidar_sem_rx_train' + '.npz', lidar_train=data_lidar_train)

    if enable_plot:
        print_scene(data_lidar_process_all[num_scene_to_plot], data_position_rx[num_scene_to_plot], data_position_tx[num_scene_to_plot])

    data_path = "../data/lidar/lidar_validation_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = read_data(data_path)
    data_lidar_validation = pre_process_lidar_without_rx(data_lidar_process_all)

    saveInputPath = "../data/lidar/sem_rx/"
    np.savez(saveInputPath + 'all_data_lidar_sem_rx_test' + '.npz', lidar_test=data_lidar_validation)

    return data_lidar_train, data_lidar_validation

def read_data(data_path):
    label_cache_file = np.load(data_path)
    data_lidar = label_cache_file['input']

    data_lidar_process = np.where(data_lidar == -1, 0, data_lidar)
    data_lidar_process_all = np.where(data_lidar_process == -2, 1, data_lidar_process)

    data_position_rx = np.where(data_lidar == -2, 1, 0)
    data_position_tx = np.where(data_lidar == -1, 1, 0)

    return data_lidar_process_all, data_position_rx, data_position_tx

def pre_process_lidar(data_lidar_process, data_position_rx):

    x_dimension = len(data_lidar_process[0,:,0,0])
    y_dimension = len(data_lidar_process[0,0,:,0])
    z_dimension = len(data_lidar_process[0,0,0,:])
    dimension_of_coordenadas = x_dimension * y_dimension * z_dimension
    number_of_samples = data_lidar_process.shape[0]
    lidar_data_vector = np.zeros([number_of_samples, dimension_of_coordenadas])
    position_of_rx_vector = np.zeros([number_of_samples, dimension_of_coordenadas])

    #lidar_data_vector = np.zeros([dimension_of_coordenadas])
    #position_of_rx_vector = np.zeros([dimension_of_coordenadas])
    all_data = np.zeros([number_of_samples, dimension_of_coordenadas*2], dtype=np.int8)
    a = np.zeros(dimension_of_coordenadas,dtype=np.int8)
    b = np.zeros(dimension_of_coordenadas,dtype=np.int8)

    for i in range(number_of_samples):
        #lidar_data_vector[i, :] = data_lidar_process[i,:,:,:].reshape(1, dimension_of_coordenadas)
        #position_of_rx_vector[i,:] = data_position_rx[i,:,:,:].reshape(1, dimension_of_coordenadas)

        a = data_lidar_process[i,:,:,:].reshape(1, dimension_of_coordenadas)
        b = data_position_rx[i,:,:,:].reshape(1, dimension_of_coordenadas)
        all_data[i] = np.concatenate((a,b), axis=1)

    return all_data

def pre_process_lidar_without_rx(data_lidar_process):

    x_dimension = len(data_lidar_process[0,:,0,0])
    y_dimension = len(data_lidar_process[0,0,:,0])
    z_dimension = len(data_lidar_process[0,0,0,:])
    dimension_of_coordenadas = x_dimension * y_dimension * z_dimension
    number_of_samples = data_lidar_process.shape[0]
    lidar_data_vector = np.zeros([number_of_samples, dimension_of_coordenadas])
    position_of_rx_vector = np.zeros([number_of_samples, dimension_of_coordenadas])

    all_data = np.zeros([number_of_samples, dimension_of_coordenadas], dtype=np.int8)
    a = np.zeros(dimension_of_coordenadas,dtype=np.int8)


    for i in range(number_of_samples):
        #lidar_data_vector[i, :] = data_lidar_process[i,:,:,:].reshape(1, dimension_of_coordenadas)
        #position_of_rx_vector[i,:] = data_position_rx[i,:,:,:].reshape(1, dimension_of_coordenadas)

        a = data_lidar_process[i,:,:,:].reshape(1, dimension_of_coordenadas)
        #b = data_position_rx[i,:,:,:].reshape(1, dimension_of_coordenadas)
        all_data[i] = a
        #all_data[i] = np.concatenate((a,b), axis=1)

    return all_data

def pre_process_lidar_without_rx_transposto(data_lidar_process):

    x_dimension = len(data_lidar_process[0,:,0,0])
    y_dimension = len(data_lidar_process[0,0,:,0])
    z_dimension = len(data_lidar_process[0,0,0,:])
    dimension_of_coordenadas = x_dimension * y_dimension * z_dimension
    number_of_samples = data_lidar_process.shape[0]
    lidar_data_vector = np.zeros([number_of_samples, dimension_of_coordenadas])
    position_of_rx_vector = np.zeros([number_of_samples, dimension_of_coordenadas])

    all_data = np.zeros([number_of_samples, dimension_of_coordenadas], dtype=np.int8)
    a = np.zeros(dimension_of_coordenadas,dtype=np.int8)


    for i in range(number_of_samples):
        #lidar_data_vector[i, :] = data_lidar_process[i,:,:,:].reshape(1, dimension_of_coordenadas)
        #position_of_rx_vector[i,:] = data_position_rx[i,:,:,:].reshape(1, dimension_of_coordenadas)

        a = data_lidar_process[i,:,:,:].T.reshape(1, dimension_of_coordenadas)
        #b = data_position_rx[i,:,:,:].reshape(1, dimension_of_coordenadas)
        all_data[i] = a
        #all_data[i] = np.concatenate((a,b), axis=1)

    return all_data


def print_scene(objects, rx, tx):
    objects = np.array(objects, dtype=bool)
    rx = np.array(rx, dtype=bool)
    tx = np.array(tx, dtype=bool)

    voxelarray = objects | rx | tx

    # set the colors of each object
    colors = np.empty(voxelarray.shape, dtype=object)



    color_object = '#cccccc90'
    color_rx = 'red'
    color_tx = 'blue'

    colors[objects] = color_object
    colors[rx] = color_rx
    colors[tx] = color_tx

    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')

    # Set axes label
    ax.set_xlabel('x', labelpad=10)
    ax.set_ylabel('y', labelpad=10)
    ax.set_zlabel('z', labelpad=10)

    #set predefine rotation
    #ax.view_init(elev=49, azim=115)

    ax.voxels(voxelarray, facecolors=colors, edgecolor=None, antialiased=False)

    c1 = mpatches.Patch(color=color_object, label='Objects')
    c2 = mpatches.Patch(color=color_rx, label='Rx')
    c3 = mpatches.Patch(color=color_tx, label='Tx')

    ax.legend(handles=[c1, c2, c3], loc='center left', bbox_to_anchor=(-0.1, 0.9))

    plt.show()


######

def read_all_LiDAR_without_rx():
    lidar_path = "../data/Lidar/sem_rx/"

    #np.savez(saveInputPath + 'all_data_lidar_sem_rx_train' + '.npz', lidar_train=data_lidar_train)

    input_cache_file = np.load(lidar_path + "all_data_lidar_sem_rx_train.npz", allow_pickle=True)
    all_lidar_train = input_cache_file["lidar_train"]
    input_cache_file = np.load(lidar_path + "all_data_lidar_sem_rx_test.npz", allow_pickle=True)
    all_lidar_test = input_cache_file["lidar_test"]

    print('Leu os arquivos npz do lidar')

    return all_lidar_train, all_lidar_test

def read_all_LiDAR_data():

    #lidar_path = "../data/lidar/process/"
    lidar_path = "../data/Lidar/ALL/"

    input_cache_file = np.load(lidar_path + "all_data_lidar_train.npz", allow_pickle=True)
    all_lidar_train = input_cache_file["lidar_train"]
    input_cache_file = np.load(lidar_path + "all_data_lidar_test.npz", allow_pickle=True)
    all_lidar_test = input_cache_file["lidar_test"]

    return all_lidar_train,all_lidar_test

def read_LiDAR_LOS_data():

    lidar_path = "../data/lidar/LOS/"

    input_cache_file = np.load(lidar_path + "lidar_train_LOS.npz", allow_pickle=True)
    lidar_train = input_cache_file["lidar_train"]
    input_cache_file = np.load(lidar_path + "lidar_test_LOS.npz", allow_pickle=True)
    lidar_test = input_cache_file["lidar_test"]

    return lidar_train, lidar_test

def read_LiDAR_NLOS_data():
    lidar_path = "../data/lidar/NLOS/"

    input_cache_file = np.load(lidar_path + "lidar_NLOS_train.npz", allow_pickle=True)
    lidar_train = input_cache_file["lidar_train"]
    input_cache_file = np.load(lidar_path + "lidar_NLOS_test.npz", allow_pickle=True)
    lidar_test = input_cache_file["lidar_test"]

    return lidar_train, lidar_test

def separed_data_lidar_LOS_NLOS():

    data_lidar_train, data_lidar_test = process_data(enable_plot=False, num_scene_to_plot=0)

    filename = "../data/coordinates/CoordVehiclesRxPerScene_s008.csv"
    limit_ep_train = 1564

    coord = np.zeros([11194, 6], dtype=object)

    lidar_LOS_train = []
    lidar_LOS_test = []

    lidar_NLOS_train = []
    lidar_NLOS_test = []

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)

        cont = 0
        for row in reader:
            if row['Val'] == 'V':
                coord[cont] = int(row['EpisodeID']), row['Val'], float(row['x']), float(row['y']), float(row['z']), row[
                    'LOS']
                cont += 1

    coord_train = coord[(coord[:, 0] < limit_ep_train + 1)]
    coord_test = coord[(coord[:, 0] > limit_ep_train)]

    coord_lidar_train = np.column_stack((coord_train, data_lidar_train))
    coord_lidar_test = np.column_stack((coord_test, data_lidar_test))

    LOS_channel = 5
    cont_LOS_train = 0
    cont_NLOS_train = 0
    cont_LOS_test = 0
    cont_NLOS_test = 0

    for cont_samples in range(len(coord_lidar_train)):
        if coord_lidar_train[cont_samples, LOS_channel] == 'LOS=1':
            lidar_LOS_train.append(coord_lidar_train[cont_samples, 6:80006])
            cont_LOS_train += 1
        else:
            lidar_NLOS_train.append(coord_lidar_train[cont_samples, 6:80006])
            cont_NLOS_train += 1

    for cont_samples in range(len(coord_lidar_test)):
        if coord_lidar_test[cont_samples, LOS_channel] == 'LOS=1':
            lidar_LOS_test.append(coord_lidar_test[cont_samples, 6:80006])
            cont_LOS_test += 1
        else:
            lidar_NLOS_test.append(coord_lidar_test[cont_samples, 6:80006])
            cont_NLOS_test += 1

    saveInputPath = "../data/lidar/LOS/"
    # LOS train
    np.savez(saveInputPath + 'lidar_LOS_train' + '.npz', lidar_train=lidar_LOS_train)
    # LOS test
    np.savez(saveInputPath + 'lidar_LOS_test' + '.npz', lidar_test=lidar_LOS_test)

    saveInputPath = "../data/lidar/NLOS/"
    # NLOS train
    np.savez(saveInputPath + 'lidar_NLOS_train' + '.npz', lidar_train=lidar_NLOS_train)
    # LOS test
    np.savez(saveInputPath + 'lidar_NLOS_test' + '.npz', lidar_test=lidar_NLOS_test)

process_data_rx_like_cube()