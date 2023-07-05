import numpy as np
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import statsmodels.api as sm
import matplotlib.pyplot as plt
import preprocesamento as preprocess
import read_data as read
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import statsmodels.api as sm
import matplotlib.pyplot as plt



def plot_histogram_beam(index_beam, user, color, connection, set, pp_folder, config):

    print("Histograma dos indices dos Beams do ", user ," [" + connection + "] em config ", config , " \n usando dados de ",  set)

    path = pp_folder + '/histogram/'+connection + '/'
    title = 'Distribuicao dos indices dos Beams do ' + user +' [' + connection + '] em config ' + config + ' \n usando dados de ' + set
    sns.set(style='darkgrid')
    sns.set(rc={'figure.figsize': (8, 4)})
    plot = sns.histplot(data=index_beam,
                        bins=15,
                        stat='frequency',
                        color=color,
                        legend=False)
    plt.title(title, fontweight='bold')
    plt.xlabel('Indices no ' + user)
    plt.ylabel('Frequência')
    plt.legend(bbox_to_anchor=(1.05, 1),
               borderaxespad=0,
               loc='upper left',
               title='Amostras',
               labels=[str(index_beam.shape[0])])
    # plot.fig.set_figwidth(4)
    # plot.fig.set_figheight(8)
    plt.subplots_adjust(right=0.786, bottom=0.155)

    name = path + 'Histogram_dos_Beams_' + user + '_' + connection + '_' + set + '.png'
    plt.savefig(name, transparent=False, dpi=300)
    plt.show()

def relation_coord_with_beams_Plot2D(all_data, set, connection, pp_folder, config, user):

    data = pd.DataFrame(all_data, columns=['EpisodeID','x','y','z','LOS','rxBeams','txBeams','combinedBeams'])
    #print(data.head())
    path = pp_folder + '/relation_beams_coord/'+ connection +'/'

    ############################################
    #plt.figure(figsize=(10, 8))
    sns.set(style='darkgrid')

    # Tx position
    x_Tx = [746]
    y_Tx = [560]
    z_Tx = [4]

    # Plota o Tx
    #plt.plot(x_Tx,
    #         y_Tx,
    #         marker=5,
    #         markersize=20,
    #         markeredgecolor="black",
    #         markerfacecolor="red",
    #         label='Tx')


    if user == '1':
        user_type = 'Rx'
        print(
            'plot da Distribuicao dos indices dos Beams do '+user_type+' [' + connection + '] \n em config ' + config + ' relativo à posicao usando dados ' + set)
        plot = sns.relplot(data=data,
                           x='x',
                           y='y',
                           kind='scatter',
                           hue='rxBeams',
                           alpha=0.5,
                           palette='deep')
        sns.set(rc={'figure.figsize': (60, 5)})

        plot.fig.suptitle(
            'Distribuicao dos indices dos Beams do '+user_type+' [' + connection + '] em config ' + config + ' \n relativo à posicao usando dados ' + set,
            fontweight='bold')
        plot.fig.subplots_adjust(top=0.90)
        plot.fig.set_figwidth(6)
        plot.fig.set_figheight(8)
        name = path + 'relation_coord_with_Rx_beams_' + connection + '_' + set + '.png'
        plt.savefig(name, transparent=False, dpi=300)
        plt.show()

    if user == '2':
        user_type = 'Tx'
        print('Plot da Distribuicao dos indices dos Beams do '+user_type+' [', connection, '] \n em config ', config,
              ' relativo à posicao usando dados ', set)
        plot = sns.scatterplot(data=data,
                                x='x',
                                y='y',
                                style='txBeams',
                                palette='deep',
                                hue='txBeams',
                                #style='txBeams',
                                sizes=(20,200),
                                legend='brief'
                                )


        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.title('Distribuicao dos indices dos Beams do '+user_type+' ['+ connection +'] \n em config ' + config + ' relativo à posicao usando dados ' + set)
        plt.subplots_adjust(top=0.94, left = 0.082)
        name = path + 'relation_coord_with_Tx_beams_' + connection+'_' + set + '.png'
        plt.savefig(name, transparent=False, dpi=300)
        plt.show()

    ############################################



    ############################################
    if user =='3':
        #sns.displot(x=x_Tx, y=y_Tx)
        user_type = 'Combinados'
        print(
            'plot da Distribuicao dos indices dos Beams '+user_type+' ['+connection+'] \n em config ' +config+ ' relativo à posicao usando dados ' +set)


        plot = sns.relplot(data=data,
                           x='x',
                           y='y',
                           kind='scatter',
                           hue='combinedBeams',
                           #size='combinedBeams',
                           #sizes=(12,200),
                           #alpha=0.5,
                           palette='dark',
                           legend=False)

        #sns.set(rc={'figure.figsize': (60, 5)})

        plot.fig.suptitle(
            'Distribuicao dos indices dos Beams Combinados ['+connection+'] \n em config ' +config+ ' \n relativo à posicao usando dados ' +set,
            fontweight='bold')
        plot.fig.subplots_adjust(top=0.90)
        plot.fig.set_figwidth(6)
        plot.fig.set_figheight(8)
        name = path + 'relation_coord_with_combined_beams_' +connection+ '_' +set+ '.png'
        plt.savefig(name, transparent=False, dpi=300)
        plt.show()

def relation_coord_with_beams_extend_Plot2D(all_data, set, connection, pp_folder, config, user):
    data = pd.DataFrame(all_data, columns=['EpisodeID', 'x', 'y', 'z', 'LOS', 'rxBeams', 'txBeams', 'combinedBeams'])
    #print(data.head())
    path = pp_folder + 'relation_beam_coord_extend/'+ connection +'/'


    if user == '1':
        user_type = 'Rx'
        print('plot da Dist. dos ind. Extendidos dos Beams do '+user_type+' ['+ connection +'] em config ' + config + '\n relativo à posição usando dados ' + set)
        name = path + 'relation_coord_with_Rx_beams_extend_' + connection + '_' + set + '.png'

        sns.set(style='darkgrid')
        plot = sns.relplot(data=data,
                           x='x',
                           y='y',
                           kind='scatter',
                           hue='rxBeams',
                           palette='dark',
                           style='rxBeams',
                           col='rxBeams',
                           legend=False)
        plot.fig.suptitle('Dist. dos ind. dos Beams do '+user_type+' ['+ connection +'] em config ' +' ['+ config +']\n relativo à posição usando dados ' +' ['+set+']',
                         fontweight='bold')
        plot.fig.subplots_adjust(top=0.825, left=0.048)

        plot.fig.set_figwidth(15)
        plot.fig.set_figheight(6)
        plt.savefig(name, transparent=False, dpi=300)
        #plt.show()


    if user == '2':
        user_type = 'Tx'
        print('plot da Dist. dos ind. Extendidos dos Beams do '+user_type+' ['+ connection +'] em config ' + config + '\n relativo à posição usando dados ' + set)
        name = path + 'relation_coord_with_Tx_beams_extend_'+ connection + '_' + set + '.png'
        plot = sns.relplot(data=data,
                           x='x',
                           y='y',
                           kind='scatter',
                           hue='txBeams',
                           palette='dark',
                           style='txBeams',
                           col='txBeams',
                           legend=False,
                           col_wrap=8)
        #plot.fig.set_size_inches(20, 6)
        plot.fig.set_figheight(6)
        plot.fig.set_figwidth(40)
        plot.fig.suptitle('Dist. dos ind. dos Beams do '+user_type+' ['+ connection +'] em config ' + config + '\n relativo à posição usando dados ' + set,
                          fontweight='bold')
        plot.fig.subplots_adjust(top=0.825, left=0.048)
        plot.fig.set_figwidth(15)
        plot.fig.set_figheight(6)
        plt.savefig(name, transparent=False, dpi=300)
        plt.show()



    # pairplot explore the pairwise relationships between variables.
    #sns.pairplot(data[['x', 'y', 'z', 'rxBeams']], hue='rxBeams', height=3)
    #plt.show()

    if user == '3':
        user_type = 'combinados'
        print('plot da Dist. dos ind. Extendidos dos Beams '+user_type+' [' + connection + '] em config ' + config + '\n relativo à posição usando dados ' + set)
        name = path + 'relation_coord_with_combined_beams_extend_' + connection + '_' + set + '.png'
        plot = sns.relplot(data=data,
                           x='x',
                           y='y',
                           kind='scatter',
                           hue='combinedBeams',
                           palette='dark',
                           style='combinedBeams',
                           col='combinedBeams',
                           legend=False,
                           col_wrap=8)
        # plot.fig.set_size_inches(20, 6)
        plot.fig.set_figheight(6)
        plot.fig.set_figwidth(40)
        plot.fig.suptitle(
            'Dist. dos ind. dos Beams '+user_type+' [' + connection + '] em config ' + config + '\n relativo à posição usando dados ' + set,
            fontweight='bold')
        plot.fig.subplots_adjust(top=0.825, left=0.048)
        plot.fig.set_figwidth(15)
        plot.fig.set_figheight(6)
        plt.savefig(name, transparent=False, dpi=300)
        plt.show()

def plot_distribution_beams_displot(beams_tx, beams_rx, pp_folder, connection, set):

    path = pp_folder + 'histogram/'+connection + '/'
    #plt.Figure(figsize=(32, 8))
    #sns.set_style('darkgrid')
    plot = sns.displot(x=beams_tx,
                y=beams_rx,
                row_order=range(2),
                col_order=range(32),
                binwidth=(2, 2),
                cmap='Blues',
                aspect=2.9, #10.67,
                height=3,#2.5,
                       cbar=True,
                       #cbar_kws={'panchor':(0.5,1.0)}
                )

    plt.title("Distribuicao dos Beams (Tx-Rx) ["+set+"]")
    plt.xlabel("beams_tx_index")
    plt.ylabel("beams_rx_index")
    plt.gca().invert_yaxis()

    plt.subplots_adjust(left=0.08)
    #plt.subplots_adjust(right=5)
    plt.subplots_adjust(bottom=0.179)
    plt.subplots_adjust(top=0.879)
    #plot.fig.set_figwidth(10)
    #plot.fig.set_figheight(6)
    name = path+"Beams_distribution_"+set+".png"
    print(name)
    plt.savefig(name, transparent=False, dpi=300)
    plt.show()

def plot_hist_prob_beam(beam, set, pp_folder, connection, x_label='indice dos beams'):

    path = pp_folder + 'histogram/'+connection + '/'
    print(path)

    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(beam, bins='auto', density=True, color="darkblue", ec="darkblue")
    # ax.plot(data, pdf_lognorm)
    ax.set_ylabel('P', fontsize=12, rotation=0)
    ax.yaxis.set_label_coords(-0.13, 0.5)
    ax.set_xlabel('Índice do par de beams', fontsize=12)
    #ax.xaxis.set_label_coords(1.05, -0.025)
    plt.grid(axis='y', alpha=0.9, color='white')
    ax.set_facecolor('#EEEEF5')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    # plt.tick_params(axis='x', colors='red', direction='out', length=7, width=2)
    title = "Probabilidade do beam ["+set+"]"
    plt.title(title)

    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))

    plt.savefig(path+"histogram_prob_all_Beams_combined_"+set+".png", bbox_inches='tight')
    plt.show()


def beam_analyses(antenna_config, connection, user):

    pp_folder = '../results/analyses/' + antenna_config + '/'

    all_info_coord_val,  coord_val_train,    coord_val_test = preprocess.read_valid_coordinates()
    coord_val_train_LOS, coord_val_test_LOS, coord_val_train_NLOS, coord_val_test_NLOS = preprocess.divide_valid_coord_in_LOS_NLOS_connection(all_info_coord_val)


    if connection =='ALL':

        index_beam_rx_train, index_beam_rx_test, index_beam_tx_train, index_beam_tx_test, index_beam_combined_train, index_beam_combined_test = read.read_all_beams(antenna_config)
        beam_rx_train = np.array([int(i) for i in index_beam_rx_train])
        beam_rx_test = np.array([int(i) for i in index_beam_rx_test])
        beam_tx_train = np.array([int(i) for i in index_beam_tx_train])
        beam_tx_test = np.array([int(i) for i in index_beam_tx_test])
        beam_combined_train = np.array([int(i) for i in index_beam_combined_train])
        beam_combined_test = np.array([int(i) for i in index_beam_combined_test])

        all_info_train = np.column_stack((coord_val_train, index_beam_rx_train, index_beam_tx_train, beam_combined_train))
        all_info_test  = np.column_stack((coord_val_test,  index_beam_rx_test,  index_beam_tx_test, beam_combined_test))

    if connection =='LOS':
        beam_rx_LOS_train, beam_rx_LOS_test, beam_tx_LOS_train, beam_tx_LOS_test, index_beam_combined_LOS_train, index_beam_combined_LOS_test = read.read_LOS_beams(antenna_config)
        beam_rx_train = np.array([int(i) for i in beam_rx_LOS_train])
        beam_rx_test = np.array([int(i) for i in beam_rx_LOS_test])
        beam_tx_train = np.array([int(i) for i in beam_tx_LOS_train])
        beam_tx_test = np.array([int(i) for i in beam_tx_LOS_test])
        beam_combined_train = np.array([int(i) for i in index_beam_combined_LOS_train])
        beam_combined_test = np.array([int(i) for i in index_beam_combined_LOS_test])

        all_info_train = np.column_stack((coord_val_train_LOS, beam_rx_LOS_train, beam_tx_LOS_train, beam_combined_train))
        all_info_test  = np.column_stack((coord_val_test_LOS,  beam_rx_LOS_test,  beam_tx_LOS_test, beam_combined_test))

    if connection =='NLOS':
        beam_rx_NLOS_train, beam_rx_NLOS_test, beam_tx_NLOS_train, beam_tx_NLOS_test, index_beam_combined_NLOS_train, index_beam_combined_NLOS_test= read.read_NLOS_beams(antenna_config)
        beam_rx_train = np.array([int(i) for i in beam_rx_NLOS_train])
        beam_rx_test = np.array([int(i) for i in beam_rx_NLOS_test])
        beam_tx_train = np.array([int(i) for i in beam_tx_NLOS_train])
        beam_tx_test = np.array([int(i) for i in beam_tx_NLOS_test])
        beam_combined_train = np.array([int(i) for i in index_beam_combined_NLOS_train])
        beam_combined_test = np.array([int(i) for i in index_beam_combined_NLOS_test])

        all_info_train = np.column_stack((coord_val_train_NLOS, beam_rx_NLOS_train, beam_tx_NLOS_train, beam_combined_train))
        all_info_test  = np.column_stack((coord_val_test_NLOS, beam_rx_NLOS_test,  beam_tx_NLOS_test, beam_combined_test))

    if user == '1':
        plot_histogram_beam(beam_rx_train,
                            user='rx',
                            color='skyblue',
                            connection=connection,
                            set='train',
                            pp_folder=pp_folder,
                            config=antenna_config)

        plot_histogram_beam(beam_rx_test,
                            user='rx',
                            color='deepskyblue',
                            connection=connection,
                            set='test',
                            pp_folder=pp_folder,
                            config=antenna_config)

    elif user == '2':
        plot_histogram_beam(beam_tx_train,
                            user='tx',
                            color='lightslategray',
                            connection=connection,
                            set='train',
                            pp_folder=pp_folder,
                            config=antenna_config)

        plot_histogram_beam(beam_tx_test,
                            user='tx',
                            color='darkslategrey',
                            connection=connection,
                            set='test',
                            pp_folder=pp_folder,
                            config=antenna_config)

    elif user == '3':
        user='combinados'

        plot_histogram_beam(beam_combined_train,
                            user=user,
                            color='darkblue',
                            connection=connection,
                            set='train',
                            pp_folder=pp_folder,
                            config=antenna_config)

        plot_histogram_beam(beam_combined_test,
                            user=user,
                            color='mediumblue',
                            connection=connection,
                            set='test',
                            pp_folder=pp_folder,
                            config=antenna_config)

        plot_hist_prob_beam(beam_combined_train,
                            set="train",
                            pp_folder=pp_folder,
                            connection=connection)

        plot_hist_prob_beam(beam_combined_test,
                            set="test",
                            pp_folder=pp_folder,
                            connection=connection)

        plot_distribution_beams_displot(beams_tx=beam_tx_train,
                                beams_rx=beam_rx_train,
                                pp_folder=pp_folder,
                                connection=connection,
                                set='train')

        plot_distribution_beams_displot(beams_tx=beam_tx_test,
                                        beams_rx=beam_rx_test,
                                        pp_folder=pp_folder,
                                        connection=connection,
                                        set='test')



    relation_coord_with_beams_Plot2D(all_info_train,
                                     set='train',
                                     connection=connection,
                                     pp_folder=pp_folder,
                                     config=antenna_config,
                                     user=user)

    relation_coord_with_beams_Plot2D(all_info_test,
                                     set='test',
                                     connection=connection,
                                     pp_folder=pp_folder,
                                     config=antenna_config,
                                     user=user)

    relation_coord_with_beams_extend_Plot2D(all_info_train,
                                            set='train',
                                            connection=connection,
                                            pp_folder=pp_folder,
                                            config=antenna_config,
                                            user=user)

    relation_coord_with_beams_extend_Plot2D(all_info_test,
                                            set='test',
                                            connection=connection,
                                            pp_folder=pp_folder,
                                            config=antenna_config,
                                            user=user)







def plot_distribution_beams(beams_tx, beams_rx):
    path = '/Users/Joanna/git/Analise_de_dados/results/analyzes/histogram/'
    plt.Figure(figsize=(32, 8))
    plot = sns.displot(x=beams_tx,
                y=beams_rx,
                row_order=range(8),
                col_order=range(32),
                binwidth=(1, 1),
                aspect=10.67,
                height=3,
                cbar=True)

    plt.title("Distribuicao dos Beams (Tx-Rx)")
    plt.xlabel("beams_tx_index")
    plt.ylabel("beams_rx_index")

    plt.subplots_adjust(left=0.05)
    plt.subplots_adjust(bottom=0.179)
    plt.subplots_adjust(top=0.879)
    #plot.fig.set_figwidth(10)
    #plot.fig.set_figheight(6)
    name = 'Beams_distribution.png'
    plt.savefig(name, transparent=False, dpi=300)
    plt.show()


def plot_variables_correlation(data1, data2, title, x_label, y_label, color):
        # custom = []
        plt.scatter(data1, data2, c=color)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # pyplot.legend('test', 'test1')
        plt.show()

