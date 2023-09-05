import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv

rota = '../results/accuracy/8X32/coord_in_termometro/Combined/'
file_name = 'acuracia_beam_selection_Combined_[8X32]_coord_in_termometro_2_ALL.csv'
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

rota = '../results/accuracy/8X32/coord_in_termometro/Combined/'
j=1
#all_info = [6, 12, 18,24,28,34,38,44,48,54,58,64]
all_info=[]
for i in range(1,7):
    file_name = 'acuracia_beam_selection_Combined_[8X32]_coord_in_termometro_'+str(j)+'_ALL.csv'

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
    j=j*2
    #all_info = pd.concat([all_info, pd], axis=1)

#print(df_1)
#print(df_2)

print(all_info)

'''
data = np.random.randint(low=1,
                         high=100,
                         size=(10, 10))
'''

y_axis_labels = [1,2,4,8,16,32]
x_axis_labels = [6, 12, 18,24,28,34,38,44,48,54,58,64]
# plotting the heatmap
plt.figure(figsize=(6,4))

annot_kws={'fontsize':10,
           'fontstyle': 'italic',
           'color': "k"}
hm = sns.heatmap(data=all_info,
                 annot=True,
                 cmap="YlGnBu",
                 xticklabels=x_axis_labels,
                 yticklabels=y_axis_labels,
                 )
hm.set_xlabel('Tamanho da Memoria', fontsize=10)
hm.set_ylabel('Resolucao do Termometro', fontsize=10)
# displaying the plotted heatmap
plt.subplots_adjust(right=1, left=0.09)
plt.show()
