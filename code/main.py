
import preprocesamento as preprocess

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌃' to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    coord = preprocess.read_all_coordinates()
    #preprocess.position_just_rx(coord)
    train, test, coordenadas_LOS_train, coordenadas_NLOS_train, coordenadas_LOS_test, coordenadas_NLOS_test = preprocess.coord_to_Qs_matrix(coord)


