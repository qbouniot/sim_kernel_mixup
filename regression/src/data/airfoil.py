import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

def get_Airfoil_data_packet(args,path = Path('src/data/Airfoil/')):

    ########### input ###########
    fboj = open(path / 'airfoil_self_noise.dat')

    data = []

    for eachline in fboj:
        t=eachline.strip().split('\t')
        data.append([*map(float, t)])

    data = np.array(data)

    ########### shuffle ############

    shuffle_idx = np.random.permutation(data.shape[0])
    data = data[shuffle_idx]

    ########## x normalization ########

    x_data = data[:,0:5]
    y_data = data[:,5:]

    x_max = np.amax(x_data, axis = 0)
    x_min = np.amin(x_data, axis = 0)
    x_data = (x_data - x_min) / (x_max - x_min)

    ########## split ###########

    if args.use_cv:

        cv = KFold(n_splits=5, shuffle=False)
        data_packet = []
        for train_fold, test_fold in cv.split(x_data, y_data):
            data_fold = {
                'x_train': x_data[train_fold[:1003]],
                'x_valid': x_data[train_fold[1003:]],
                'x_test': x_data[test_fold],
                'y_train': y_data[train_fold[:1003]],
                'y_valid': y_data[train_fold[1003:]],
                'y_test': y_data[test_fold],
            }
            data_packet.append(data_fold)

        return data_packet

    else:

        x_train = x_data[:1003,:]
        x_valid = x_data[1003:1303,:]
        x_test = x_data[1303:1503,:]

        y_train = y_data[:1003,:]
        y_valid = y_data[1003:1303,:]
        y_test = y_data[1303:1503,:]

        data_packet = {
            'x_train': x_train,
            'x_valid': x_valid,
            'x_test': x_test,
            'y_train': y_train,
            'y_valid': y_valid,
            'y_test': y_test,
        }

    return data_packet


if __name__ == '__main__':
    data_packet = get_Airfoil_data_packet()