import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets.base import load_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from qiskit.aqua.input import ClassificationInput
import numpy as np
import random


data_file = 'data.csv'
data_path = ''
algo_obj = None
feature_dim = 9


def vqc_move(board):
    board = [int(x) if int(x) != -1 else 0 for x in board]

    global data_file, data_path, algo_obj, feature_dim
    to_predict = singleDataItem(data_path, data_file, board, n=feature_dim)


    move = algo_obj.predict(to_predict)[0]

    # if the move selected already contains a play, choose randomly
    if board[move]:
        spaces = [index for index, x in enumerate(board) if x == 0]
        move = random.choice(spaces)

    return int(move)


def setup():
    sample_Total, training_input, test_input, class_labels = userDefinedData(data_path,
                                                                                        data_file,
                                                                                        ['0', '1', '2', '3', '4', '5', '6',
                                                                              '7', '8'],
                                                                                        training_size=6000, test_size=500,
                                                                                        n=feature_dim, PLOT_DATA=False)

    temp = [test_input[k] for k in test_input]
    total_array = np.concatenate(temp)

    aqua_dict = {
        'problem': {'name': 'classification'},
        'algorithm': {
            'name': 'SVM'
        },
        'multiclass_extension': {'name': 'AllPairs'}
    }

    algo_input = ClassificationInput(training_input, test_input, total_array)

    from qiskit.aqua import QiskitAqua
    aqua_obj = QiskitAqua(aqua_dict, algo_input)
    global algo_obj
    algo_obj = aqua_obj.quantum_algorithm

    print("Training the SVM....")
    aqua_obj.run()
    print("Trained!")

def userDefinedData(location, file, class_labels, training_size, test_size, n=2, PLOT_DATA=True):

    data, target, target_names = load_data(location, file)
    # sample_train is of the same form as data
    sample_train, sample_test, label_train, label_test = train_test_split(
        data, target, test_size=0.25, train_size=0.75, random_state=22)

    # Now we standarize for gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Now reduce number of features to number of qubits
    pca = PCA(n_components=n).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)

    # Samples are pairs of points
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)

    # If class labels are numeric
    if class_labels[0].isdigit():
        # Pick training size number of samples from each distro
        training_input = {key: (sample_train[label_train == int(key), :])[:training_size] for k, key in
                          enumerate(class_labels)}
        test_input = {key: (sample_test[label_test == int(key), :])[: test_size] for k, key in
                      enumerate(class_labels)}

    else:
        # if they aren't
        training_input = {key: (sample_train[label_train == k, :])[:training_size] for k, key in
                          enumerate(class_labels)}
        test_input = {key: (sample_train[label_train == k, :])[training_size:(
                training_size + test_size)] for k, key in enumerate(class_labels)}

    if PLOT_DATA:
        for k in range(0, 9):
            plt.scatter(sample_train[label_train == k, 0][:training_size],
                        sample_train[label_train == k, 1][:training_size])

        plt.title("PCA dim. reduced user dataset")
        plt.show()

    return sample_train, training_input, test_input, class_labels


def singleDataItem(location, file, data, n=2):

    # load the old data so that this new example gets transformed correctly
    test_data, target, target_names = load_data(location, file)

    # Only use sample_train to help get the right shape for the new input
    sample_train, sample_test, label_train, label_test = train_test_split(
        test_data, target, test_size=0.25, train_size=0.75, random_state=22)

    # convert to np array to enable joining to other data
    data = np.array(data)
    # add the to be tested data onto the end of the overall data
    sample_train = np.vstack((sample_train, data))

    # Now we standarize for gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)

    # Now reduce number of features to number of qubits
    pca = PCA(n_components=n).fit(sample_train)
    sample_train = pca.transform(sample_train)

    # Scale to the range (-1,+1)
    minmax_scale = MinMaxScaler((-1, 1)).fit(sample_train)
    sample_train = minmax_scale.transform(sample_train)

    # returns array which contains one array with is the 2 data points to use
    # OUTPUT IS READY TO BE PUT INTO SVM.PREDICT
    return [sample_train[-1]]
