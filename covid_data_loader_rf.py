import pickle
import numpy as np

data_path = "data/data_pck/"


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def load_data():
    train_images_X = load(data_path + "train_images_X.pck")
    train_images_Y = load(data_path + "train_images_Y.pck")
    test_images_X = load(data_path + "test_images_X.pck")
    test_images_Y = load(data_path + "test_images_Y.pck")
    valid_image_X = load(data_path + "valid_images_X.pck")
    valid_image_Y = load(data_path + "valid_images_Y.pck")

    training_target = [np.argmax(y) for y in train_images_Y]
    training_data = [train_images_X, training_target]
    test_target = [np.argmax(y) for y in test_images_Y]
    test_data = [test_images_X, test_target]
    valid_target = [np.argmax(y) for y in valid_image_Y]
    validation_data = [valid_image_X, valid_target]
    return training_data, validation_data, test_data


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (4096, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (4096, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (4096, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e
