from load_nets import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pickle


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


NET_PATH = "models/"


# ================== ENSEMBLE'S ===========================

def load_tfl_net_ensemble(folder_path, ensemble_list, loader_funs):
    assert len(ensemble_list) == len(loader_funs)
    loaded_ensemble = []
    for model, loader_fun in zip(ensemble_list, loader_funs):
        tf.reset_default_graph()
        model = loader_fun(folder_path + model)
        loaded_ensemble.append(model)
    return loaded_ensemble


def predict_with_tfl_audio_model(network_model, audio_example):
    prediction = network_model.predict(audio_example.reshape([-1, 4000, 1, 1]))
    return np.argmax(prediction, axis=1)[0]


def predict_with_tfl_ensemble(ensemble, example, predict_fun):
    d = {}
    for net in ensemble:
        prediction = predict_fun(net, example)
        if prediction in d:
            d[prediction] += 1
        else:
            d[prediction] = 1
    dlist = []
    for kv in d.items():
        dlist.append(kv)
    dlist.sort(key=lambda x: x[1], reverse=True)
    return dlist[0][0]


def evaluate_net_ensemble(net_ensemble, Valid_X, Valid_Y):
    correctOutputs = 0
    for (x, y) in zip(Valid_X, Valid_Y):
        votes = np.zeros(3, dtype=int)
        for net in net_ensemble:
            # this if condition checks if this model is decision tree, random forest or SVM
            if (isinstance(net, type(RandomForestClassifier())) or isinstance(net, type(
                    DecisionTreeClassifier()))):
                resultIndex = net.predict(x.reshape(-1, 4096))[0]
            else:
                prediction = net.predict(x.reshape([-1, 64, 64, 1]))
                resultIndex = np.argmax(prediction, axis=1)[0]
            votes[resultIndex] += 1
        if (np.argmax(votes) == np.argmax(y)):
            correctOutputs += 1
    return (correctOutputs / len(Valid_X))


# =============== ENSEMBLE LOADERS =============================

def load_cnn_ensemble():
    tf.reset_default_graph()
    ensemble_list = ["image_cnn_model_001.tfl", "image_cnn_model_002.tfl", "image_cnn_model_003.tfl"]
    loader_funs = [load_cnn_model_001, load_cnn_002, load_cnn_model_003]
    return load_tfl_net_ensemble(NET_PATH, ensemble_list, loader_funs)


def load_ann_ensemble():
    tf.reset_default_graph()
    ensemble_list = ["image_ann_model_001.tfl", "image_ann_model_002.tfl", "image_ann_model_003.tfl"]
    loader_funs = [load_ann_model_001, load_ann_model_002, load_ann_model_003]
    return load_tfl_net_ensemble(NET_PATH, ensemble_list, loader_funs)


def load_ann_cnn_dt_ensemble():
    tf.reset_default_graph()
    ensemble_list = ["image_ann_model_002.tfl", "image_cnn_model_001.tfl", "dtr.pck"]
    loader_funs = [load_ann_model_002, load_cnn_model_001, load_decision_tree]
    return load_tfl_net_ensemble(NET_PATH, ensemble_list, loader_funs)


valid_image_x = load("data/data_pck/valid_images_X.pck")
valid_image_y = load("data/data_pck/valid_images_Y.pck")

ensemble = load_cnn_ensemble()
print("CNN ensemble: ")
print(evaluate_net_ensemble(ensemble, valid_image_x, valid_image_y))

ensemble1 = load_ann_ensemble()
print("ANN ensemble: ")
print(evaluate_net_ensemble(ensemble1, valid_image_x, valid_image_y))

ensemble2 = load_ann_cnn_dt_ensemble()
print("ANN + CNN + Decision Tree Ensemble: ")
print(evaluate_net_ensemble(ensemble1, valid_image_x, valid_image_y))
