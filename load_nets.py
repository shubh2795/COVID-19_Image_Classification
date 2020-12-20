import tflearn
from tflearn import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected, dropout
import pickle


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

# ======================= ANNs ===========================

def load_ann_model_001(model_path):
    input_layer = input_data(shape=[None, 64, 64, 1])
    fc_layer_1 = fully_connected(input_layer, 128, activation='relu', name='fc_layer_1')

    fc_layer_2 = fully_connected(fc_layer_1, 32, activation='relu', name='fc_layer_2')

    fc_layer_3 = fully_connected(fc_layer_2, 2, activation='softmax', name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model


def load_ann_model_002(model_path):
    input_layer = input_data(shape=[None, 64, 64, 1])

    fc_layer_1 = fully_connected(input_layer, 2048, activation='relu', name='fc_layer_1')

    fc_layer_2 = fully_connected(fc_layer_1, 512, activation='relu', name='fc_layer_2')

    fc_layer_3 = fully_connected(fc_layer_2, 128, activation='relu', name='fc_layer_3')

    fc_layer_4 = fully_connected(fc_layer_3, 64, activation='relu', name='fc_layer_4')

    fc_layer_5 = fully_connected(fc_layer_4, 16, activation='relu', name='fc_layer_5')

    fc_layer_6 = fully_connected(fc_layer_5, 2, activation='softmax', name='fc_layer_6')
    model = tflearn.DNN(fc_layer_6)
    model.load(model_path)
    return model


def load_ann_model_003(model_path):
    input_layer = input_data(shape=[None, 64, 64, 1])

    fc_layer_1 = fully_connected(input_layer, 64,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model


# ======================= ConvNets ===========================

def load_cnn_model_001(model_path):
    input_layer = input_data(shape=[None, 64, 64, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=64,
                           filter_size=10,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 4, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=16,
                           filter_size=4,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 4, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer_2, 512,
                                 activation='relu',
                                 name='fc_layer_1')

    fc_layer_2 = fully_connected(fc_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 2,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model


def load_cnn_002(model_path):
    input_layer = input_data(shape=[None, 64, 64, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=64,
                           filter_size=10,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 4, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=16,
                           filter_size=4,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 4, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer_2, 512,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_1 = dropout(fc_layer_1, 0.5)
    fc_layer_2 = fully_connected(fc_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_2 = dropout(fc_layer_2, 0.5)
    fc_layer_3 = fully_connected(fc_layer_2, 2,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model


def load_cnn_model_003(model_path):
    input_layer = input_data(shape=[None, 64, 64, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

def load_decision_tree(model_path):
    return load(model_path)

# Summary:
# image_ann_model_001.tfl        validation accuracy:0.8205128205128205
# image_ann_model_002.tfl      validation accuracy:  0.8125
# image_ann_model_003.tfl         validation accuracy: 0.7996794871794872

# image_cnn_model_001.tfl              validation accuracy: 0.8766025641025641
# image_cnn_model_002.tfl              validation accuracy: 0.7932692307692307
# image_cnn_model_003.tfl           validation accuracy: 0.8189102564102564
