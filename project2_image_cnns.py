import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

base_path = "data/data_pck/"


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


print('loading datasets from {}...'.format(base_path))
train_X = load(base_path + 'train_images_X.pck')
train_Y = load(base_path + 'train_images_Y.pck')
test_X = load(base_path + 'test_images_X.pck')
test_Y = load(base_path + 'test_images_Y.pck')
valid_X = load(base_path + 'valid_images_X.pck')
valid_Y = load(base_path + 'valid_images_Y.pck')
print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)
print(valid_X.shape)
print(valid_Y.shape)
print('dataset from {} loaded...'.format(base_path))
train_X = train_X.reshape([-1, 64, 64, 1])
test_X = test_X.reshape([-1, 64, 64, 1])

assert train_X.shape[0] == train_Y.shape[0]
assert test_X.shape[0] == test_Y.shape[0]
assert valid_X.shape[0] == valid_Y.shape[0]


def make_image_cnn_model():
    tflearn.init_graph()
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

    d1 = dropout(fc_layer_1, 0.5)

    fc_layer_2 = fully_connected(d1, 128,
                                 activation='relu',
                                 name='fc_layer_2')
    d2 = dropout(fc_layer_2, 0.5)
    fc_layer_3 = fully_connected(d2, 2,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    model = tflearn.DNN(network, tensorboard_verbose=3)
    return model


def load_image_cnn_model(model_path):
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


def test_tfl_image_cnn_model(network_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = network_model.predict(validX[i].reshape([-1, 64, 64, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True))) / float(len(results))


#  train a tfl cnn model on train_X, train_Y, test_X, test_Y.
def train_tfl_image_cnn_model(model, train_X, train_Y, test_X, test_Y, num_epochs=2, batch_size=10):
    tf.reset_default_graph()
    model.fit(train_X, train_Y, n_epoch=num_epochs,
              shuffle=True,
              validation_set=(test_X, test_Y),
              show_metric=True,
              batch_size=batch_size,
              run_id='image_cnn_model')


def validate_tfl_image_cnn_model(model, valid_X, valid_Y):
    return test_tfl_image_cnn_model(model, valid_X, valid_Y)
