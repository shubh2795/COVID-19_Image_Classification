from project2_image_cnns import *
from project2_image_anns import *


def get_ann_models():
    epochs = 20
    batch = 32
    model = make_image_ann_model()
    train_tfl_image_ann_model(model, train_X, train_Y, test_X, test_Y, epochs, batch)
    print(validate_tfl_image_ann_model(model, valid_X, valid_Y))
    model.save("models/image_ann_model.tfl")


def get_cnn_models():
    epochs = 20
    batch = 32
    model = make_image_cnn_model()
    train_tfl_image_cnn_model(model, train_X, train_Y, test_X, test_Y, epochs, batch)
    print(validate_tfl_image_cnn_model(model, valid_X, valid_Y))
    model.save("models/image_cnn_model.tfl")


#get_ann_models()
#get_cnn_models()
