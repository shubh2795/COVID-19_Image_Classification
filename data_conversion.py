import csv
from sklearn.model_selection import train_test_split
from PIL import Image
import pickle
import numpy as np

dir_path = "data/"
csv_path = "data/Chest_xray_Corona_Metadata.csv"

train_images_X = []
test_images_X = []
valid_image_X = []
train_images_Y = []
test_images_Y = []
valid_image_Y = []

normal = np.array([1, 0])
pneumonia = np.array([0, 1])


def load_image_get_numpy_array(path):
    original_Image = (Image.open(path)).convert("L")
    image = original_Image.resize((64, 64))
    return np.array(image)


def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)


# restore() function to restore the file
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def _append(img, input_array, output_array, t):
    input_array.append(img)
    if t == "Normal":
        output_array.append(normal)
    elif t == "Pnemonia":
        output_array.append(pneumonia)


def test():
    train_images_X_loaded = load(dir_path + "data_pck/train_images_X.pck")
    train_images_Y_loaded = load(dir_path + "data_pck/train_images_Y.pck")
    test_images_X_loaded = load(dir_path + "data_pck/test_images_X.pck")
    test_images_Y_loaded = load(dir_path + "data_pck/test_images_Y.pck")
    valid_image_X_loaded = load(dir_path + "data_pck/valid_images_X.pck")
    valid_image_Y_loaded = load(dir_path + "data_pck/valid_images_Y.pck")

    cmp = train_images_X == train_images_X_loaded
    print(cmp.all())
    cmp = train_images_Y == train_images_Y_loaded
    print(cmp.all())
    cmp = test_images_X == test_images_X_loaded
    print(cmp.all())
    cmp = test_images_Y == test_images_Y_loaded
    print(cmp.all())
    cmp = valid_image_X == valid_image_X_loaded
    print(cmp.all())
    cmp = valid_image_Y == valid_image_Y_loaded
    print(cmp.all())

    print(train_images_X.shape == train_images_X_loaded.shape)
    print(train_images_Y.shape == train_images_Y_loaded.shape)
    print(test_images_X.shape == test_images_X_loaded.shape)
    print(test_images_Y.shape == test_images_Y_loaded.shape)
    print(valid_image_X.shape == valid_image_X_loaded.shape)
    print(valid_image_Y.shape == valid_image_Y_loaded.shape)


with open(csv_path, 'r') as file:
    next(file)
    reader = csv.reader(file)
    for row in reader:
        # r=random.randint(1,4)
        image = load_image_get_numpy_array(dir_path + "Coronahack-Chest-XRay-Dataset/" + row[3] + "/" + row[1])
        if image.shape != (64, 64):
            print("problem in reshaping")
            print(image.shape)
        if row[3] == "TEST":
            _append(image, valid_image_X, valid_image_Y, row[2])
        elif row[3] == "TRAIN":
            _append(image, train_images_X, train_images_Y, row[2])

train_images_X, test_images_X, train_images_Y, test_images_Y = train_test_split(train_images_X, train_images_Y,
                                                                                random_state=0, test_size=0.25,
                                                                                shuffle=True)
train_images_X = np.array(train_images_X)
test_images_X = np.array(test_images_X)
valid_image_X = np.array(valid_image_X)
train_images_Y = np.array(train_images_Y)
test_images_Y = np.array(test_images_Y)
valid_image_Y = np.array(valid_image_Y)

save(train_images_X, dir_path + "data_pck/train_images_X.pck")
save(train_images_Y, dir_path + "data_pck/train_images_Y.pck")
save(valid_image_X, dir_path + "data_pck/valid_images_X.pck")
save(valid_image_Y, dir_path + "data_pck/valid_images_Y.pck")
save(test_images_X, dir_path + "data_pck/test_images_X.pck")
save(test_images_Y, dir_path + "data_pck/test_images_Y.pck")

print(train_images_X.shape)
test()
