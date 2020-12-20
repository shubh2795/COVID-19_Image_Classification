from sklearn import tree, metrics
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from covid_data_loader_rf import load_data_wrapper
import pickle

covid_train_data, covid_valid_data, covid_test_data = \
    load_data_wrapper()

covid_train_data_dc = np.zeros((len(covid_train_data), 4096))
covid_test_data_dc = np.zeros((len(covid_test_data), 4096))
covid_valid_data_dc = np.zeros((len(covid_valid_data), 4096))

covid_train_target_dc = None
covid_test_target_dc = None
covid_valid_target_dc = None


def reshape_covid_aux(covid_data, covid_data_dc):
    for i in range(len(covid_data)):
        covid_data_dc[i] = covid_data[i][0].reshape((4096,))


def reshape_covid_data():
    global covid_train_data
    global covid_train_data_dc
    global covid_test_data
    global covid_test_data_dc
    global covid_valid_data
    global covid_valid_data_dc
    reshape_covid_aux(covid_train_data, covid_train_data_dc)
    reshape_covid_aux(covid_test_data, covid_test_data_dc)
    reshape_covid_aux(covid_valid_data, covid_valid_data_dc)


def reshape_covid_target(covid_data):
    return np.array([np.argmax(covid_data[i][1])
                     for i in range(len(covid_data))])


def reshape_covid_target2(covid_data):
    return np.array([covid_data[i][1] for i in range(len(covid_data))])


def prepare_covid_data():
    global covid_train_data
    global covid_test_data
    global covid_valid_data
    reshape_covid_data()

    for i in range(len(covid_train_data)):
        assert np.array_equal(covid_train_data[i][0].reshape((4096,)),
                              covid_train_data_dc[i])

    for i in range(len(covid_test_data)):
        assert np.array_equal(covid_test_data[i][0].reshape((4096,)),
                              covid_test_data_dc[i])

    for i in range(len(covid_valid_data)):
        assert np.array_equal(covid_valid_data[i][0].reshape((4096,)),
                              covid_valid_data_dc[i])


def prepare_covid_targets():
    global covid_train_target_dc
    global covid_test_target_dc
    global covid_valid_target_dc
    covid_train_target_dc = reshape_covid_target(covid_train_data)
    covid_test_target_dc = reshape_covid_target2(covid_test_data)
    covid_valid_target_dc = reshape_covid_target2(covid_valid_data)


def fit_validate_dt():
    clf = tree.DecisionTreeClassifier(random_state=random.randint(0, 1000))
    dtr = clf.fit(covid_train_data_dc, covid_train_target_dc)
    save(dtr, "models/dtr.pck")
    valid_preds = dtr.predict(covid_valid_data_dc)
    print(metrics.classification_report(covid_valid_target_dc, valid_preds))
    cm1 = confusion_matrix(covid_valid_target_dc, valid_preds)
    print("confusion matrix ", cm1)


def fit_validate_dts(num_dts):
    for _ in range(num_dts):
        fit_validate_dt()


def fit_validate_rf(num_dts):
    rs = random.randint(0, 1000)
    clf = RandomForestClassifier(n_estimators=num_dts, random_state=rs)
    rf = clf.fit(covid_train_data_dc, covid_train_target_dc)
    valid_preds = rf.predict(covid_valid_data_dc)
    print(metrics.classification_report(covid_valid_target_dc, valid_preds))
    cm1 = confusion_matrix(covid_valid_target_dc, valid_preds)
    print(cm1)


def fit_validate_rfs(low_nt, high_nt):
    for i in range(low_nt, high_nt + 1, 10):
        print(i, ": ")
        fit_validate_rf(i)

def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

prepare_covid_data()
prepare_covid_targets()
