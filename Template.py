import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
from hdf5storage import loadmat
from scipy.stats import zscore
from PIL import Image
import cv2
import csv


def data_block_loader_train_val_test_image_separate(use_all=False, eye_remove=True, do_norm=True, do_zscore=True):
    # Load data
    if use_all:
        file_name1 = 'Data/01_All/Block_Sub001_all.mat'
        full_data1 = loadmat(file_name1)
        file_name2 = 'Data/01_All/Block_Sub002_all.mat'
        full_data2 = loadmat(file_name2)
        file_name3 = 'Data/01_All/Block_Sub003_all.mat'
        full_data3 = loadmat(file_name3)
        file_name4 = 'Data/01_All/Block_Sub004_all.mat'
        full_data4 = loadmat(file_name4)
        EEGs = np.concatenate([full_data1['data'], full_data2['data'],
                               full_data3['data'], full_data4['data']], axis=0)
        images = np.concatenate([full_data1['images'], full_data2['images'],
                                 full_data3['images'], full_data4['images']], axis=0)
        labels = np.concatenate([full_data1['labels'], full_data2['labels'],
                                 full_data3['labels'], full_data4['labels']], axis=0)

    else:
        if eye_remove:
            file_name1 = 'Data/03_Remove_Bad_Trial_and_Eye_Artifact/Block_Sub001_Remove_Eye.mat'
            full_data1 = loadmat(file_name1)
            file_name2 = 'Data/03_Remove_Bad_Trial_and_Eye_Artifact/Block_Sub002_Remove_Eye.mat'
            full_data2 = loadmat(file_name2)
            file_name3 = 'Data/03_Remove_Bad_Trial_and_Eye_Artifact/Block_Sub003_Remove_Eye.mat'
            full_data3 = loadmat(file_name3)
            file_name4 = 'Data/03_Remove_Bad_Trial_and_Eye_Artifact/Block_Sub004_Remove_Eye.mat'
            full_data4 = loadmat(file_name4)
            EEGs = np.concatenate([full_data1['data'], full_data2['data'],
                                   full_data3['data'], full_data4['data']], axis=0)
            images = np.concatenate([full_data1['images'], full_data2['images'],
                                     full_data3['images'], full_data4['images']], axis=0)
            labels = np.concatenate([full_data1['labels'], full_data2['labels'],
                                     full_data3['labels'], full_data4['labels']], axis=0)

        else:
            file_name1 = 'Data/02_Remove_Bad_Trial/Block_Sub001_Remove_Trial.mat'
            full_data1 = loadmat(file_name1)
            file_name2 = 'Data/02_Remove_Bad_Trial/Block_Sub002_Remove_Trial.mat'
            full_data2 = loadmat(file_name2)
            file_name3 = 'Data/02_Remove_Bad_Trial/Block_Sub003_Remove_Trial.mat'
            full_data3 = loadmat(file_name3)
            file_name4 = 'Data/02_Remove_Bad_Trial/Block_Sub004_Remove_Trial.mat'
            full_data4 = loadmat(file_name4)
            EEGs = np.concatenate([full_data1['data'], full_data2['data'],
                                   full_data3['data'], full_data4['data']], axis=0)
            images = np.concatenate([full_data1['images'], full_data2['images'],
                                     full_data3['images'], full_data4['images']], axis=0)
            labels = np.concatenate([full_data1['labels'], full_data2['labels'],
                                     full_data3['labels'], full_data4['labels']], axis=0)

    # Parameters
    window_start = 0
    window_end = window_start + 125
    category_size = 40

    # Shuffle
    seed = 12345
    np.random.seed(seed)
    np.random.shuffle(EEGs)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # Get Image features
    images_for_vis_feature = np.empty((images.shape[0], 224, 224, 3))
    for i in range(images.shape[0]):
        image_path = 'Data/Class_Images_Selected/' + images[i]
        img = Image.open(image_path)
        img_array = np.asarray(img)
        img_array = Image.fromarray(img_array)
        img_array = np.array(img_array.resize((224, 224), Image.LANCZOS))
        if img_array.reshape(-1).shape[0] == 224 * 224:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            img_array = np.array(Image.fromarray(img_array))
        images_for_vis_feature[i] = img_array
    ResNet_Images = tf.keras.applications.resnet.preprocess_input(images_for_vis_feature)
    ResNet = tf.keras.applications.resnet.ResNet101()
    ResNet_model = Model(ResNet.input, ResNet.layers[-2].output)  # Input = (224, 224, 3), Output = (2048,)
    ResNet_Features = ResNet_model.predict(ResNet_Images)

    train_data = []
    train_vis_features = []
    train_label = []
    train_image = []
    validation_data = []
    validation_vis_features = []
    validation_label = []
    validation_image = []
    test_dat = []
    test_vis_features = []
    test_label = []
    # test_image = []
    counter = np.zeros(category_size)
    counter_train = np.zeros(category_size)
    counter_val = np.zeros(category_size)
    counter_test = np.zeros(category_size)

    images_file = 'Data/Class_Images_Selected/Image_Stimuli_Full.csv'
    with open(images_file) as f:
        data = csv.reader(f, delimiter='/')
        images_list = [row for row in data]

    val_list = []
    test_list = []
    for i in range(40):
        part_list = images_list[i * 50:i * 50 + 50]
        np.random.seed(i)
        np.random.shuffle(part_list)
        for ii in range(5):
            img_path = part_list[ii][1] + '/' + part_list[ii][2]
            val_list.append(img_path)
            img_path2 = part_list[ii + 5][1] + '/' + part_list[ii + 5][2]
            test_list.append(img_path2)

    for i in range(EEGs.shape[0]):
        if images[i] in val_list:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                validation_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                validation_data.append(norm[:, window_start:window_end])
            validation_vis_features.append(ResNet_Features[i])
            validation_label.append(labels[i])
            validation_image.append(images[i])
            counter[labels[i]] += 1
            counter_val[labels[i]] += 1

        elif images[i] in test_list:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                test_dat.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                test_dat.append(norm[:, window_start:window_end])
            test_vis_features.append(ResNet_Features[i])
            test_label.append(labels[i])
            # test_image.append(images[i])
            counter[labels[i]] += 1
            counter_test[labels[i]] += 1
        else:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                train_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                train_data.append(norm[:, window_start:window_end])
            train_vis_features.append(ResNet_Features[i])
            train_label.append(labels[i])
            train_image.append(images[i])
            counter[labels[i]] += 1
            counter_train[labels[i]] += 1

    print('Train Data')
    print(counter_train)
    print('Validation Data')
    print(counter_val)
    print('Test Data')
    print(counter_test)
    train_data = np.array(train_data)
    train_vis_features = np.array(train_vis_features)
    train_label = np.array(train_label)
    train_image = np.array(train_image)
    validation_data = np.array(validation_data)
    validation_vis_features = np.array(validation_vis_features)
    validation_label = np.array(validation_label)
    validation_image = np.array(validation_image)
    test_dat = np.array(test_dat)
    test_vis_features = np.array(test_vis_features)
    test_label = np.array(test_label)

    return train_data, train_vis_features, train_label, train_image, validation_data, validation_vis_features, validation_label, validation_image, test_dat, test_vis_features, test_label, counter


def data_block_loader_train_test_image_separate2(use_all=False, eye_remove=True, do_norm=True, do_zscore=True):
    # Load data
    if use_all:
        file_name1 = 'Data/01_All/Block_Sub001_all.mat'
        full_data1 = loadmat(file_name1)
        file_name2 = 'Data/01_All/Block_Sub002_all.mat'
        full_data2 = loadmat(file_name2)
        file_name3 = 'Data/01_All/Block_Sub003_all.mat'
        full_data3 = loadmat(file_name3)
        file_name4 = 'Data/01_All/Block_Sub004_all.mat'
        full_data4 = loadmat(file_name4)
        EEGs = np.concatenate([full_data1['data'], full_data2['data'],
                               full_data3['data'], full_data4['data']], axis=0)
        images = np.concatenate([full_data1['images'], full_data2['images'],
                                 full_data3['images'], full_data4['images']], axis=0)
        labels = np.concatenate([full_data1['labels'], full_data2['labels'],
                                 full_data3['labels'], full_data4['labels']], axis=0)

    else:
        if eye_remove:
            file_name1 = 'Data/03_Remove_Bad_Trial_and_Eye_Artifact/Block_Sub001_Remove_Eye.mat'
            full_data1 = loadmat(file_name1)
            file_name2 = 'Data/03_Remove_Bad_Trial_and_Eye_Artifact/Block_Sub002_Remove_Eye.mat'
            full_data2 = loadmat(file_name2)
            file_name3 = 'Data/03_Remove_Bad_Trial_and_Eye_Artifact/Block_Sub003_Remove_Eye.mat'
            full_data3 = loadmat(file_name3)
            file_name4 = 'Data/03_Remove_Bad_Trial_and_Eye_Artifact/Block_Sub004_Remove_Eye.mat'
            full_data4 = loadmat(file_name4)
            EEGs = np.concatenate([full_data1['data'], full_data2['data'],
                                   full_data3['data'], full_data4['data']], axis=0)
            images = np.concatenate([full_data1['images'], full_data2['images'],
                                     full_data3['images'], full_data4['images']], axis=0)
            labels = np.concatenate([full_data1['labels'], full_data2['labels'],
                                     full_data3['labels'], full_data4['labels']], axis=0)

        else:
            file_name1 = 'Data/02_Remove_Bad_Trial/Block_Sub001_Remove_Trial.mat'
            full_data1 = loadmat(file_name1)
            file_name2 = 'Data/02_Remove_Bad_Trial/Block_Sub002_Remove_Trial.mat'
            full_data2 = loadmat(file_name2)
            file_name3 = 'Data/02_Remove_Bad_Trial/Block_Sub003_Remove_Trial.mat'
            full_data3 = loadmat(file_name3)
            file_name4 = 'Data/02_Remove_Bad_Trial/Block_Sub004_Remove_Trial.mat'
            full_data4 = loadmat(file_name4)
            EEGs = np.concatenate([full_data1['data'], full_data2['data'],
                                   full_data3['data'], full_data4['data']], axis=0)
            images = np.concatenate([full_data1['images'], full_data2['images'],
                                     full_data3['images'], full_data4['images']], axis=0)
            labels = np.concatenate([full_data1['labels'], full_data2['labels'],
                                     full_data3['labels'], full_data4['labels']], axis=0)

    # Parameters
    window_start = 0
    window_end = window_start + 125
    category_size = 40

    # Shuffle
    seed = 12345
    np.random.seed(seed)
    np.random.shuffle(EEGs)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    train_data = []
    train_label = []
    train_image = []
    test_dat = []
    test_label = []
    test_image = []
    counter = np.zeros(category_size)
    counter_train = np.zeros(category_size)
    counter_test = np.zeros(category_size)

    images_file = 'Data/Class_Images_Selected/Image_Stimuli_Full.csv'
    with open(images_file) as f:
        data = csv.reader(f, delimiter='/')
        images_list = [row for row in data]

    test_list = []
    for i in range(40):
        part_list = images_list[i * 50:i * 50 + 50]
        np.random.seed(i)
        np.random.shuffle(part_list)
        for ii in range(5):
            img_path2 = part_list[ii + 5][1] + '/' + part_list[ii + 5][2]
            test_list.append(img_path2)

    for i in range(EEGs.shape[0]):
        if images[i] in test_list:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                test_dat.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                test_dat.append(norm[:, window_start:window_end])
            test_label.append(labels[i])
            test_image.append(images[i])
            counter[labels[i]] += 1
            counter_test[labels[i]] += 1
        else:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                train_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                train_data.append(norm[:, window_start:window_end])
            train_label.append(labels[i])
            train_image.append(images[i])
            counter[labels[i]] += 1
            counter_train[labels[i]] += 1

    print('Train Data')
    print(counter_train)
    print('Test Data')
    print(counter_test)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_image = np.array(train_image)
    test_dat = np.array(test_dat)
    test_label = np.array(test_label)
    test_image = np.array(test_image)

    return train_data, train_label, train_image, test_dat, test_label, test_image, counter


def data_imagination40_loader_train_val_test_sub001(do_norm=True, do_zscore=True):
    # Load data
    file_name1 = 'Data/04_Imagination/Imagine_Sub001.mat'
    full_data1 = loadmat(file_name1)
    EEGs = full_data1['data']
    images = full_data1['images']
    labels = full_data1['labels']

    # Parameters
    window_start = 0
    window_end = window_start + 125
    category_size = 40

    # Shuffle
    seed = 12345
    np.random.seed(seed)
    np.random.shuffle(EEGs)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # Get Image features
    images_for_vis_feature = np.empty((images.shape[0], 224, 224, 3))
    for i in range(images.shape[0]):
        image_path = 'Data/Class_Images_Selected/' + images[i]
        img = Image.open(image_path)
        img_array = np.asarray(img)
        img_array = Image.fromarray(img_array)
        img_array = np.array(img_array.resize((224, 224), Image.LANCZOS))
        if img_array.reshape(-1).shape[0] == 224 * 224:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            img_array = np.array(Image.fromarray(img_array))
        images_for_vis_feature[i] = img_array
    ResNet_Images = tf.keras.applications.resnet.preprocess_input(images_for_vis_feature)
    ResNet = tf.keras.applications.resnet.ResNet101()
    ResNet_model = Model(ResNet.input, ResNet.layers[-2].output)  # Input = (224, 224, 3), Output = (2048,)
    ResNet_Features = ResNet_model.predict(ResNet_Images)

    train_data = []
    train_vis_features = []
    train_label = []
    train_image = []
    validation_data = []
    validation_vis_features = []
    validation_label = []
    validation_image = []
    test_dat = []
    test_vis_features = []
    test_label = []
    # test_image = []
    counter = np.zeros(category_size)
    counter_train = np.zeros(category_size)
    counter_val = np.zeros(category_size)
    counter_test = np.zeros(category_size)

    for i in range(EEGs.shape[0]):
        if counter[labels[i]] < EEGs.shape[0] / category_size * 0.8:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                train_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                train_data.append(norm[:, window_start:window_end])
            train_vis_features.append(ResNet_Features[i])
            train_label.append(labels[i])
            train_image.append(images[i])
            counter[labels[i]] += 1
            counter_train[labels[i]] += 1
        elif counter[labels[i]] < EEGs.shape[0] / category_size * 0.9:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                validation_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                validation_data.append(norm[:, window_start:window_end])
            validation_vis_features.append(ResNet_Features[i])
            validation_label.append(labels[i])
            validation_image.append(images[i])
            counter[labels[i]] += 1
            counter_val[labels[i]] += 1
        else:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                test_dat.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                test_dat.append(norm[:, window_start:window_end])
            test_vis_features.append(ResNet_Features[i])
            test_label.append(labels[i])
            # test_image.append(images[i])
            counter[labels[i]] += 1
            counter_test[labels[i]] += 1

    print('Train Data')
    print(counter_train)
    print('Validation Data')
    print(counter_val)
    print('Test Data')
    print(counter_test)

    train_data = np.array(train_data)
    train_vis_features = np.array(train_vis_features)
    train_label = np.array(train_label)
    train_image = np.array(train_image)
    validation_data = np.array(validation_data)
    validation_vis_features = np.array(validation_vis_features)
    validation_label = np.array(validation_label)
    validation_image = np.array(validation_image)
    test_dat = np.array(test_dat)
    test_vis_features = np.array(test_vis_features)
    test_label = np.array(test_label)

    return train_data, train_vis_features, train_label, train_image, validation_data, validation_vis_features, validation_label, validation_image, test_dat, test_vis_features, test_label, counter


def data_imagination40_loader_train_val_test_sub002(do_norm=True, do_zscore=True):
    # Load data
    file_name1 = 'Data/04_Imagination/Imagine_Sub002.mat'
    full_data1 = loadmat(file_name1)
    EEGs = full_data1['data']
    images = full_data1['images']
    labels = full_data1['labels']

    # Parameters
    window_start = 0
    window_end = window_start + 125
    category_size = 40

    # Shuffle
    seed = 54321
    np.random.seed(seed)
    np.random.shuffle(EEGs)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # Get Image features
    images_for_vis_feature = np.empty((images.shape[0], 224, 224, 3))
    for i in range(images.shape[0]):
        image_path = 'Data/Class_Images_Selected/' + images[i]
        img = Image.open(image_path)
        img_array = np.asarray(img)
        img_array = Image.fromarray(img_array)
        img_array = np.array(img_array.resize((224, 224), Image.LANCZOS))
        if img_array.reshape(-1).shape[0] == 224 * 224:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            img_array = np.array(Image.fromarray(img_array))
        images_for_vis_feature[i] = img_array
    ResNet_Images = tf.keras.applications.resnet.preprocess_input(images_for_vis_feature)
    ResNet = tf.keras.applications.resnet.ResNet101()
    ResNet_model = Model(ResNet.input, ResNet.layers[-2].output)  # Input = (224, 224, 3), Output = (2048,)
    ResNet_Features = ResNet_model.predict(ResNet_Images)

    train_data = []
    train_vis_features = []
    train_label = []
    train_image = []
    validation_data = []
    validation_vis_features = []
    validation_label = []
    validation_image = []
    test_dat = []
    test_vis_features = []
    test_label = []
    # test_image = []
    counter = np.zeros(category_size)
    counter_train = np.zeros(category_size)
    counter_val = np.zeros(category_size)
    counter_test = np.zeros(category_size)

    for i in range(EEGs.shape[0]):
        if counter[labels[i]] < EEGs.shape[0] / category_size * 0.8:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                train_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                train_data.append(norm[:, window_start:window_end])
            train_vis_features.append(ResNet_Features[i])
            train_label.append(labels[i])
            train_image.append(images[i])
            counter[labels[i]] += 1
            counter_train[labels[i]] += 1
        elif counter[labels[i]] < EEGs.shape[0] / category_size * 0.9:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                validation_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                validation_data.append(norm[:, window_start:window_end])
            validation_vis_features.append(ResNet_Features[i])
            validation_label.append(labels[i])
            validation_image.append(images[i])
            counter[labels[i]] += 1
            counter_val[labels[i]] += 1
        else:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                test_dat.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                test_dat.append(norm[:, window_start:window_end])
            test_vis_features.append(ResNet_Features[i])
            test_label.append(labels[i])
            # test_image.append(images[i])
            counter[labels[i]] += 1
            counter_test[labels[i]] += 1

    print('Train Data')
    print(counter_train)
    print('Validation Data')
    print(counter_val)
    print('Test Data')
    print(counter_test)

    train_data = np.array(train_data)
    train_vis_features = np.array(train_vis_features)
    train_label = np.array(train_label)
    train_image = np.array(train_image)
    validation_data = np.array(validation_data)
    validation_vis_features = np.array(validation_vis_features)
    validation_label = np.array(validation_label)
    validation_image = np.array(validation_image)
    test_dat = np.array(test_dat)
    test_vis_features = np.array(test_vis_features)
    test_label = np.array(test_label)

    return train_data, train_vis_features, train_label, train_image, validation_data, validation_vis_features, validation_label, validation_image, test_dat, test_vis_features, test_label, counter


def data_imagination40_loader_train_val_test_sub003(do_norm=True, do_zscore=True):
    # Load data
    file_name1 = 'Data/04_Imagination/Imagine_Sub003.mat'
    full_data1 = loadmat(file_name1)
    EEGs = full_data1['data']
    images = full_data1['images']
    labels = full_data1['labels']

    # Parameters
    window_start = 0
    window_end = window_start + 125
    category_size = 40

    # Shuffle
    seed = 13579
    np.random.seed(seed)
    np.random.shuffle(EEGs)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # Get Image features
    images_for_vis_feature = np.empty((images.shape[0], 224, 224, 3))
    for i in range(images.shape[0]):
        image_path = 'Data/Class_Images_Selected/' + images[i]
        img = Image.open(image_path)
        img_array = np.asarray(img)
        img_array = Image.fromarray(img_array)
        img_array = np.array(img_array.resize((224, 224), Image.LANCZOS))
        if img_array.reshape(-1).shape[0] == 224 * 224:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            img_array = np.array(Image.fromarray(img_array))
        images_for_vis_feature[i] = img_array
    ResNet_Images = tf.keras.applications.resnet.preprocess_input(images_for_vis_feature)
    ResNet = tf.keras.applications.resnet.ResNet101()
    ResNet_model = Model(ResNet.input, ResNet.layers[-2].output)  # Input = (224, 224, 3), Output = (2048,)
    ResNet_Features = ResNet_model.predict(ResNet_Images)

    train_data = []
    train_vis_features = []
    train_label = []
    train_image = []
    validation_data = []
    validation_vis_features = []
    validation_label = []
    validation_image = []
    test_dat = []
    test_vis_features = []
    test_label = []
    # test_image = []
    counter = np.zeros(category_size)
    counter_train = np.zeros(category_size)
    counter_val = np.zeros(category_size)
    counter_test = np.zeros(category_size)

    for i in range(EEGs.shape[0]):
        if counter[labels[i]] < EEGs.shape[0] / category_size * 0.8:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                train_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                train_data.append(norm[:, window_start:window_end])
            train_vis_features.append(ResNet_Features[i])
            train_label.append(labels[i])
            train_image.append(images[i])
            counter[labels[i]] += 1
            counter_train[labels[i]] += 1
        elif counter[labels[i]] < EEGs.shape[0] / category_size * 0.9:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                validation_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                validation_data.append(norm[:, window_start:window_end])
            validation_vis_features.append(ResNet_Features[i])
            validation_label.append(labels[i])
            validation_image.append(images[i])
            counter[labels[i]] += 1
            counter_val[labels[i]] += 1
        else:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                test_dat.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                test_dat.append(norm[:, window_start:window_end])
            test_vis_features.append(ResNet_Features[i])
            test_label.append(labels[i])
            # test_image.append(images[i])
            counter[labels[i]] += 1
            counter_test[labels[i]] += 1

    print('Train Data')
    print(counter_train)
    print('Validation Data')
    print(counter_val)
    print('Test Data')
    print(counter_test)

    train_data = np.array(train_data)
    train_vis_features = np.array(train_vis_features)
    train_label = np.array(train_label)
    train_image = np.array(train_image)
    validation_data = np.array(validation_data)
    validation_vis_features = np.array(validation_vis_features)
    validation_label = np.array(validation_label)
    validation_image = np.array(validation_image)
    test_dat = np.array(test_dat)
    test_vis_features = np.array(test_vis_features)
    test_label = np.array(test_label)

    return train_data, train_vis_features, train_label, train_image, validation_data, validation_vis_features, validation_label, validation_image, test_dat, test_vis_features, test_label, counter


def data_imagination40_loader_train_val_test_sub004(do_norm=True, do_zscore=True):
    # Load data
    file_name1 = 'Data/04_Imagination/Imagine_Sub004.mat'
    full_data1 = loadmat(file_name1)
    EEGs = full_data1['data']
    images = full_data1['images']
    labels = full_data1['labels']

    # Parameters
    window_start = 0
    window_end = window_start + 125
    category_size = 40

    # Shuffle
    seed = 97531
    np.random.seed(seed)
    np.random.shuffle(EEGs)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # Get Image features
    images_for_vis_feature = np.empty((images.shape[0], 224, 224, 3))
    for i in range(images.shape[0]):
        image_path = 'Data/Class_Images_Selected/' + images[i]
        img = Image.open(image_path)
        img_array = np.asarray(img)
        img_array = Image.fromarray(img_array)
        img_array = np.array(img_array.resize((224, 224), Image.LANCZOS))
        if img_array.reshape(-1).shape[0] == 224 * 224:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            img_array = np.array(Image.fromarray(img_array))
        images_for_vis_feature[i] = img_array
    ResNet_Images = tf.keras.applications.resnet.preprocess_input(images_for_vis_feature)
    ResNet = tf.keras.applications.resnet.ResNet101()
    ResNet_model = Model(ResNet.input, ResNet.layers[-2].output)  # Input = (224, 224, 3), Output = (2048,)
    ResNet_Features = ResNet_model.predict(ResNet_Images)

    train_data = []
    train_vis_features = []
    train_label = []
    train_image = []
    validation_data = []
    validation_vis_features = []
    validation_label = []
    validation_image = []
    test_dat = []
    test_vis_features = []
    test_label = []
    # test_image = []
    counter = np.zeros(category_size)
    counter_train = np.zeros(category_size)
    counter_val = np.zeros(category_size)
    counter_test = np.zeros(category_size)

    for i in range(EEGs.shape[0]):
        if counter[labels[i]] < EEGs.shape[0] / category_size * 0.8:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                train_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                train_data.append(norm[:, window_start:window_end])
            train_vis_features.append(ResNet_Features[i])
            train_label.append(labels[i])
            train_image.append(images[i])
            counter[labels[i]] += 1
            counter_train[labels[i]] += 1
        elif counter[labels[i]] < EEGs.shape[0] / category_size * 0.9:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                validation_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                validation_data.append(norm[:, window_start:window_end])
            validation_vis_features.append(ResNet_Features[i])
            validation_label.append(labels[i])
            validation_image.append(images[i])
            counter[labels[i]] += 1
            counter_val[labels[i]] += 1
        else:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                test_dat.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                test_dat.append(norm[:, window_start:window_end])
            test_vis_features.append(ResNet_Features[i])
            test_label.append(labels[i])
            # test_image.append(images[i])
            counter[labels[i]] += 1
            counter_test[labels[i]] += 1

    print('Train Data')
    print(counter_train)
    print('Validation Data')
    print(counter_val)
    print('Test Data')
    print(counter_test)

    train_data = np.array(train_data)
    train_vis_features = np.array(train_vis_features)
    train_label = np.array(train_label)
    train_image = np.array(train_image)
    validation_data = np.array(validation_data)
    validation_vis_features = np.array(validation_vis_features)
    validation_label = np.array(validation_label)
    validation_image = np.array(validation_image)
    test_dat = np.array(test_dat)
    test_vis_features = np.array(test_vis_features)
    test_label = np.array(test_label)

    return train_data, train_vis_features, train_label, train_image, validation_data, validation_vis_features, validation_label, validation_image, test_dat, test_vis_features, test_label, counter


def data_imagination40_loader_train_val_test_sub_all(do_norm=True, do_zscore=True):
    train_data = []
    train_vis_features = []
    train_label = []
    train_image = []
    validation_data = []
    validation_vis_features = []
    validation_label = []
    validation_image = []
    test_dat = []
    test_vis_features = []
    test_label = []
    # test_image = []

    # Load data Sub001
    file_name1 = 'Data/04_Imagination/Imagine_Sub001.mat'
    full_data1 = loadmat(file_name1)
    EEGs = full_data1['data']
    images = full_data1['images']
    labels = full_data1['labels']

    # Parameters
    window_start = 0
    window_end = window_start + 125
    category_size = 40

    # Shuffle
    seed = 12345
    np.random.seed(seed)
    np.random.shuffle(EEGs)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # Get Image features
    images_for_vis_feature = np.empty((images.shape[0], 224, 224, 3))
    for i in range(images.shape[0]):
        image_path = 'Data/Class_Images_Selected/' + images[i]
        img = Image.open(image_path)
        img_array = np.asarray(img)
        img_array = Image.fromarray(img_array)
        img_array = np.array(img_array.resize((224, 224), Image.LANCZOS))
        if img_array.reshape(-1).shape[0] == 224 * 224:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            img_array = np.array(Image.fromarray(img_array))
        images_for_vis_feature[i] = img_array
    ResNet_Images = tf.keras.applications.resnet.preprocess_input(images_for_vis_feature)
    ResNet = tf.keras.applications.resnet.ResNet101()
    ResNet_model = Model(ResNet.input, ResNet.layers[-2].output)  # Input = (224, 224, 3), Output = (2048,)
    ResNet_Features = ResNet_model.predict(ResNet_Images)

    counter = np.zeros(category_size)
    counter_train = np.zeros(category_size)
    counter_val = np.zeros(category_size)
    counter_test = np.zeros(category_size)

    for i in range(EEGs.shape[0]):
        if counter[labels[i]] < EEGs.shape[0] / category_size * 0.8:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                train_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                train_data.append(norm[:, window_start:window_end])
            train_vis_features.append(ResNet_Features[i])
            train_label.append(labels[i])
            train_image.append(images[i])
            counter[labels[i]] += 1
            counter_train[labels[i]] += 1
        elif counter[labels[i]] < EEGs.shape[0] / category_size * 0.9:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                validation_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                validation_data.append(norm[:, window_start:window_end])
            validation_vis_features.append(ResNet_Features[i])
            validation_label.append(labels[i])
            validation_image.append(images[i])
            counter[labels[i]] += 1
            counter_val[labels[i]] += 1
        else:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                test_dat.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                test_dat.append(norm[:, window_start:window_end])
            test_vis_features.append(ResNet_Features[i])
            test_label.append(labels[i])
            # test_image.append(images[i])
            counter[labels[i]] += 1
            counter_test[labels[i]] += 1

    print('Sub 001')
    print('Train Data')
    print(counter_train)
    print('Validation Data')
    print(counter_val)
    print('Test Data')
    print(counter_test)

    # Load data Sub002
    file_name1 = 'Data/04_Imagination/Imagine_Sub002.mat'
    full_data1 = loadmat(file_name1)
    EEGs = full_data1['data']
    images = full_data1['images']
    labels = full_data1['labels']

    # Parameters
    window_start = 0
    window_end = window_start + 125
    category_size = 40

    # Shuffle
    seed = 54321
    np.random.seed(seed)
    np.random.shuffle(EEGs)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # Get Image features
    images_for_vis_feature = np.empty((images.shape[0], 224, 224, 3))
    for i in range(images.shape[0]):
        image_path = 'Data/Class_Images_Selected/' + images[i]
        img = Image.open(image_path)
        img_array = np.asarray(img)
        img_array = Image.fromarray(img_array)
        img_array = np.array(img_array.resize((224, 224), Image.LANCZOS))
        if img_array.reshape(-1).shape[0] == 224 * 224:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            img_array = np.array(Image.fromarray(img_array))
        images_for_vis_feature[i] = img_array
    ResNet_Images = tf.keras.applications.resnet.preprocess_input(images_for_vis_feature)
    ResNet = tf.keras.applications.resnet.ResNet101()
    ResNet_model = Model(ResNet.input, ResNet.layers[-2].output)  # Input = (224, 224, 3), Output = (2048,)
    ResNet_Features = ResNet_model.predict(ResNet_Images)

    counter = np.zeros(category_size)
    counter_train = np.zeros(category_size)
    counter_val = np.zeros(category_size)
    counter_test = np.zeros(category_size)

    for i in range(EEGs.shape[0]):
        if counter[labels[i]] < EEGs.shape[0] / category_size * 0.8:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                train_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                train_data.append(norm[:, window_start:window_end])
            train_vis_features.append(ResNet_Features[i])
            train_label.append(labels[i])
            train_image.append(images[i])
            counter[labels[i]] += 1
            counter_train[labels[i]] += 1
        elif counter[labels[i]] < EEGs.shape[0] / category_size * 0.9:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                validation_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                validation_data.append(norm[:, window_start:window_end])
            validation_vis_features.append(ResNet_Features[i])
            validation_label.append(labels[i])
            validation_image.append(images[i])
            counter[labels[i]] += 1
            counter_val[labels[i]] += 1
        else:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                test_dat.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                test_dat.append(norm[:, window_start:window_end])
            test_vis_features.append(ResNet_Features[i])
            test_label.append(labels[i])
            # test_image.append(images[i])
            counter[labels[i]] += 1
            counter_test[labels[i]] += 1

    print('Sub 002')
    print('Train Data')
    print(counter_train)
    print('Validation Data')
    print(counter_val)
    print('Test Data')
    print(counter_test)

    # Load data Sub003
    file_name1 = 'Data/04_Imagination/Imagine_Sub003.mat'
    full_data1 = loadmat(file_name1)
    EEGs = full_data1['data']
    images = full_data1['images']
    labels = full_data1['labels']

    # Parameters
    window_start = 0
    window_end = window_start + 125
    category_size = 40

    # Shuffle
    seed = 13579
    np.random.seed(seed)
    np.random.shuffle(EEGs)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # Get Image features
    images_for_vis_feature = np.empty((images.shape[0], 224, 224, 3))
    for i in range(images.shape[0]):
        image_path = 'Data/Class_Images_Selected/' + images[i]
        img = Image.open(image_path)
        img_array = np.asarray(img)
        img_array = Image.fromarray(img_array)
        img_array = np.array(img_array.resize((224, 224), Image.LANCZOS))
        if img_array.reshape(-1).shape[0] == 224 * 224:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            img_array = np.array(Image.fromarray(img_array))
        images_for_vis_feature[i] = img_array
    ResNet_Images = tf.keras.applications.resnet.preprocess_input(images_for_vis_feature)
    ResNet = tf.keras.applications.resnet.ResNet101()
    ResNet_model = Model(ResNet.input, ResNet.layers[-2].output)  # Input = (224, 224, 3), Output = (2048,)
    ResNet_Features = ResNet_model.predict(ResNet_Images)

    counter = np.zeros(category_size)
    counter_train = np.zeros(category_size)
    counter_val = np.zeros(category_size)
    counter_test = np.zeros(category_size)

    for i in range(EEGs.shape[0]):
        if counter[labels[i]] < EEGs.shape[0] / category_size * 0.8:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                train_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                train_data.append(norm[:, window_start:window_end])
            train_vis_features.append(ResNet_Features[i])
            train_label.append(labels[i])
            train_image.append(images[i])
            counter[labels[i]] += 1
            counter_train[labels[i]] += 1
        elif counter[labels[i]] < EEGs.shape[0] / category_size * 0.9:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                validation_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                validation_data.append(norm[:, window_start:window_end])
            validation_vis_features.append(ResNet_Features[i])
            validation_label.append(labels[i])
            validation_image.append(images[i])
            counter[labels[i]] += 1
            counter_val[labels[i]] += 1
        else:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                test_dat.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                test_dat.append(norm[:, window_start:window_end])
            test_vis_features.append(ResNet_Features[i])
            test_label.append(labels[i])
            # test_image.append(images[i])
            counter[labels[i]] += 1
            counter_test[labels[i]] += 1

    print('Sub 003')
    print('Train Data')
    print(counter_train)
    print('Validation Data')
    print(counter_val)
    print('Test Data')
    print(counter_test)

    # Load data Sub004
    file_name1 = 'Data/04_Imagination/Imagine_Sub004.mat'
    full_data1 = loadmat(file_name1)
    EEGs = full_data1['data']
    images = full_data1['images']
    labels = full_data1['labels']

    # Parameters
    window_start = 0
    window_end = window_start + 125
    category_size = 40

    # Shuffle
    seed = 97531
    np.random.seed(seed)
    np.random.shuffle(EEGs)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # Get Image features
    images_for_vis_feature = np.empty((images.shape[0], 224, 224, 3))
    for i in range(images.shape[0]):
        image_path = 'Data/Class_Images_Selected/' + images[i]
        img = Image.open(image_path)
        img_array = np.asarray(img)
        img_array = Image.fromarray(img_array)
        img_array = np.array(img_array.resize((224, 224), Image.LANCZOS))
        if img_array.reshape(-1).shape[0] == 224 * 224:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            img_array = np.array(Image.fromarray(img_array))
        images_for_vis_feature[i] = img_array
    ResNet_Images = tf.keras.applications.resnet.preprocess_input(images_for_vis_feature)
    ResNet = tf.keras.applications.resnet.ResNet101()
    ResNet_model = Model(ResNet.input, ResNet.layers[-2].output)  # Input = (224, 224, 3), Output = (2048,)
    ResNet_Features = ResNet_model.predict(ResNet_Images)

    counter = np.zeros(category_size)
    counter_train = np.zeros(category_size)
    counter_val = np.zeros(category_size)
    counter_test = np.zeros(category_size)

    for i in range(EEGs.shape[0]):
        if counter[labels[i]] < EEGs.shape[0] / category_size * 0.8:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                train_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                train_data.append(norm[:, window_start:window_end])
            train_vis_features.append(ResNet_Features[i])
            train_label.append(labels[i])
            train_image.append(images[i])
            counter[labels[i]] += 1
            counter_train[labels[i]] += 1
        elif counter[labels[i]] < EEGs.shape[0] / category_size * 0.9:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                validation_data.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                validation_data.append(norm[:, window_start:window_end])
            validation_vis_features.append(ResNet_Features[i])
            validation_label.append(labels[i])
            validation_image.append(images[i])
            counter[labels[i]] += 1
            counter_val[labels[i]] += 1
        else:
            if do_norm:
                norm = EEGs[i] - np.average(EEGs[i], axis=0)
            else:
                norm = EEGs[i]

            if do_zscore:
                test_dat.append(zscore(norm, axis=1, ddof=1)[:, window_start:window_end])
            else:
                test_dat.append(norm[:, window_start:window_end])
            test_vis_features.append(ResNet_Features[i])
            test_label.append(labels[i])
            # test_image.append(images[i])
            counter[labels[i]] += 1
            counter_test[labels[i]] += 1

    print('Sub 004')
    print('Train Data')
    print(counter_train)
    print('Validation Data')
    print(counter_val)
    print('Test Data')
    print(counter_test)

    train_data = np.array(train_data)
    train_vis_features = np.array(train_vis_features)
    train_label = np.array(train_label)
    train_image = np.array(train_image)
    validation_data = np.array(validation_data)
    validation_vis_features = np.array(validation_vis_features)
    validation_label = np.array(validation_label)
    validation_image = np.array(validation_image)
    test_dat = np.array(test_dat)
    test_vis_features = np.array(test_vis_features)
    test_label = np.array(test_label)

    return train_data, train_vis_features, train_label, train_image, validation_data, validation_vis_features, validation_label, validation_image, test_dat, test_vis_features, test_label, counter
    
