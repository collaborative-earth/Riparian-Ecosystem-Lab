from pathlib import Path

import numpy as np
import tensorflow.keras as keras
import pickle
import os


# requires an image path with directories of train_images and test_images (with individual images stored as .npy) and the following files:
# testims_inds.npy, test_dictionary.pkl, trainims_inds.npy, and train_dictionary.pkl
# These are made by running make_data.py

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, data_path: Path,
                 batch_size=512, imsize=(256, 256, 4), shuffle_seed=444, noise_std=0,
                 test=False, val=True, dev=False, RGBN=[0, 1, 2, 3], loc=None):

        self.test = test
        self.imsize = imsize
        self.noise_std = noise_std
        self.val = False if test else val
        self.RGBN = RGBN

        labels_path = data_path / "labels.npy"
        self.labels = np.load(str(labels_path.absolute()))

        if self.test:
            test_ids_path = data_path / "testims_inds.npy"
            im_IDs = np.load(str(test_ids_path.absolute()))
            self.im_path = data_path / "test_images"
            if loc is not None:
                with Path(data_path, "test_dictionary.pkl").open('rb') as f:
                    im_dict = pickle.load(f)
                im_IDs = self.get_loc_im_IDs(im_IDs, loc, im_dict)
            if len(im_IDs) < batch_size:
                batch_size = len(im_IDs)
                print('Batch size limited to %d' % batch_size)

        else:
            train_ids_path = data_path / "trainims_inds.npy"
            im_IDs = np.load(str(train_ids_path.absolute()))
            if dev:
                im_IDs = im_IDs[:2 * batch_size]
            self.im_path = data_path / 'train_images'
            if loc is not None:
                with Path(data_path, "train_dictionary.pkl").open('rb') as f:
                    im_dict = pickle.load(f)
                im_IDs = self.get_loc_im_IDs(im_IDs, loc, im_dict)

        self.num_ims = len(im_IDs)

        np.random.seed(shuffle_seed)
        np.random.shuffle(im_IDs)
        self.im_IDs = im_IDs

        if self.val:
            self.val_IDs = self.im_IDs[:batch_size]
            self.im_IDs = self.im_IDs[batch_size:]
            self.num_ims = len(self.im_IDs)

        self.batch_size = batch_size
        self.batch_per_epoch = self.__len__()
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        n_batches = int(np.floor(self.num_ims / self.batch_size))
        if self.num_ims % self.batch_size != 0:
            n_batches += 1
        return n_batches

    def get_loc_im_IDs(self, im_IDs, loc, im_dict):
        new_inds = []
        for i in im_IDs:
            if loc in im_dict[i]:
                new_inds.append(i)
        return np.array(new_inds)

    def get_valbatch(self):
        X, y = self.__data_generation(self.val_IDs)
        X = X[:, :self.imsize[0], :self.imsize[1], :self.imsize[2]]
        return X[:, :, :, self.RGBN], y

    def get_testbatch(self, index=0):
        if self.test:
            X, y = self.__data_generation(self.im_IDs[self.batch_size * index:self.batch_size * (index + 1)])
            return X[:, :, :, self.RGBN], y
        else:
            print('Test set not loaded')  # todo: make this an error message
            return None

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.im_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X[:, :, :, self.RGBN], y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        ##self.batch_id = 0
        self.indexes = np.arange(len(self.im_IDs))
        np.random.shuffle(self.indexes)

    def label_convert(self, l):
        return int(l)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        curr_batch_size = min(self.batch_size, len(list_IDs_temp))
        X = np.zeros((curr_batch_size,) + self.imsize)
        y = np.zeros(curr_batch_size, dtype=int)

        noise = np.random.randn(X.shape[0], X.shape[1], X.shape[2], X.shape[3]) * self.noise_std
        noise[np.random.choice(curr_batch_size, size=(int(curr_batch_size * .45)), replace=False), :, :, :] = 0

        # training set mean and std
        r = [0.4023569156422335, 0.16273552077138295]
        g = [0.38864821031358476, 0.1344561089663052]
        b = [0.3274616864771629, 0.11263959320845508]
        ir = [0.47780249980851786, 0.15415727970818688]

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            curr_im_path = self.im_path / f"im{ID:06}_l{self.labels[ID]}.npy"
            im = np.load(str(curr_im_path))
            X[i,] = im[:self.imsize[0], :self.imsize[1],
                    :self.imsize[2]] / 255.  # basic crop in case some have extra pixels
            X[i, :, :, 0] = (X[i, :, :, 0] - r[0]) / r[1]
            X[i, :, :, 1] = (X[i, :, :, 1] - g[0]) / g[1]
            X[i, :, :, 2] = (X[i, :, :, 2] - b[0]) / b[1]
            X[i, :, :, 3] = (X[i, :, :, 3] - ir[0]) / ir[1]

            # Store class
            y[i] = self.label_convert(self.labels[ID])

        X += noise
        return X, y


class NoLabelDataGenerator():
    def __init__(self, data_path, batch_size=512, imsize=None):

        file_names = os.listdir(data_path)

        if imsize is None:
            # load first image from the folder and use its size:
            first_im = np.load(os.path.join(data_path, file_names[0]))
            self.imsize = first_im.shape
            print('Image size is set to ', self.imsize)
        else:
            self.imsize = imsize

        if batch_size > len(file_names):
            self.batch_size = len(file_names)
            print('Batch size is now %d' % self.batch_size)
        else:
            self.batch_size = batch_size

        self.num_ims = len(file_names)

        self.batch_per_epoch = int(np.ceil(self.num_ims / self.batch_size))
        self.im_path = data_path
        self.file_names = file_names

    def get_batch(self, index=0):
        # returns a batch of images and a list of their file names
        if index < self.batch_per_epoch:
            X = self.__data_generation(self.file_names[self.batch_size * index:self.batch_size * (index + 1)])
            return X, self.file_names[self.batch_size * index:self.batch_size * (index + 1)]
        else:
            print('Exceeds number of images')  # todo: make this an error message
            return None

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size,) + self.imsize) * np.nan

        # training set mean and std
        r = [0.4023569156422335, 0.16273552077138295]
        g = [0.38864821031358476, 0.1344561089663052]
        b = [0.3274616864771629, 0.11263959320845508]
        ir = [0.47780249980851786, 0.15415727970818688]

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            im = np.load(os.path.join(self.im_path, ID))
            X[i,] = im[:self.imsize[0], :self.imsize[1],
                    :self.imsize[2]] / 255.  # basic crop in case some have extra pixels
            X[i, :, :, 0] = (X[i, :, :, 0] - r[0]) / r[1]
            X[i, :, :, 1] = (X[i, :, :, 1] - g[0]) / g[1]
            X[i, :, :, 2] = (X[i, :, :, 2] - b[0]) / b[1]
            X[i, :, :, 3] = (X[i, :, :, 3] - ir[0]) / ir[1]

        return X
