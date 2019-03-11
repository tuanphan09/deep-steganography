import numpy as np
import cv2

import keras
 
class ImageDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_files, dim, std, mean, batch_size=64, shuffle=True):
        'Initialization'
        
        self.list_files = list_files
        self.dim = dim
        self.std = std
        self.mean = mean
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_files_batch = [self.list_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_files_batch)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def normalize_batch(self, imgs):
        imgs = np.array(imgs, dtype=np.float32)
        imgs /= 255
        return (imgs -  self.mean) / self.std
                                                            
    def denormalize_batch(self, imgs, should_clip=True):
        imgs= imgs * self.std + self.mean 
        if should_clip:
            imgs = np.clip(imgs,0,1)
        imgs *= 255
        return imgs

    def __data_generation(self, list_files_batch):
        'Generates data containing batch_size samples' 
        X = []
        for i, fname in enumerate(list_files_batch):
            img = cv2.imread(fname)
            img = cv2.resize(img, (self.dim[1], self.dim[0]))
            # print(img.shape)
            X.append(img)
        X = self.normalize_batch(X)
        mid = int(len(X) / 2)
        secret = X[:mid]
        cover = X[mid:]
        return [secret, cover], [cover, secret]