#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pickle
from ssd_utils import BBoxUtility
from generator import Generator
from ssd_training import MultiboxLoss
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from time import gmtime, strftime
import os


def schedule(epoch, base_lr=3e-4, decay=0.9):
    return base_lr * decay ** (epoch)


class Trainer(object):
    """
    Trainer for ssd_model
    """
    def __init__(self,
                 class_number=21,
                 input_shape=(300, 300, 3),
                 priors_file='prior_boxes_ssd300.pkl',
                 train_file='VOC2007.pkl',
                 path_prefix='./VOCdevkit/VOC2007/JPEGImages/',
                 model=None,
                 weight_file='weights_SSD300.hdf5',
                 freeze=('input_1', 'conv1_1', 'conv1_2', 'pool1',
                         'conv2_1', 'conv2_2', 'pool2',
                         'conv3_1', 'conv3_2', 'conv3_3', 'pool3'),
                 save_weight_file='/src/resource/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',  # noqa
                 optim=None,
                ):
        """
        Setting below parameter 
        :param class_number(int): class number
        :param input_shape(set): set input shape  
        :param priors_file(str): set prior file name 
        :param train_file(str): train file name  
        :param path_prefix(str): path prefix 
        :param model(keras model): set the keras model such as the ssd 
        :param weight_file(str): weight file name 
        :param freeze(set): set untraining layer 
        """
        self.input_shape = input_shape
        priors = pickle.load(open(priors_file, 'rb'))
        self.bbox_utils = BBoxUtility(class_number, priors)
        self.train_data = pickle.load(open(train_file, 'rb'))
        keys = sorted(self.train_data.keys())
        num_train = int(round(0.8 * len(keys)))
        self.train_keys = keys[:num_train]
        self.val_keys = keys[num_train:]
        self.num_val = len(self.val_keys)
        self.gen = Generator(self.train_data, self.bbox_utils, 20, path_prefix,
                             self.train_keys, self.val_keys,
                             (self.input_shape[0], self.input_shape[1]),
                             do_crop=False)
        self.model = model
        model.load_weights(weight_file, by_name=True)
        self.freeze = list(freeze)
        self.save_weight_file = save_weight_file
        self.optim = optim
        self.model.compile(optimizer=optim,
                           metrics=['accuracy'],
                           loss=MultiboxLoss(class_number,
                                             neg_pos_ratio=2.0).compute_loss)

    def train(self, nb_epoch):
        """
        Call Train
        :param nb_epoch(int): setting number of epoch 
        """
        for L in self.model.layers:
            if L.name in self.freeze:
                L.trainable = False
        callbacks = [ModelCheckpoint(self.save_weight_file, verbose=1,
                                     save_weights_only=True)]
        callbacks.append(self.__make_tensorboard())
        history = self.model.fit_generator(self.gen.generate(True),
                                           self.gen.train_batches,
                                      nb_epoch, verbose=1,
                                      callbacks=callbacks,
                                      validation_data=self.gen.generate(False),
                                      nb_val_samples=self.gen.val_batches,
                                      nb_worker=10)

    def __make_tensorboard(self):
        """
        Make tensorboard for visualize information
        :return: tensorboard
        """
        tictoc = strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime())
        directory_name = tictoc
        self.log_dir = "./log/" + directory_name
        os.mkdir(self.log_dir)
        tensorboard = TensorBoard(log_dir=self.log_dir, histogram_freq=1,
                                  write_graph=True, )
        return tensorboard

