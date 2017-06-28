#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
from trainer import Trainer
from ssd_v2 import SSD300v2
import keras
import subprocess


class Test_Trainer(unittest.TestCase):

    def setUp(self):
        self.class_number = 21
        self.input_shape = (300, 300, 3)
        self.model = SSD300v2(self.input_shape, num_classes=self.class_number)

    def test_train(self):
        base_lr=3e-4
        self.trainer = Trainer(class_number=self.class_number,
                               input_shape=self.input_shape,
                               priors_file='prior_boxes_ssd300.pkl',
                               train_file='VOC2007_test.pkl',
                               path_prefix='./VOCdevkit/VOC2007/JPEGImages/',
                               model=self.model,
                               weight_file='weights_SSD300.hdf5',
                               freeze=('input_1', 'conv1_1', 'conv1_2', 'pool1',
                                       'conv2_1', 'conv2_2', 'pool2',
                                       'conv3_1', 'conv3_2', 'conv3_3', 'pool3'),
                               save_weight_file='./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',  # noqa
                               optim=keras.optimizers.Adam(lr=base_lr),
                               )
        self.trainer.train(nb_epoch=1)

    def teardown(self):
        try:
            subprocess.call("rm -rf " + self.trainer.log_dir, shell=True)
        except subprocess.CalledProcessError as cpe:
            print(str(cpe))


if __name__ == '__main__':
    unittest.main()