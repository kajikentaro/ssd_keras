#!/usr/bin/python3
# -*- coding: utf-8 -*-

# from trainer import Trainer
import pyximport
pyximport.install()
from cython_train.trainer_cython import Trainer
from ssd_v2 import SSD300v2
import keras
import argparse


def main():
    parser = argparse.ArgumentParser(description="Training ssd model with keras")
    parser.add_argument("-c", "--class_number", metavar="class_number",
                        type=int, default=21,
                        dest="class_number", help="set the classify number ")
    parser.add_argument("-b", "--prior_boxes_ssd300", metavar="prior_boxes_ssd300",
                        type=str, default='prior_boxes_ssd300.pkl',
                        dest="prior_boxes_ssd300", help="set the prior boxes file")
    parser.add_argument("-t", "--train_file", metavar="train_file",
                        type=str, default='VOC2007.pkl',
                        dest="train_file", help="set the train file")
    parser.add_argument("-p", "--path_prefix", metavar="path_prefix",
                        type=str, default='./VOCdevkit/VOC2007/JPEGImages/',
                        dest="path_prefix", help="set the path prefix")
    parser.add_argument("-w", "--weight_file", metavar="weight_file",
                        type=str, default='weights_SSD300.hdf5',
                        dest="weight_file", help="set the weight file")
    parser.add_argument("-s", "--save_weight_file", metavar="save_weight_file",
                        type=str,
                        default='./resource/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                        dest="save_weight_file", help="set the save weight file")
    parser.add_argument("-n", "--nb_epoch", metavar="nb_epoch",
                        type=int,
                        default=100,
                        dest="nb_epoch", help="set the number of epoch")
    args = parser.parse_args()
    input_shape = (300, 300, 3)
    model = SSD300v2(input_shape, num_classes=args.class_number)
    base_lr=3e-4
    trainer = Trainer(class_number=args.class_number,
                           input_shape=input_shape,
                           priors_file=args.prior_boxes_ssd300,
                           train_file=args.train_file,
                           path_prefix=args.path_prefix,
                           model=model,
                           weight_file=args.weight_file,
                           freeze=('input_1', 'conv1_1', 'conv1_2', 'pool1',
                                   'conv2_1', 'conv2_2', 'pool2',
                                   'conv3_1', 'conv3_2', 'conv3_3', 'pool3'),
                           save_weight_file=args.save_weight_file,
                           optim=keras.optimizers.Adam(lr=base_lr),
                           )
    trainer.train(nb_epoch=args.nb_epoch)


if __name__ == "__main__":
    main()
