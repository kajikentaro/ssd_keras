import numpy as np
import os
from xml.etree import ElementTree
import argparse
import yaml

class XML_preprocessor(object):

    def __init__(self, data_path: str='',
                 num_classes: int=20):
        self.path_prefix = data_path
        self.num_classes = num_classes
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)/width
                    ymin = float(bounding_box.find('ymin').text)/height
                    xmax = float(bounding_box.find('xmax').text)/width
                    ymax = float(bounding_box.find('ymax').text)/height
                bounding_box = [xmin,ymin,xmax,ymax]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
                one_hot_class = self._to_one_hot(class_name)
                one_hot_classes.append(one_hot_class)
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data

    def _to_one_hot(self,name):
        one_hot_vector = [0] * self.num_classes
        with open("/conf/class_id.yml") as cf:  # noqa
            class_id = yaml.load(cf)
        class_id_to_name = \
            class_id["class_id_to_name_ssd"]
        for key, value in class_id_to_name.items():
            one_hot_vector[key - 1] = 1

        return one_hot_vector

## example on how to use it
import pickle
parser = argparse.ArgumentParser(description="Training voxnex with keras")
parser.add_argument("train_or_test",
                    help="set train or test")
parser.add_argument("xml_data_path", default="VOCdevkit/VOC2007/Annotations/",
                    help="set xml_data_path")
parser.add_argument("out_pkl_file", default="VOC2007.pkl",
                    help="set output pkl file name")
args = parser.parse_args()
if args.train_or_test == 'train':
    data = XML_preprocessor(args.xml_data_path).data
    pickle.dump(data,open(args.out_pkl_file,'wb'))
elif args.train_or_test == 'test':
    data = XML_preprocessor('VOCdevkit/VOC2007/Annotations_test/').data
    pickle.dump(data,open('log/VOC2007_test.pkl','wb'))

