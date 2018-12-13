import os
import random
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Tuple, List

import PIL
import torch.utils.data
from PIL import Image, ImageOps
from torch import Tensor
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        EVAL = 'eval'

    class Annotation(object):
        class Object(object):
            def __init__(self, name: str, difficult: bool):
                super().__init__()
                self.name = name
                self.difficult = difficult

            def __repr__(self) -> str:
                return 'Object[name={:s}, difficult={!s}]'.format(self.name, self.difficult)

        def __init__(self, filename: str, objects: List[Object]):
            super().__init__()
            self.filename = filename
            self.objects = objects

    NUM_CLASSES = 20

    CATEGORY_TO_LABEL_DICT = {
        'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
        'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
    }

    LABEL_TO_CATEGORY_DICT = {v: k for k, v in CATEGORY_TO_LABEL_DICT.items()}

    def __init__(self, path_to_data_dir: str, mode: Mode):
        super().__init__()

        self._path_to_data_dir = path_to_data_dir
        self._mode = mode

        path_to_voc2007_dir = os.path.join(self._path_to_data_dir, 'VOCdevkit', 'VOC2007')
        path_to_imagesets_main_dir = os.path.join(path_to_voc2007_dir, 'ImageSets', 'Main')
        path_to_annotations_dir = os.path.join(path_to_voc2007_dir, 'Annotations')
        self._path_to_jpeg_images_dir = os.path.join(path_to_voc2007_dir, 'JPEGImages')

        if self._mode == Dataset.Mode.TRAIN:
            path_to_image_ids_txt = os.path.join(path_to_imagesets_main_dir, 'trainval.txt')
        elif self._mode == Dataset.Mode.EVAL:
            path_to_image_ids_txt = os.path.join(path_to_imagesets_main_dir, 'test.txt')
        else:
            raise ValueError('invalid mode')
        
        with open(path_to_image_ids_txt, 'r') as f:
            lines = f.readlines()
            self._image_ids = [line.rstrip() for line in lines]

        self._image_id_to_annotation_dict = {}
        for image_id in self._image_ids:
            path_to_annotation_xml = os.path.join(path_to_annotations_dir, f'{image_id}.xml')
            tree = ET.ElementTree(file=path_to_annotation_xml)
            root = tree.getroot()

            self._image_id_to_annotation_dict[image_id] = Dataset.Annotation(
                filename=next(root.iterfind('filename')).text,
                objects=[Dataset.Annotation.Object(name=next(tag_object.iterfind('name')).text,
                                                   difficult=next(tag_object.iterfind('difficult')).text == '1')
                         for tag_object in root.iterfind('object')]
            )

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image_id = self._image_ids[index]
        annotation = self._image_id_to_annotation_dict[image_id]

        multilabels = [0] * Dataset.NUM_CLASSES
        labels = [Dataset.CATEGORY_TO_LABEL_DICT[obj.name] for obj in annotation.objects if not obj.difficult]
        for label in labels:
            multilabels[label] = 1
        multilabels = torch.tensor(multilabels, dtype=torch.long)

        image = Image.open(os.path.join(self._path_to_jpeg_images_dir, annotation.filename))

        # random flip on only training mode
        if self._mode == Dataset.Mode.TRAIN and random.random() > 0.5:
            image = ImageOps.mirror(image)

        image = self.preprocess(image)

        return image, multilabels

    def preprocess(self, image: PIL.Image.Image) -> Tensor:
        crop_transform = transforms.RandomCrop(224) if self._mode == Dataset.Mode.TRAIN else transforms.CenterCrop(224)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            crop_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
        return image


if __name__ == '__main__':
    def main():
        dataset = Dataset('data', Dataset.Mode.TRAIN)
        print('dataset length:', len(dataset))
        image, multilabels = dataset[1017]
        print('image.shape:', image.shape)
        print('multilabels:', multilabels)

    main()
