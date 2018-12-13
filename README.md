# VOC2007-Multilabel-TODO.pytorch

![](images/sample.jpg)

> Image with multilabel of `diningtable`, `person` and `sofa`


## Requirements

* Python 3.6
* PyTorch 0.4.1


## Setup

1. Download `PASCAL VOC 2007` dataset

    - [Training / Validation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (5011 images)
    - [Test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (4952 images)

1. Extract to data folder, now your folder structure should be like:

    ```
    VOC2007-Multilabel-TODO.pytorch
        - data
            - VOCdevkit
                - VOC2007
                    - Annotations
                        - 000001.xml
                        - 000002.xml
                        ...
                    - ImageSets
                        - Main
                            ...
                            test.txt
                            ...
                            trainval.txt
                            ...
                    - JPEGImages
                        - 000001.jpg
                        - 000002.jpg
                        ...
            - ...
    ```


## Usage

1. Train
    ```
    $ python train.py -d=./data -c=./checkpoints
    ```

1. Evaluate
    ```
    $ python eval.py ./checkpoint/model-100.pth -d=./data
    ```
