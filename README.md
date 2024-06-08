# DL-Assignment-2

[Report for this Assignment](https://drive.google.com/file/d/14Ppg4bplJ4vF4cHtvGvWguA-b2pV_-Se/view?usp=sharing)

## Task1 - Designing a Convolution Module for Variable Input Channels

[Checkpoints - Dynamic weight-generating network with DenseNet-121](https://drive.google.com/file/d/1t97D2Pwd6Krz_G9l0GUDrSo_1ZRD0CJZ/view?usp=sharing)

**Performance and Complexity comparison on mini-ImageNet. Mulit-Adds are calculated for a 3x224x224 input.**
| Model | Params | Multi-Adds | Forward | FLOPs | Accuracy | Precision | Recall | f1-score |
|:-----------:|:---------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [Dynamic weight-generating network with DenseNet-121](https://drive.google.com/file/d/1mHkrJAbGcwyt3P3oGzKoBOCKLWtD9U6p/view?usp=sharing) |   32.84M    | 14.13G | 174.19M | 2.65 G | 65.78% | 69.79% | 65.78% | 65.47% |\
| [Dynamic weight-generating network with DenseNet-121 (Channel Shuffling Augmentation)](https://drive.google.com/file/d/1t97D2Pwd6Krz_G9l0GUDrSo_1ZRD0CJZ/view?usp=sharing) |  32.84M  | 14.13G | 174.19M | 2.65 G | 76.00% | 76.61% | 76.00% | 75.20% |

## Task2 - Designing a Convolution Module for Variable Input Channels

#Baseline:
[Checkpoints - ResNet34](https://drive.google.com/file/d/17uau_f-7IzebhZIz8jcXrah0atqQPPna/view?usp=sharing)

#Q2-1:
[Checkpoints - RRDB](https://drive.google.com/file/d/1Wa5fsheFg95qoBxrXBLmUT4KASdz5yjs/view?usp=sharing)

#Q2-2:
[Checkpoints - RRDB_RNN](https://drive.google.com/file/d/1qGkMu-ePu2W2EttpEi3Em9R37So3hqLg/view?usp=sharing)

**Performance and Complexity comparison on mini-ImageNet without pretraining on ImageNet-1K. 
Mulit-Adds are calculated for a 3x224x224 input.**
| Model | Params | Multi-Adds | Forward | FLOPs | Accuracy | Precision | Recall | f1-score |
|:-----------:|:---------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [ResNet34](https://drive.google.com/file/d/17uau_f-7IzebhZIz8jcXrah0atqQPPna/view?usp=sharing) |   90.16M    | 3.66G | 59.81M | 3.41 G | 26.67% | 11.81% | 26.67% | 15.83% |
| [RRDB](https://drive.google.com/file/d/1Wa5fsheFg95qoBxrXBLmUT4KASdz5yjs/view?usp=sharing) |  0.17M  | 0.20G | 20.07M |  0.19 G | 35.11% | 44.25% | 35.11% | 34.35%  |
| [RRDB_RNN](https://drive.google.com/file/d/1qGkMu-ePu2W2EttpEi3Em9R37So3hqLg/view?usp=sharing) |  0.30M  | 0.42G | 20.07M | 0.39G | 42.67% | 46.73% | 42.67% | 41.33% |


## Usage

Download the [mini-ImageNet](https://cchsu.info/files/images.zip) dataset, unzip its, and put them on `.\images`, then

Change the path configuration for all `.py` files.

```
python train_dw.py
```


## Environment

```
conda create --name assignment python=3.8 -y
conda activate assignment
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install scikit-learn tqdm pandas torchinfo torchprofile pillow
```

## Contact

If you have any question, please feel free to contact the author via [zuw408421476@gmail.com].
