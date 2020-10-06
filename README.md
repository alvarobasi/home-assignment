# Home assignment - Tomato allergies

Open domain image classification.
In this repository can be found a Deep Learning solution for the tomato detection problem based on a Transfer Learning methodology using a ResNet50V2 as the base model.

## Requirements

Some external libraries have been used to run the program:
 - TensorFlow 2.3.1
 - matplotlib
 - numpy
 - scikit-learn

## Usage

The program is divided into 3 python files, each of one according to its respective task: [train.py](https://github.com/alvarobasi/home-assignment/blob/master/train.py), [eval.py](https://github.com/alvarobasi/home-assignment/blob/master/eval.py) and [test.py](https://github.com/alvarobasi/home-assignment/blob/master/test.py).

Every parameter used in each program can be tuned, either by placing arguments or by changing them directly from the [config.py](https://github.com/alvarobasi/home-assignment/blob/master/config.py) file.

```python
# Paths
PLOTS_PATH = "outputs/plots/"
DATASET_PATH = "datasets/images/"
LABELS_PATH = "datasets/labels/img_annotations.json"
CHECKPOINTS_PATH = "outputs/checkpoints/saved_model/saved_model"
TEST_SET_FILE_PATH = "datasets/test_dataset_array.npy"
TEST_IMAGE_PATH = "datasets/images/4dda082e4a1d820f7cc32f5cd9dc79be.jpeg"

# Training hyperparameters
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
FINE_TUNING_LR = 1e-5
IMAGE_SHAPE = (600, 600, 3)
FINE_TUNE_LAYERS = 70
BALANCED_DATASET = True
MIXED_PRECISION = True
XLA = False
```

## Training

To train the model, the [train.py](https://github.com/alvarobasi/home-assignment/blob/master/train.py) file should be executed. These are the following arguments that can be introduced:
 - `-d`: Path to the dataset directory. 
 - `-a`: Path to the img_annotations.json file.
 - `-e`: # of epochs to initially train the network for. For the Fine tuning step, it will be 3 times the selected epoch value, as it is the most important part of the training process in this case.
 - `-bs`: Size of the mini-batch to be used in the training process.
 - `-lr`: Set learning rate value.
 - `-ftl`: # of last layers to fine tune in the base model used for the Transfer Learning task.
 - `-fp16`: Enables mixed_precision training. Only available for Volta, Turing and Ampere GPUs.
 - `-xla`: Enables XLA compilation mode in order to accelerate training. Linux only.
 
The program will print some information about the metrics states during the training process. As a Transfer Learning technique has been used, the training step will be automatically executed twice, the first one for fiting the new classification layers and the second one for fiting the classification layers along with the selected convolutional layers from the base model. At the end of the process, the model will be evaluated and the error rate obtained in the test set will be printed.

## Evaluating

To evaluate the model,  [eval.py](https://github.com/alvarobasi/home-assignment/blob/master/eval.py) file should be executed. These are the following arguments that can be introduced:
- `-tds`: Path to the test dataset file. This file is called by default `test_dataset_array.npy` an is generated during the execution of the [train.py](https://github.com/alvarobasi/home-assignment/blob/master/train.py) file so that this data can be retreived later.
- `-w`: Path to model weights to be loaded.

Once the model is trained, the test dataset paths and labels are saved in a file called `test_dataset_array.npy`. This file is required to be entered in the evaluation step.

## Testing

To test the model with a single image,  [test.py](https://github.com/alvarobasi/home-assignment/blob/master/test.py) file should be executed. These are the following arguments that can be introduced:
- `-i`: Path to the test dataset file. This file is called by default `test_dataset_array.npy` an is generated during the execution of the [train.py](https://github.com/alvarobasi/home-assignment/blob/master/train.py) file so that this data can be retreived later.
- `-w`: Path to model weights to be loaded.
- `-cam`: Enable the computation and overlay of the class activation map on the image.

It will print the selected image along with its prediction. If `-cam` is set to `True`, the CAM will be overlayed on the printed image. The function will also return `True` or `False` depending on the result.

## Results

The results obtained using the current parameters and configurations are the following:

Before fine tuning layers: 

![pre_fine_tuning_plot.png](https://github.com/alvarobasi/home-assignment/blob/master/outputs/plots/pre_fine_tuning_plot.png)

After fine tuning layers:

![fine_tuning_plot.png](https://github.com/alvarobasi/home-assignment/blob/master/outputs/plots/fine_tuning_plot.png)

## Assignment 2 Results

When testing an image and `-cam` is enabled, the resulting CAM will be overlayed on the top of the printed image as follows:

![result_camp.png](https://github.com/alvarobasi/home-assignment/blob/master/outputs/result_cam.png)

## Credits

https://keras.io/guides/transfer_learning/

https://www.tensorflow.org/tutorials/images/transfer_learning

https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751

https://towardsdatascience.com/demystifying-convolutional-neural-networks-using-class-activation-maps-fe94eda4cef1

https://arxiv.org/pdf/1512.04150.pdf
