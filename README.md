# Home assignment - Tomato allergies

Open domain image classification.
In this repository can be found a Deep Learning solution for the tomato detection problem based on a Transfer Learning methodology using a ResNet50V2 as the base model.

## Requirements

Some external libraries have been used to run the program:
 - TensorFlow 2.3.1
 - CUDA 10.1
 - cuDNN 7.6.5
 - matplotlib
 - numpy
 - scikit-learn
 - scipy
 - OpenCV 4.4.0.44

## Usage

The program is divided into 3 python files, each of one according to its respective task: [train.py](https://github.com/alvarobasi/home-assignment/blob/master/train.py), [eval.py](https://github.com/alvarobasi/home-assignment/blob/master/eval.py) and [test.py](https://github.com/alvarobasi/home-assignment/blob/master/test.py).

Every parameter used in each program can be tuned, either by placing arguments or by changing them directly from the [config.py](https://github.com/alvarobasi/home-assignment/blob/master/config.py) file.

```python
# Paths
PLOTS_PATH = "outputs/plots/"
DATASET_PATH = "datasets/images/"
LABELS_PATH = "datasets/labels/img_annotations.json"
CHECKPOINTS_PATH = "outputs/checkpoints/saved_model/saved_model"
TEST_SET_FILE_PATH = "test_dataset_array.npy"
TEST_IMAGE_PATH = "datasets/images/149762_2018_04_19_08_31_57_347139.jpg"

# Training hyperparameters
EPOCHS = 5
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 0.001
FINE_TUNING_LR = 1e-5
IMAGE_SHAPE = (600, 600, 3)
FINE_TUNE_LAYERS = 70
BALANCED_DATASET = True
MIXED_PRECISION = True
XLA = False
ENABLE_CAM_COMPUTATION = True
ENABLE_SHOW_EVAL_IMAGES = True
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
 
The program will print some information about the metrics states during the training process. As a Transfer Learning technique has been used, the training process will be automatically executed twice, the first one with `EPOCHS = 5` for fiting the new classification layers and the second one during 3 times the `EPOCHS = 5` value for fiting the classification layers along with the selected convolutional layers from the base model, which is configured in the paramameter `FINE_TUNE_LAYERS = 70`. At the end of the process, the model will be evaluated without printing the images and the error rate obtained in the test set will be printed. Some hyperparameters used during the process can be found in the [config.py](https://github.com/alvarobasi/home-assignment/blob/master/config.py) file.

Finally, as the dataset is quite unbalanced, the parameter `BALANCED_DATASET = True` balances the dataset by oversampling the positive images (tomato present) in order to match the number of positive cases with the negative ones. In the case this is disabled, a tuple of weights is computed and passed to the training process so that the minority class will gain more relevance against the mayority class.

## Evaluating

To evaluate the model,  [eval.py](https://github.com/alvarobasi/home-assignment/blob/master/eval.py) file should be executed. These are the following arguments that can be introduced:
- `-tds`: Path to the test dataset file. This file is called by default `test_dataset_array.npy` an is generated during the execution of the [train.py](https://github.com/alvarobasi/home-assignment/blob/master/train.py) file so that the same test data can be retreived later.
- `-w`: Path to model weights to be loaded.
- `-cam`: Enable the computation and overlay of the class activation map on the evaluation images.
- `-clas_evl`: Selects the evaluation of the classification performance of the model.
- `-loc_evl`: Selects the evaluation of the object localization performance of the model.

Regarding the evaluation of the classification performance, it should be noted that the parameter `ENABLE_SHOW_EVAL_IMAGES = True` located in the [config.py](https://github.com/alvarobasi/home-assignment/blob/master/config.py) file enables the evaluation process to show a plot with 4 test images and their predictions for each batch `EVAL_BATCH_SIZE = 4`. Furthermore, it is possible to set the argument `-cam True` (or the parameter `ENABLE_CAM_COMPUTATION` within the [config.py](https://github.com/alvarobasi/home-assignment/blob/master/config.py)) so that the CAM matrices for each of the test batch images are calculated and overlayed over these 4 test images. This is only possible if and only if `ENABLE_SHOW_EVAL_IMAGES` and `ENABLE_CAM_COMPUTATION` are enabled together. In case `ENABLE_SHOW_EVAL_IMAGES` is disabled, the evaluation process will only output the classification error rate in the test set, even if `ENABLE_CAM_COMPUTATION` is enabled. Finally, the paramter `EVAL_BATCH_SIZE = 4` controls the size of the test batch, that is, the number of images that will be processed at the same time. It is set with such a low number as the CAM upsampling is computationally intensive and two big models (the base model and the entire model) are used separatelly in order to perform the CAM computation, which take a lot of VRAM memory. At least in my 6GB VRAM graphics card wasn't possible to raise that number.

An example of what the classification evaluation returns for each batch when enabled CAM calculation and images printing can be seen in the following figure:

![result_eval.png](https://github.com/alvarobasi/home-assignment/blob/master/outputs/result_eval.png)

On the other hand, the evaluation of the localization performance will draw in green on the main image the ground truth bounding boxes, which are stored inside the `test_dataset_array.npy` file along with the corresponding image path and label. Furthermore, the bounding boxes predicted by the model will also be drawn in blue so that they can be visually compared with the groung truths in order to evaluate the performance.

An example of what the evaluation of the localization performance will output each iteration is shown in the following figure:

![results_boxes.png](https://github.com/alvarobasi/home-assignment/blob/master/outputs/results_boxes.png)


## Testing

To test the model with a single image,  [test.py](https://github.com/alvarobasi/home-assignment/blob/master/test.py) file should be executed. These are the following arguments that can be introduced:
- `-i`: Path to the image path.
- `-w`: Path to model weights to be loaded.
- `-cam`: Enable the computation and overlay of the class activation map on the image.

It will print the selected image along with its prediction. If `-cam` is set to `True`, the CAM will be overlayed on the printed image. The function will also return `True` or `False` depending on the result.

When testing an image and `-cam` is enabled, the resulting CAM will be overlayed on the top of the printed image as follows:

![result_camp.png](https://github.com/alvarobasi/home-assignment/blob/master/outputs/result_cam.png)

## Results

The results obtained using the current parameters and configurations are the following:

Before fine tuning layers: 

![pre_fine_tuning_plot.png](https://github.com/alvarobasi/home-assignment/blob/master/outputs/plots/pre_fine_tuning_plot.png)

After fine tuning layers:

![fine_tuning_plot.png](https://github.com/alvarobasi/home-assignment/blob/master/outputs/plots/fine_tuning_plot.png)

## Assignment 2 Results
The results of this section are shown in the [Evaluating](https://github.com/alvarobasi/home-assignment/blob/master/README.md#evaluating) and  [Testing](https://github.com/alvarobasi/home-assignment/blob/master/README.md#testing).


## Credits

https://keras.io/guides/transfer_learning/

https://www.tensorflow.org/tutorials/images/transfer_learning

https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751

https://towardsdatascience.com/demystifying-convolutional-neural-networks-using-class-activation-maps-fe94eda4cef1

https://arxiv.org/pdf/1512.04150.pdf
