"""
Within this file some parameters can be tuned.
"""

# Paths
PLOTS_PATH = "outputs/plots/"
DATASET_PATH = "datasets/images/"
LABELS_PATH = "datasets/labels/img_annotations.json"
CHECKPOINTS_PATH = "outputs/checkpoints/saved_model/saved_model"
TEST_SET_FILE_PATH = "datasets/test_dataset_array.npy"
TEST_IMAGE_PATH = "datasets/images/149762_2018_04_19_08_31_57_347139.jpg"

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
ENABLE_CAM_COMPUTATION = True



