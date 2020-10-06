"""
Within this file some parameters can be tuned.
"""

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


