"""
Within this file some parameters can be tuned.
"""
# Paths
PLOTS_PATH = "outputs/plots/"
DATASET_PATH = "datasets/images/"
LABELS_PATH = "datasets/labels/"
CHECKPOINTS_PATH = "outputs/checkpoints/"
TEST_SET_FILE_PATH = "datasets/test_dataset_array.npy"

# Training parameters
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
FINE_TUNING_LR = 1e-5
IMAGE_SHAPE = (600, 600, 3)
FINE_TUNE_LAYERS = 70
BALANCED_DATASET = True
MIXED_PRECISION = True
XLA = False



