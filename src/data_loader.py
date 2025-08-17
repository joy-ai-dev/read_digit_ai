'''
The data loader is the first file I created. 

'''
import numpy as np
from typing import Tuple, Iterator
from tensorflow.keras.datasets import mnist
import warnings
from tensorflow.keras.utils import to_categorical

# To suppress user warnings of protobuf incompatible versions
warnings.filterwarnings("ignore", category=UserWarning)
                        
def load_mnist(flatten: bool = True,
               normalize: bool = True,
               one_hot: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Go and get MNIST data (images of digits 0-9). 
    Bring it into the program as an array of numbers.

    Args:
        flatten (bool): If True, each image is reshaped into a 1D array (e.g., 28x28 → 784). 
                        If False, images remain as 2D arrays.
        normalize (bool): If True, pixel values are scaled to the range [0, 1]. 
                          If False, pixel values remain in the range [0, 255].
        one_hot (bool): If True, labels are converted to one-hot encoded format.
                        If False, labels remain as integers (0–9).
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - Training images
            - Training labels
            - Test images
            - Test labels   

    """
    # 1.  load the MNIST dataset and return processed trained and test sets.
    (x_train, y_train) , (x_test, y_test) = mnist.load_data()

    # 2. Flatten images from 60000 x 28 x 28 to 60000 x 784. [28, 28] is turned to [784, 0]
    if flatten:
        # -1 asks NumPy to fill automatically based on total size
        x_train = x_train.reshape(x_train.shape[0], -1) 
        x_test = x_test.reshape(x_test.shape[0], -1)
        
    #3. Normalize number 0 to 255 to 0 to 1.0 decimals 
    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        
    #4. Converts 0 to 9 values to one-hot format. to_categorical() is a tensorflow function
    if one_hot:
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test
def preprocess_image(image, flatten = True, normalize = True):
    """
    Converts image to array and then normalizes and flattens it

    Args:
        image (np.ndarray): The input of image in 28 x 28 or other format
        flatten (bool): 
        normalize (bool): 
    Returns:
        np.ndarray: The preprocessed image, ready to feed into neural network. Shape (784,) or (28, 28) 
    """
    # 1. If it is a list or matrix, it is converted to a numpy array
    image = np.array(image)

    # 2. 
    if normalize:
        image = image.astype(np.float32) / 255.0

    # 3. Flattens 28 x 28 to 784 x 1 
    if flatten:
        image = image.reshape(-1)

    return image

def batch_generator(
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates batches of data for training a neural network

    Args:
        X (np.ndarray): Array of images, shapes
        Y (np.ndarray): Array of labels, shapes
        batch_size (int): Number of samples per batch
        shuffle (bool): If true, shuffles data befor batching

    Yields:
        Iterator[Tuple[np.ndarray, np.ndarray]]: A tuple for each batch 
    """
    num_samples = X.shape[0] #total number of images

    # Shuffle X and Y array
    if shuffle:
        indices = np.arrange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

    # returns batches one at a time. 
    for start in range (0, num_samples, batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        Y_batch = Y[start:end]
        yield X_batch, Y_batch
        
