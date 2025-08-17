# read_digit_ai
A simple AI program I am creating to learn how AI works and also to practice python. It will read AI digits hand written in 28 x 28 px and print the number.

# PIP INSTALLS I USED
pip install numpy
pip install typing
pip install tensorflow

# Project structure
read_digit_ai/
│
├── data/                # Store datasets (MNIST or any other)
│
├── models/              # Store saved trained models (.pth files if using PyTorch)
│
├── src/                 # All source code lives here
│   ├── __init__.py      # Makes this folder a package
│   ├── data_loader.py   # Code to load and preprocess MNIST data
│   ├── model.py         # AI model definition (neural network architecture)
│   ├── train.py         # Training loop
│   ├── evaluate.py      # Code to test/evaluate the model
│   └── predict.py       # Code to make predictions on new images
│
├── tests/               # Test scripts (optional for now)
│
├── requirements.txt     # List of dependencies (so others can install easily)
├── .gitignore           # Ignore venv, cache, datasets, etc.
├── README.md            # Project description
└── main.py              # Entry point for the program

# Functions in files
src/data_loader.py

def load_mnist(flatten: bool = True, normalize: bool = True, one_hot: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def preprocess_image(image: np.ndarray, flatten: bool = True, normalize: bool = True) -> np.ndarray

def batch_generator(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]

def train_val_split(X: np.ndarray, y: np.ndarray, val_fraction: float = 0.1, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

src/model.py

def initialize_weights(input_size: int, hidden_sizes: List[int], output_size: int, init: str = "he") -> Tuple[List[np.ndarray], List[np.ndarray]]

def activation_relu(x: np.ndarray) -> np.ndarray

def activation_softmax(x: np.ndarray) -> np.ndarray

def forward_pass(X: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray]) -> Dict[str, List[np.ndarray]]

def compute_loss(predictions: np.ndarray, targets: np.ndarray) -> float

def backward_pass(a_values: List[np.ndarray], z_values: List[np.ndarray], weights: List[np.ndarray], targets: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]

def update_weights(weights: List[np.ndarray], biases: List[np.ndarray], dW: List[np.ndarray], db: List[np.ndarray], lr: float) -> Tuple[List[np.ndarray], List[np.ndarray]]

def save_weights(weights: List[np.ndarray], biases: List[np.ndarray], path: str) -> None

def load_weights(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]

def predict_proba(X: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray

def predict_classes(X: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray

src/train.py

def train(input_size: int, hidden_sizes: List[int], output_size: int, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, epochs: int = 10, batch_size: int = 64, lr: float = 0.01, seed: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]

def train_epoch(weights: List[np.ndarray], biases: List[np.ndarray], X: np.ndarray, y: np.ndarray, batch_size: int, lr: float) -> Tuple[float, float]

src/evaluate.py

def evaluate(weights: List[np.ndarray], biases: List[np.ndarray], X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 64) -> Dict[str, float]

def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> np.ndarray

src/predict.py

def predict_image(image: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray], preprocess: bool = True) -> int

def predict_batch(X: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray

src/utils.py

def accuracy(preds: np.ndarray, labels: np.ndarray) -> float

def to_one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray

def save_npz(path: str, weights: List[np.ndarray], biases: List[np.ndarray]) -> None

def load_npz(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]

src/main.py

def run_train_config(config_path: str) -> None

def run_predict_image(image_path: str, weights_path: str) -> None