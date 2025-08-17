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