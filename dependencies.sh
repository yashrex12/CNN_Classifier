#!/bin/bash

# Install dependencies using pip
pip install numpy pandas torchvision matplotlib scikit-learn

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "All dependencies installed successfully."
else
    echo "Failed to install dependencies."
fi
