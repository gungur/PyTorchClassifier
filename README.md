# FashionMNIST Classifier with PyTorch

This repository contains a PyTorch implementation for classifying FashionMNIST images. The program includes functions for data loading, model building, training, evaluation, and prediction.

## Features

- Data loading and preprocessing for FashionMNIST dataset
- Neural network model with customizable architecture
- Training with progress tracking (accuracy and loss)
- Model evaluation on test data
- Prediction with probability outputs for top 3 classes

## Requirements

- Python 3.x
- PyTorch
- torchvision

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install torch torchvision
   ```

## Usage

### Importing the Module
```python
import intro_pytorch
```

### Loading Data
```python
train_loader = intro_pytorch.get_data_loader(training=True)
test_loader = intro_pytorch.get_data_loader(training=False)
```

### Building the Model
```python
model = intro_pytorch.build_model()
```

### Training the Model
```python
criterion = nn.CrossEntropyLoss()
intro_pytorch.train_model(model, train_loader, criterion, T=5)  # T is number of epochs
```

### Evaluating the Model
```python
intro_pytorch.evaluate_model(model, test_loader, criterion, show_loss=True)
```

### Making Predictions
```python
test_images = next(iter(test_loader))[0]
intro_pytorch.predict_label(model, test_images, index=1)  # Predict for image at index 1
```

## Example Output

When running the main script, you'll see output similar to:

```
Train Epoch: 0 Accuracy: 1234/60000(85.67%) Loss: 0.123
Train Epoch: 1 Accuracy: 2345/60000(88.90%) Loss: 0.098
...
Average loss: 0.1234
Accuracy: 87.65%
T-shirt/top: 95.23%
Trouser: 3.45%
Pullover: 1.32%
```

## File Structure

- `intro_pytorch.py`: Main Python script containing all the functionality
- `metadata.yml`: Configuration file (for internal use)
