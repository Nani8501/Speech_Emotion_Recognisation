# Speech Emotion Recognition Model

This project, led by Me, is a major college project aimed at developing a Speech Emotion Recognition (SER) model using audio data. The model is trained to recognize emotions such as Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

## Overview

The project spanned from June 2023 to November 2023 and served as a valuable learning experience in the field of machine learning and audio signal processing. This README.md provides an in-depth overview of the project, its structure, usage, and the underlying model architecture.

## Project Details

### Dataset

The dataset used for training the model was obtained from [Kaggle](https://www.kaggle.com/code/vafaknm/speech-emotion-recognition-model/input). Jagadeesh Kokkula modified the original dataset to have separate folders for each emotion, making it more suitable for the SER model.

#### Emotions
- Angry
- Disgusted
- Fearful
- Happy
- Neutral
- Sad
- Surprised

### File Structure

```python
import os
import librosa
import numpy as np
import tensorflow
from tensorflow import keras
from sklearn.model_selection import train_test_split
# ... (rest of the code)
```

- `train.py`: Python script for training the emotion recognition model.
- `predict.py`: Python script for predicting emotion from an audio file.
- `emotion_model.h5`: Trained model saved in the Hierarchical Data Format (HDF5) file.
- `utils.py`: Utility functions for data loading, preprocessing, and model building.
- `visualize_results.py`: Script to visualize training history and model evaluation results.
- `requirements.txt`: List of dependencies for installing required libraries.

#### Data Directory
- `Emotions/`: Folder containing subfolders for each emotion with audio files.

## Usage

1. Install the required libraries by running:

```bash
pip install -r requirements.txt
```

2. Train the model using the following command:

```bash
python train.py
```

The training script supports various hyperparameters that can be configured for experimentation.

3. Predict emotion from an audio file:

```bash
python predict.py
```

You will be prompted to enter the path to the audio file.

## Model Architecture

The emotion recognition model is a simple neural network with fully connected layers. Batch normalization and dropout are applied to prevent overfitting. Jagadeesh Kokkula implemented the model architecture and fine-tuned it for optimal performance.

## Results

The training history and model evaluation results can be visualized by running:

```bash
python visualize_results.py
```

This script generates plots for accuracy and loss during training.

## Data Augmentation

The training data is augmented using the `DataGenerator` class in `utils.py`. This class implements a custom data generator that allows for real-time data augmentation.

## Project Duration

The project was initiated in June 2023 and successfully completed by November 2023.

## Personal Information

- **Name:** Jagadeesh Kokkula
- **LinkedIn:** [Jagadeesh Kokkula](https://www.linkedin.com/in/jagadeeshkokkula/)
- **Website:** [Jagadeesh Kokkula's Website](https://nani8501.github.io/new.github.io/)
- **GitHub:** [Jagadeesh Kokkula on GitHub](https://github.com/Nani8501)

## Contribution 

Contributions to the project are welcome! If you find any issues or have suggestions for improvements, please create an issue or submit a pull request.
