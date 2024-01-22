# Speech Emotion Recognition Model by Jagadeesh Kokkula

This project, led by **Jagadeesh Kokkula**, represents a significant undertaking in the field of Speech Emotion Recognition (SER). The primary objective of this project, spanning from June 2023 to November 2023, was to design and develop a robust model capable of identifying emotions such as Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised from audio data.

## Background

As part of my college major project, I embarked on a self-learned journey to delve into the intricate domain of SER. The motivation behind this endeavor was to explore the fusion of machine learning, audio processing, and emotional intelligence, with the ultimate goal of contributing to advancements in human-computer interaction.

## Dataset

The foundation of the project lies in a meticulously curated dataset sourced from [Kaggle](https://www.kaggle.com/code/vafaknm/speech-emotion-recognition-model/input). Noteworthy is the transformation of the dataset, where each emotion now resides in its dedicated folder, a modification introduced to enhance data organization and accessibility.

### Emotions
- Angry
- Disgusted
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Project Structure

The project boasts a comprehensive structure, reflecting the commitment to clarity and organization. Key components include:

- `train.py`: The script responsible for training the emotion recognition model.
- `predict.py`: A script facilitating emotion prediction from an audio file.
- `emotion_model.h5`: The culmination of the project - a trained model stored in HDF5 format.
- `utils.py`: A utility script encapsulating functions for data loading, preprocessing, and model building.
- `visualize_results.py`: A script for visualizing training history and model evaluation results.
- `requirements.txt`: A file listing dependencies for straightforward installation.

### Data Directory
- `Emotions/`: A dedicated folder housing subfolders for each emotion along with their respective audio files.

## Usage

### Installation

To replicate or extend the project, install the necessary dependencies using the following command:
```bash
pip install -r requirements.txt
```

### Training the Model

The model training process is facilitated through the `train.py` script, offering flexibility with various hyperparameter configurations.

```bash
python train.py
```

### Emotion Prediction

Predict emotions from an audio file using the `predict.py` script. Simply execute the following command and provide the path to the audio file when prompted:

```bash
python predict.py
```

## Model Architecture

The heart of the project lies in the model architecture. A neural network with fully connected layers forms the backbone, augmented by batch normalization and dropout layers to mitigate overfitting. The architecture is customizable through the `build_model` function in `utils.py`.

## Results and Visualization

To gain insights into the training process, utilize the `visualize_results.py` script, generating plots for accuracy and loss during model training.

```bash
python visualize_results.py
```

## Data Augmentation

Augmenting the training data is a critical step, enhancing the model's ability to generalize. The custom `DataGenerator` class, defined in `utils.py`, enables real-time data augmentation.

## Personal Achievements

This project is a testament to my commitment to self-learning and perseverance. Undertaken independently, I navigated the challenges of understanding complex concepts, implementing algorithms, and debugging issues. This project showcases my ability to independently conceive, design, and execute a machine learning project.

## Connect with Me

- **LinkedIn**: [Jagadeesh Kokkula](https://www.linkedin.com/in/jagadeeshkokkula/)
- **GitHub**: [Nani8501](https://github.com/Nani8501)
- **Website**: [Personal Website](https://nani8501.github.io/new.github.io/)

## Acknowledgments

I extend my gratitude to the open-source community, Kaggle, and various online learning platforms for providing invaluable resources that facilitated my learning journey.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This project, born out of curiosity and dedication, is a testament to the power of self-directed learning. From conceptualization to implementation, it represents a significant milestone in my academic journey, demonstrating my capability to tackle complex problems and contribute meaningfully to the field.
