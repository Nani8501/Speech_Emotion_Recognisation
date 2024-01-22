import os
import librosa
import numpy as np
import tensorflow
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from keras import layers, models, callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import keras
import matplotlib.pyplot as plt
from keras.utils import Sequence


def extract_mfcc(sr, audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed
def load_and_preprocess_data(data_dir, emotions):
    X = []
    Y = []

    for emotion in emotions:
        print(emotion,"loading..")
        emotion_dir = os.path.join(data_dir, emotion)
        for filename in os.listdir(emotion_dir):
            file_path = os.path.join(emotion_dir, filename)
            try:
                audio, sr = librosa.core.load(file_path, res_type='kaiser_best')
                features = extract_mfcc(sr, audio)
                X.append(features)
                Y.append(emotion)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    X = np.array(X)
    Y = np.array(Y)

    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    Y_one_hot = to_categorical(Y_encoded, num_classes=len(emotions))
    print("DataSet Loading Compleate!\n\n")
    return X, Y_one_hot

class DataGenerator(Sequence):
    def __init__(self, X, Y, batch_size, augmentation=False):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.augmentation = augmentation

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        batch_X = self.X[start_idx:end_idx]
        batch_Y = self.Y[start_idx:end_idx]

        if self.augmentation:
            # Apply your data augmentation techniques here if needed
            pass

        return batch_X, batch_Y

# Modify your data augmentation function
def data_augmentation(X_train, Y_train, X_val, Y_val, batch_size=32):
    print("\nPreparing Data Agumentation")

    train_generator = DataGenerator(X_train, Y_train, batch_size, augmentation=True)
    val_generator = DataGenerator(X_val, Y_val, batch_size, augmentation=False)

    return train_generator, val_generator

#Building Model
def build_model(input_shape, num_classes, dropout_rate=0.5, l2_regularization=0.01):
    model = models.Sequential()

    # Assuming input_shape is appropriate for your data
    model.add(layers.Dense(256, input_shape=(input_shape,), kernel_regularizer=keras.regularizers.l2(l2_regularization)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))  # Add activation function after BatchNormalization
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(128, kernel_regularizer=keras.regularizers.l2(l2_regularization)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(X_train, Y_train, X_val, Y_val, epochs=50, batch_size=32, validation_split=0.2):
    print("\nTraining Model:")
    model = build_model(X_train.shape[1], Y_train.shape[1])

    # Early stopping callback
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    X_train_2d = X_train.reshape(X_train.shape[0], X_train.shape[1])
    # In the training loop
    X_val_2d = X_val.reshape(X_val.shape[0], X_val.shape[1])

    # In the training loop
    train_generator, val_generator = data_augmentation(X_train_2d, Y_train, X_val_2d, Y_val, batch_size=batch_size)

    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[early_stopping])


#   history = model.fit(train_generator, epochs=epochs, validation_data=(X_val, Y_val), callbacks=[early_stopping])

    return model, history

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='accuracy', c='b')
    plt.plot(history.history['val_accuracy'], label='val_accuracy', c='r')
    plt.legend()
    plt.show()

# Set the path to your dataset
data_dir = "Emotions"

# List of emotions
emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
print(emotions)
# Load and preprocess the data
X, Y = load_and_preprocess_data(data_dir, emotions)

# Shuffle the data
X, Y = shuffle(X, Y, random_state=42)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Hyperparameters
epochs = 50
batch_size = 32
validation_split = 0.2

# Train the model
model, history = train_model(X_train, Y_train, X_val, Y_val, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

# Plot training history
plot_training_history(history)

# Save the model
model.save('emotion_model.h5')
