# classification.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import ModelCheckpoint

def load_data(features_path, groundtruth_path):
    features = np.load(features_path)
    groundtruth = np.genfromtxt(groundtruth_path, delimiter=',', skip_header=1, usecols=1)
    return features, groundtruth

def preprocess_labels(labels):
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    return labels

def train_vgg16_model(features, labels):
    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    base_model = tf.keras.applications.VGG16(include_top=False, input_shape=(256, 256, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('models/vgg16_model.h5', monitor='val_loss', save_best_only=True, mode='min')
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, callbacks=[checkpoint])

if __name__ == "__main__":
    features, groundtruth = load_data('data/features.npy', 'data/ISIC_2019_Training_GroundTruth.csv')
    labels = preprocess_labels(groundtruth)
    train_vgg16_model(features, labels)
