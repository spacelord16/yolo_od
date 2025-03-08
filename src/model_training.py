import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def build_simple_yolo(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    images = np.load('/mnt/d/Yolo_scratch/yolo_od/data/processed/images.npy')
    labels = np.load('/mnt/d/Yolo_scratch/yolo_od/data/processed/labels.npy', allow_pickle=True)

    # Example dummy label preprocessing; customize as needed
    y_train = np.random.randint(0, 2, (len(images), 20)) # adjust according to actual data

    model = build_simple_yolo(input_shape=(224,224,3), num_classes=20)
    model.fit(images, y_train, epochs=10, batch_size=32, validation_split=0.2)

    model.save('/mnt/d/Yolo_scratch/yolo_od/models/yolo_model.h5')
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train_model()
