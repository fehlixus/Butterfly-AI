import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import random
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


local_dir = os.path.dirname(__file__)
train_data = glob(os.path.join(local_dir, "train", "*.jpg"))
test_data = glob(os.path.join(local_dir, 'test', "*.jpg"))

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(.2),
    tf.keras.layers.RandomHeight(.2),
    tf.keras.layers.RandomZoom(.2),
])

random_img = random.choice(train_data)  # choose a random image
img = mpimg.imread(random_img).astype('float32') / 255  # load and normalize the random Image

labels_df = pd.read_csv('Training_set.csv')

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df['label'])

image_dir = os.path.join(local_dir, "train")


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# data generators
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=image_dir,
    x_col="filename",
    y_col="label",
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=32
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    directory=image_dir,
    x_col="filename",
    y_col="label",
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=32
)

# Get number of classes
num_classes = len(labels_df['label'].unique())
6
from tensorflow.keras import layers, models

from tensorflow.keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu', kernel_regularizer='l2')(x)
predictions = layers.Dense(num_classes, activation='softmax')(x)

model_path = os.path.join(local_dir, "model")
model = tf.keras.models.load_model(model_path)



# compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)
