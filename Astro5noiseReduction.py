import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# Constants
BASE_DIR = '/Users/aliasgari/Downloads/Astroinformatics'
CSV_FILE = '2class_list.csv'  # CSV file with image names and labels
IMAGE_SIZE = (300, 300)  # Target image size
BATCH_SIZE = 32
EPOCHS = 10
NOISE_REDUCTION_METHODS = ['Bilateral_Filtering', 'Median_Filtering', 
                           'Non-local Means Denoising', 'Total_Variation_Denoising', 
                           'Wavelet_Transform_Denoising']

# Load the CSV file to get image names and labels
df = pd.read_csv(CSV_FILE)
image_names = df['CATAID'].astype(str).tolist()
labels = df['TARGET'].tolist()

# Function to load images into a NumPy array
def load_images(image_names, image_dir):
    images = []
    for name in image_names:
        img_path = os.path.join(image_dir, name + '_giH.png')
        try:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(images)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the ResNet50 model
def create_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Loop through each noise reduction method
for method in NOISE_REDUCTION_METHODS:
    print(f"Processing {method} images in {os.path.join(BASE_DIR, method)}...")

    # Load images and labels
    images = load_images(image_names, os.path.join(BASE_DIR, method))
    labels = np.array(labels)

    # Split the data into training and validation sets
    split_index = int(0.8 * len(images))
    train_images, val_images = images[:split_index], images[split_index:]
    train_labels, val_labels = labels[:split_index], labels[split_index:]

    # Calculate class weights to handle class imbalance
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    # Create the ResNet50 model
    resnet50_model = create_resnet50_model()

    # Train the ResNet50 model
    history_resnet50 = resnet50_model.fit(
        datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(val_images, val_labels),
        class_weight=class_weights
    )

    # Fine-tuning: Unfreeze some layers in the base model
    resnet50_model.layers[0].trainable = True

    # Re-compile the model with a lower learning rate
    resnet50_model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    # Continue training the model
    history_resnet50_finetune = resnet50_model.fit(
        datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(val_images, val_labels),
        class_weight=class_weights
    )

    # Evaluate the fine-tuned ResNet50 model
    loss_resnet50, accuracy_resnet50 = resnet50_model.evaluate(val_images, val_labels)
    print(f'{method} - ResNet50 Fine-tuned Validation Loss: {loss_resnet50}')
    print(f'{method} - ResNet50 Fine-tuned Validation Accuracy: {accuracy_resnet50}')

    # Generate classification report for the fine-tuned ResNet50 model
    pred_labels_resnet50 = (resnet50_model.predict(val_images) > 0.5).astype("int32")
    print(f'Classification Report for {method}:')
    print(classification_report(val_labels, pred_labels_resnet50, target_names=['Elliptical', 'Other']))

    # Confusion Matrix
    cm = confusion_matrix(val_labels, pred_labels_resnet50)
    print(f'Confusion Matrix for {method}:')
    print(cm)

    # Save the model
    model_save_path = os.path.join(BASE_DIR, f'model_{method}.keras')
    resnet50_model.save(model_save_path)
    print(f'Model for {method} saved at {model_save_path}')

    # Plot training & validation accuracy and loss
    def plot_history(history, history_finetune, method):
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.plot(history_finetune.history['accuracy'], label='Train Accuracy (Finetune)')
        plt.plot(history_finetune.history['val_accuracy'], label='Val Accuracy (Finetune)')
        plt.title(f'{method} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        accuracy_plot_path = os.path.join(BASE_DIR, f'{method}_accuracy.png')
        plt.savefig(accuracy_plot_path)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.plot(history_finetune.history['loss'], label='Train Loss (Finetune)')
        plt.plot(history_finetune.history['val_loss'], label='Val Loss (Finetune)')
        plt.title(f'{method} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        loss_plot_path = os.path.join(BASE_DIR, f'{method}_loss.png')
        plt.savefig(loss_plot_path)

        plt.show()

    plot_history(history_resnet50, history_resnet50_finetune, method)
