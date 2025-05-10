import os
import numpy as np
import tensorflow as tf
from keras import preprocessing, callbacks, utils
from unet import unet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Set paths to your dataset
dataset_dir = "dataset"  # Replace with your dataset path
image_dir = os.path.join(dataset_dir, "images")  # Folder with input images
mask_dir = os.path.join(dataset_dir, "masks")  # Folder with generated masks
plots_dir = "plots"  # Folder to save training plots
models_dir = "models"  # Folder to save the trained model
predicts_dir = "predicts"  # Folder to save predicted masks

# Ensure directories exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(predicts_dir, exist_ok=True)

# Function to load images and masks
def load_data(image_dir, mask_dir, img_size=(256, 256)):
    images = []
    masks = []

    for split in ['train', 'valid', 'test']:  # Load all splits
        image_split_dir = os.path.join(image_dir, split)
        mask_split_dir = os.path.join(mask_dir, split)
        
        for filename in os.listdir(image_split_dir):
            if filename.endswith(".png"):  # Assuming image files are .png
                img_path = os.path.join(image_split_dir, filename)
                mask_path = os.path.join(mask_split_dir, filename)

                # Load image and mask
                image = preprocessing.image.load_img(img_path, target_size=img_size)
                image = preprocessing.image.img_to_array(image) / 255.0  # Normalize the image
                
                mask = preprocessing.image.load_img(mask_path, target_size=img_size, color_mode='grayscale')
                mask = preprocessing.image.img_to_array(mask)  # Shape: (height, width, 1)
                mask = utils.to_categorical(mask, num_classes=36)  # Convert mask to categorical

                images.append(image)
                masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)
    
    return images, masks

# Load dataset
X, y = load_data(image_dir, mask_dir)

# Split dataset into training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Build U-Net model
model = unet(input_shape=(256, 256, 3), num_classes=36)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Versioning: Create a unique model version folder
timestamp = time.strftime("%Y%m%d-%H%M%S")
versioned_model_dir = os.path.join(models_dir, f"unet_v{timestamp}")
os.makedirs(versioned_model_dir, exist_ok=True)

# Define callbacks
checkpoint = callbacks.ModelCheckpoint(os.path.join(versioned_model_dir, 'unet_best_model.h5'), save_best_only=True, verbose=1)
early_stop = callbacks.EarlyStopping(patience=10, verbose=1)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=16,
    epochs=50,
    callbacks=[checkpoint, early_stop]
)

# Save the final model
model.save(os.path.join(versioned_model_dir, 'unet_final_model.h5'))

# Plot training history
def plot_history(history, save_path):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"training_history_{timestamp}.png"))
    plt.close()

# Plot and save the training history
plot_history(history, os.path.join(plots_dir, f"training_history_{timestamp}.png"))
