import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models, saving

# Custom metric/loss
@saving.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def iou_metric(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / union

# Load model
MODEL_PATH = './models/unet_license_plate_v6_best.keras'  # update path/version as needed
model = models.load_model(MODEL_PATH, custom_objects={
    'dice_loss': dice_loss,
    'iou_metric': iou_metric
})

# Image preprocessing
def preprocess_image(img_path, size=(256, 256)):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, size)
    img_norm = img_resized / 255.0
    return img, np.expand_dims(img_norm, axis=0)

# Result visualization
def show_result(original, prediction, save_path=None):
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(original); plt.title("Original Image")
    plt.subplot(1, 2, 2); plt.imshow(prediction.squeeze(), cmap='gray'); plt.title("Predicted Mask")
    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test.py path/to/image.jpg")
        sys.exit(1)

    input_path = sys.argv[1]
    assert os.path.exists(input_path), f"File not found: {input_path}"

    img, input_tensor = preprocess_image(input_path)
    pred = model.predict(input_tensor)[0]

    os.makedirs('./custom_results', exist_ok=True)
    out_path = os.path.join('./custom_results', os.path.basename(input_path).replace('.jpg', '_result.png'))
    show_result(img, pred, out_path)
