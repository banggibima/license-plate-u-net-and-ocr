import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from keras import saving

# CONFIG 
DATASET_PATH = './dataset'

# LOSS 
@saving.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# CUSTOM METRIC 
def iou_metric(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / union

# DATA UTILS 
def load_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img

def load_mask(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def preprocess_image(img, size=(256, 256)):
    return cv2.resize(img, size) / 255.0

def load_dataset(image_dir, mask_dir, size=(256, 256)):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    images, masks = [], []

    for img_path, msk_path in zip(image_paths, mask_paths):
        img = preprocess_image(load_image(img_path), size)
        msk = cv2.resize(load_mask(msk_path), size)
        msk = np.expand_dims((msk > 0).astype(np.float32), axis=-1)
        images.append(img)
        masks.append(msk)

    return np.array(images), np.array(masks)

# MODEL 
def build_unet(input_size=(256, 256, 3), dropout_rate=0.3):
    def conv_block(x, filters):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    def attention_block(x, g, inter_channels):
        theta_x = tf.keras.layers.Conv2D(inter_channels, 1)(x)
        phi_g = tf.keras.layers.Conv2D(inter_channels, 1)(g)
        add = tf.keras.layers.Add()([theta_x, phi_g])
        act = tf.keras.layers.Activation('relu')(add)
        psi = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(act)
        return tf.keras.layers.Multiply()([x, psi])

    def up_block(x, skip, filters):
        x = tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding='same')(x)
        attn = attention_block(skip, x, filters // 2)
        x = tf.keras.layers.Concatenate()([x, attn])
        x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        return x

    inputs = tf.keras.Input(shape=input_size)

    # Downsampling
    c1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = tf.keras.layers.MaxPooling2D()(c4)

    # Bottleneck + Dropout
    c5 = conv_block(p4, 512)
    c5 = tf.keras.layers.Dropout(dropout_rate)(c5)

    # Upsampling + Attention
    u6 = up_block(c5, c4, 256)
    u7 = up_block(u6, c3, 128)
    u8 = up_block(u7, c2, 64)
    u9 = up_block(u8, c1, 32)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(u9)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy', iou_metric])
    return model
