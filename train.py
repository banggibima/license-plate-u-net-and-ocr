import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models, callbacks, saving

# CONFIG
DATASET_PATH = './dataset'
MODEL_DIR = './models'
PLOT_DIR = './plots'
PREDICT_DIR = './predicts'
RESULT_DIR = './results'

# VERSIONING UTILS
def get_versioned_path(base_dir, prefix, ext):
    os.makedirs(base_dir, exist_ok=True)
    version = 1
    while True:
        path = os.path.join(base_dir, f"{prefix}_v{version}.{ext}")
        if not os.path.exists(path):
            return path
        version += 1

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
        msk = cv2.resize(load_mask(msk_path), size, interpolation=cv2.INTER_NEAREST)
        msk = np.expand_dims((msk > 127).astype(np.float32), axis=-1)
        images.append(img)
        masks.append(msk)

    return np.array(images), np.array(masks)

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

# TRAINING
print("Loading dataset...")
X_train, y_train = load_dataset(os.path.join(DATASET_PATH, 'images/train'), os.path.join(DATASET_PATH, 'masks/train'))
X_valid, y_valid = load_dataset(os.path.join(DATASET_PATH, 'images/valid'), os.path.join(DATASET_PATH, 'masks/valid'))
X_test, y_test   = load_dataset(os.path.join(DATASET_PATH, 'images/test'),  os.path.join(DATASET_PATH, 'masks/test'))

print("Building model...")
model = build_unet()
model.summary()

model_path = get_versioned_path(MODEL_DIR, 'unet_license_plate', 'keras')
checkpoint_path = model_path.replace('.keras', '_best.keras')

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=100,
    batch_size=8,
    callbacks=[early_stop, model_checkpoint]
)

print(f"Saving model to {model_path}")
model.save(model_path)

# PLOTTING
def plot_history(hist, path):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='Train Acc')
    plt.plot(hist.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.tight_layout()
    plt.savefig(path)
    print(f"Training plot saved to {path}")

plot_path = get_versioned_path(PLOT_DIR, 'training_history', 'png')
plot_history(history, plot_path)

# EVALUATION
print("Evaluating best model...")

best_model = models.load_model(
    checkpoint_path, 
    custom_objects={'iou_metric': iou_metric}
)

loss, acc, iou = best_model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f} | Accuracy: {acc:.4f} | IOU: {iou:.4f}")

# PREDICT
def save_prediction(pred_mask, index):
    base_name = f'prediction_{index}'
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    prediction_path = get_versioned_path(PREDICT_DIR, base_name, 'png')
    
    plt.imsave(prediction_path, pred_mask.squeeze(), cmap='gray')

    print(f"Prediction {index} saved to {prediction_path}")

# Generating and saving the predictions
print("Saving predictions...")

for i in range(5):
    pred_mask = best_model.predict(np.expand_dims(X_test[i], axis=0))[0]
    save_prediction(pred_mask, i)

# RESULT
def save_result(img, true_mask, pred_mask, path=None):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1); plt.imshow(img); plt.title("Image")
    plt.subplot(1, 3, 2); plt.imshow(true_mask.squeeze(), cmap='gray'); plt.title("Ground Truth")
    plt.subplot(1, 3, 3); plt.imshow(pred_mask.squeeze(), cmap='gray'); plt.title("Prediction")

    if path:
        plt.savefig(path, bbox_inches='tight', pad_inches=0.25)
        print(f"Result visualization saved to {path}")
    else:
        plt.show()

print("Generating result visualizations...")
for i in range(10):
    pred_mask = best_model.predict(np.expand_dims(X_test[i], axis=0))[0]
    result_path = get_versioned_path(RESULT_DIR, f'result_{i}', 'png')
    save_result(X_test[i], y_test[i], pred_mask, result_path)
