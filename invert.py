import os
import glob
from PIL import Image, ImageOps

# Versioning utility
def get_versioned_path(base_dir, prefix, ext):
    os.makedirs(base_dir, exist_ok=True)
    version = 1
    while True:
        path = os.path.join(base_dir, f"{prefix}_v{version}.{ext}")
        if not os.path.exists(path):
            return path
        version += 1

def invert_image(image_path, output_dir='./inverts', save_inverted=True):
    """
    Inverts a grayscale image and saves it with versioned naming.

    Parameters:
    - image_path: Path to the image to be inverted.
    - output_dir: Directory to save inverted image.
    - save_inverted: If True, saves the result.
    """
    image = Image.open(image_path).convert("L")
    inverted_image = ImageOps.invert(image)

    if save_inverted:
        base_filename = os.path.splitext(os.path.basename(image_path))[0]  # e.g., prediction_0_v1
        versioned_path = get_versioned_path(output_dir, f"inverted_{base_filename}", "png")
        inverted_image.save(versioned_path)
        print(f"Inverted image saved to: {versioned_path}")

def process_predictions(predicts_dir='./predicts', output_dir='./inverts'):
    pred_files = sorted(glob.glob(os.path.join(predicts_dir, 'prediction_*.png')))
    
    if not pred_files:
        print("No prediction files found!")
        return

    for pred_file in pred_files:
        print(f"Inverting {pred_file}...")
        invert_image(pred_file, output_dir)

process_predictions()
