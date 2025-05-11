from PIL import Image, ImageEnhance
import pytesseract
import re
import os

def read_and_format_plate(image_path, output_dir='./formatted_images'):
    image = Image.open(image_path).convert("L")  # Convert to grayscale

    # Enhance the image for better contrast (if necessary)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast

    # Binarize the image to improve OCR performance
    image = image.point(lambda p: p > 128 and 255)  # Simple thresholding

    # Save image if requested (in this case, we're just saving the processed image)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"formatted_{filename}")
    image.save(save_path)
    print(f"Formatted image saved to: {save_path}")

    # OCR configuration
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    raw_text = pytesseract.image_to_string(image, config=config)
    raw_text = raw_text.strip().replace(" ", "").upper()

    # Format using license plate regex
    match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$', raw_text)
    if match:
        formatted = f"{match.group(1)} {match.group(2)} {match.group(3)}".strip()
        print("Formatted Plate:", formatted)
    else:
        print("Unrecognized Plate Format. Raw OCR:", raw_text)

# Example usage
read_and_format_plate('./inverts/inverted_prediction_0_v1_v1.png')
