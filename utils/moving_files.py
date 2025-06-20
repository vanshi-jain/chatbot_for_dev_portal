import os
import shutil
import re

# Source directory containing all files
source_dir = "./processed"
destination_root = "./data"

# List all image files (e.g., image1.png, image2.jpg)
image_files = [f for f in os.listdir(source_dir) if re.match(r'image\d+\.(png|jpg|jpeg)', f, re.IGNORECASE)]

for img_file in image_files:
    # Extract number from filename like 'image5.png' -> 5
    match = re.search(r'image(\d+)', img_file)
    if not match:
        continue
    x = match.group(1)
    input_dir = os.path.join(destination_root, f"input{x}")

    # Create subfolders
    raw_input = os.path.join(input_dir, "raw_input")
    ocr_output = os.path.join(input_dir, "OCR_output")
    ocr_processed = os.path.join(input_dir, "OCR_processed")
    os.makedirs(raw_input, exist_ok=True)
    os.makedirs(ocr_output, exist_ok=True)
    os.makedirs(ocr_processed, exist_ok=True)

    # Move/copy files
    img_path = os.path.join(source_dir, img_file)
    shutil.move(img_path, raw_input)

    json_file = f"image{x}.json"
    txt_file = f"image{x}.png"
    processed_file = f"image{x}.txt"

    for file, target_dir in [(json_file, ocr_output), (txt_file, ocr_output), (processed_file, ocr_processed)]:
        file_path = os.path.join(source_dir, file)
        if os.path.exists(file_path):
            shutil.move(file_path, target_dir)
