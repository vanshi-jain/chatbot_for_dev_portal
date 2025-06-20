import sys
from paddleocr import PaddleOCR
from pathlib import Path

def extract_text_from_image(image_path: str, ocr):
    """Extract ordered text blocks from swimlane diagram image using PaddleOCR."""
    result = ocr.predict(image_path)

    for idx, res in enumerate(result):
        res.save_to_json(f"./res/{idx}.json")
        res.save_to_img(f"./res/{idx}.png")    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to swimlane image (PNG/JPG)")
    parser.add_argument("--out", default="output.json", help="Output JSON file")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"[!] File not found: {args.image}")
        sys.exit(1)

    ocr = PaddleOCR(use_textline_orientation=True, lang='en')  # Latest model as of June 2025
    print(f"[→] Extracting from image: {args.image}")
    extract_text_from_image(args.image, ocr)

if __name__ == "__main__":
    main()
