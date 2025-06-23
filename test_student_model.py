import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3TokenizerFast
from config import DEVICE, NUM_CLASSES, STUDENT_MODEL_NAME
import argparse
import os


# RVL-CDIP class names (16 classes)
CLASS_NAMES = [
    "letter", "form", "email", "handwritten", "advertisement", 
    "scientific report", "scientific publication", "specification", 
    "file folder", "news article", "budget", "invoice", 
    "presentation", "questionnaire", "resume", "memo"
]


class SimpleLayoutLMv3Processor:
    """Simple processor for LayoutLMv3 (same as in models.py)"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, text=None, words=None, boxes=None, return_tensors="pt", **kwargs):
        if words is not None:
            word_list = words
        elif text is not None:
            word_list = text.split() if text else ["document"]
        else:
            word_list = ["document"]
        
        # Extract common parameters to avoid conflicts
        padding = kwargs.pop('padding', True)
        truncation = kwargs.pop('truncation', True)
        max_length = kwargs.pop('max_length', 512)
        
        if boxes is not None and len(boxes) > 0:
            if len(boxes) != len(word_list):
                if len(boxes) > len(word_list):
                    boxes = boxes[:len(word_list)]
                else:
                    while len(boxes) < len(word_list):
                        boxes.append([0, 0, 0, 0])
            
            # Clamp boxes to valid range
            if isinstance(boxes, list):
                boxes = torch.tensor(boxes)
            boxes = torch.clamp(boxes, 0, 999)
            
            encoding = self.tokenizer(
                word_list, boxes=boxes.tolist(), return_tensors=return_tensors,
                padding=padding, truncation=truncation, max_length=max_length, **kwargs
            )
        else:
            encoding = self.tokenizer(
                word_list, return_tensors=return_tensors,
                padding=padding, truncation=truncation, max_length=max_length, **kwargs
            )
        
        return encoding


def load_trained_student_model(model_path="student_model.pth"):
    """Load the trained student model"""
    print(f"Loading trained student model from {model_path}...")
    
    # Load tokenizer and processor
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(STUDENT_MODEL_NAME)
    processor = SimpleLayoutLMv3Processor(tokenizer)
    
    # Load the base model architecture
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        STUDENT_MODEL_NAME, 
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    
    # Load trained weights
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Trained weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading trained weights: {e}")
        print("Using pre-trained weights instead...")
    
    model.to(DEVICE)
    model.eval()
    
    return model, processor


def extract_ocr_with_easyocr(image_path):
    """Extract OCR data using easyOCR"""
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        
        # Read image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Convert to numpy for easyOCR
        image_np = np.array(image)
        
        # Extract text and boxes
        results = reader.readtext(image_np)
        
        words = []
        boxes = []
        
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Filter low confidence detections
                words.append(text)
                
                # Convert bbox to [x1, y1, x2, y2] format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                
                # Normalize to 1000 scale (LayoutLMv3 expects this)
                img_width, img_height = image.size
                norm_box = [
                    min(max(int(1000 * x1 / img_width), 0), 1000),
                    min(max(int(1000 * y1 / img_height), 0), 1000),
                    min(max(int(1000 * x2 / img_width), 0), 1000),
                    min(max(int(1000 * y2 / img_height), 0), 1000)
                ]
                boxes.append(norm_box)
        
        return words, boxes
        
    except ImportError:
        print("⚠️ easyOCR not installed. Install with: pip install easyocr")
        return None, None
    except Exception as e:
        print(f"❌ Error during OCR extraction: {e}")
        return None, None


def test_image_with_ocr_data(model, processor, words, boxes):
    """Test model with pre-existing OCR data"""
    print(f"Testing with {len(words)} words and {len(boxes) if boxes else 0} bounding boxes...")
    
    # Prepare inputs
    inputs = processor(
        words=words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    
    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_class_id].item()
    
    return predicted_class_id, confidence, probabilities[0].cpu().numpy()


def test_image_with_auto_ocr(model, processor, image_path):
    """Test model with automatic OCR extraction"""
    print(f"Testing image: {image_path}")
    print("Extracting OCR data automatically...")
    
    words, boxes = extract_ocr_with_easyocr(image_path)
    
    if words is None or len(words) == 0:
        print("❌ No OCR data extracted. Using fallback...")
        words = ["document"]
        boxes = None
    else:
        print(f"✅ Extracted {len(words)} words from image")
    
    return test_image_with_ocr_data(model, processor, words, boxes)


def print_results(predicted_class_id, confidence, probabilities):
    """Print prediction results"""
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Class: {CLASS_NAMES[predicted_class_id]} (ID: {predicted_class_id})")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    print("\nTop 5 Predictions:")
    print("-" * 40)
    top_indices = np.argsort(probabilities)[-5:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {CLASS_NAMES[idx]}: {probabilities[idx]:.4f} ({probabilities[idx]*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Test trained student model")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--model", type=str, default="student_model.pth", 
                       help="Path to trained model file")
    parser.add_argument("--words", type=str, nargs="+", 
                       help="OCR words (if available)")
    parser.add_argument("--boxes", type=str, 
                       help="OCR bounding boxes as comma-separated values (x1,y1,x2,y2)")
    
    args = parser.parse_args()
    
    # Load model
    model, processor = load_trained_student_model(args.model)
    
    if args.image:
        if args.words and args.boxes:
            # Use provided OCR data
            print("Using provided OCR data...")
            words = args.words
            
            # Parse boxes
            box_values = [int(x) for x in args.boxes.split(',')]
            boxes = [box_values[i:i+4] for i in range(0, len(box_values), 4)]
            
            predicted_class_id, confidence, probabilities = test_image_with_ocr_data(
                model, processor, words, boxes
            )
        else:
            # Use automatic OCR
            predicted_class_id, confidence, probabilities = test_image_with_auto_ocr(
                model, processor, args.image
            )
        
        print_results(predicted_class_id, confidence, probabilities)
    
    else:
        # Interactive mode
        print("\n🔍 INTERACTIVE TESTING MODE")
        print("="*50)
        
        while True:
            print("\nOptions:")
            print("1. Test image with automatic OCR")
            print("2. Test with manual OCR data")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                image_path = input("Enter image path: ").strip()
                if os.path.exists(image_path):
                    try:
                        predicted_class_id, confidence, probabilities = test_image_with_auto_ocr(
                            model, processor, image_path
                        )
                        print_results(predicted_class_id, confidence, probabilities)
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Image file not found!")
            
            elif choice == "2":
                words_input = input("Enter words (space-separated): ").strip()
                words = words_input.split() if words_input else ["document"]
                
                boxes_input = input("Enter boxes (format: x1,y1,x2,y2 x1,y1,x2,y2 ...): ").strip()
                if boxes_input:
                    try:
                        box_groups = boxes_input.split()
                        boxes = []
                        for box_group in box_groups:
                            box_values = [int(x) for x in box_group.split(',')]
                            if len(box_values) == 4:
                                boxes.append(box_values)
                    except:
                        print("❌ Invalid box format. Using no boxes.")
                        boxes = None
                else:
                    boxes = None
                
                try:
                    predicted_class_id, confidence, probabilities = test_image_with_ocr_data(
                        model, processor, words, boxes
                    )
                    print_results(predicted_class_id, confidence, probabilities)
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            elif choice == "3":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice!")


if __name__ == "__main__":
    main() 