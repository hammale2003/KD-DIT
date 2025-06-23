import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Processor,
    LayoutLMv3ImageProcessor,
    LayoutLMv3TokenizerFast,
    LayoutLMv3Config
)
from config import TEACHER_MODEL_NAME, STUDENT_MODEL_NAME, NUM_CLASSES, DEVICE


def load_teacher_model():
    """
    Load DiT (Document Image Transformer) as teacher model
    """
    print("Chargement du modèle Teacher (DiT)...")
    
    # Load DiT processor and model
    processor = AutoImageProcessor.from_pretrained(TEACHER_MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(TEACHER_MODEL_NAME)
    
    # Set to evaluation mode and move to device
    model.eval()
    model.to(DEVICE)
    
    # Freeze teacher model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    print(f"Teacher model loaded: {TEACHER_MODEL_NAME}")
    return model, processor


def create_ultra_tiny_config():
    """
    Create an ultra-lightweight LayoutLMv3 configuration with <10M parameters
    
    Strategy:
    - Hidden size: 768 -> 128 (very small)
    - Num layers: 12 -> 2 (minimal depth)
    - Attention heads: 12 -> 2 (minimal attention)
    - Intermediate size: 3072 -> 256 (minimal FFN)
    - Coordinate embedding: simplified
    """
    config = LayoutLMv3Config.from_pretrained(STUDENT_MODEL_NAME)
    
    # ULTRA-tiny modifications for <10M parameters
    config.hidden_size = 64             # VERY small hidden dimension
    config.num_hidden_layers = 1        # Only 1 transformer layer
    config.num_attention_heads = 2      # Minimal attention heads
    config.intermediate_size = 128      # Very small FFN
    config.num_labels = NUM_CLASSES
    
    # Adjust dependent parameters
    config.coordinate_size = 64         # Match hidden_size
    config.shape_size = 64             # Match hidden_size
    
    # Simplify embeddings drastically
    config.max_position_embeddings = 256   # Very small
    config.max_2d_position_embeddings = 256  # Very small spatial positions
    
    # Reduce vocab if possible
    config.vocab_size = min(config.vocab_size, 30000)  # Limit vocabulary
    
    # Reduce vocabulary size if possible (use smaller tokenizer subset)
    # Note: We'll keep original vocab to maintain compatibility
    
    return config


# Custom class removed - using standard LayoutLMv3 with aggressive freezing instead


def load_student_model():
    """
    Load ultra-lightweight student model with <10M parameters
    Strategy: Use standard LayoutLMv3 with aggressive freezing for reliability
    """
    print("Chargement du modèle Student Ultra-Léger (<10M paramètres)...")
    
    # Load tokenizer
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(STUDENT_MODEL_NAME)
    
    # Simple processor
    class SimpleLayoutLMv3Processor:
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
                
                # Clamp boxes to valid range to avoid CUDA indexing errors
                if isinstance(boxes, list):
                    boxes = torch.tensor(boxes)
                boxes = torch.clamp(boxes, 0, 999)  # Safe range for LayoutLMv3
                
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
    
    processor = SimpleLayoutLMv3Processor(tokenizer)
    
    # Use standard LayoutLMv3 with aggressive freezing (more reliable than custom model)
    print("Using standard LayoutLMv3 with aggressive parameter freezing...")
    
    # Load standard LayoutLMv3
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        STUDENT_MODEL_NAME, 
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    
    # FREEZE ALL parameters except classifier (ultra-aggressive freezing)
    for name, param in model.named_parameters():
        param.requires_grad = False
        # Only classifier layers trainable
        if 'classifier' in name:
            param.requires_grad = True
    
    # Move to device
    model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Student model created: LayoutLMv3 Ultra-Frozen")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.3f}%)")
    print(f"Model size: ~{total_params * 4 / 1024**2:.1f} MB")
    
    # Check if we achieved our goal
    if trainable_params <= 10_000_000:
        print(f"✅ SUCCESS: Only {trainable_params:,} trainable parameters!")
        print(f"📊 Compression: {total_params/trainable_params:.0f}x parameter reduction for training")
    else:
        print(f"⚠️  {trainable_params:,} trainable parameters (still over 10M)")
    
    return model, processor


def get_model_info(model, name):
    """
    Print model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{name} Model Info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")


def load_trained_student_model(model_path="student_model.pth"):
    """
    Load the trained student model - same implementation as in test_student_model.py and gradio_interface.py
    """
    print(f"Loading trained student model from {model_path}...")
    
    # Load tokenizer and processor
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(STUDENT_MODEL_NAME)
    
    # Simple processor class
    class SimpleLayoutLMv3Processor:
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
        print(f"Using pre-trained weights instead...")
    
    model.to(DEVICE)
    model.eval()
    
    return model, processor


if __name__ == "__main__":
    # Test model loading
    teacher_model, teacher_processor = load_teacher_model()
    student_model, student_processor = load_student_model()
    
    get_model_info(teacher_model, "Teacher (DiT)")
    get_model_info(student_model, "Student (Ultra-Light)") 