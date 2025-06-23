import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import numpy as np
import io
from typing import List, Dict, Any
from config import DATASET_NAME, BATCH_SIZE, MAX_LENGTH, NUM_CLASSES


class RVLCDIPEnhancedDataset(Dataset):
    """
    Enhanced RVL-CDIP Dataset that handles:
    - image: PIL.Image
    - width, height: int
    - category: str
    - ocr_words: List[str]
    - word_boxes: List[List[int]]
    - ocr_paragraphs: List[str]
    - paragraph_boxes: List[List[int]]
    - label: int
    """
    
    def __init__(self, dataset_split, teacher_processor, student_processor):
        self.dataset = dataset_split
        self.teacher_processor = teacher_processor
        self.student_processor = student_processor
        self.label_list = list(range(NUM_CLASSES))
        
        print(f"Taille du dataset: {len(self.dataset)}")
        print(f"Colonnes disponibles: {list(self.dataset[0].keys())}")
        self._debug_dataset_samples()

    def _debug_dataset_samples(self, num_samples=3):
        """Debug first few samples to understand data structure"""
        for i in range(min(num_samples, len(self.dataset))):
            item = self.dataset[i]
            print(f"\nDébogage échantillon {i}:")
            print(f"  Type d'image: {type(item['image'])}")
            print(f"  Dimensions: {item.get('width', 'N/A')} x {item.get('height', 'N/A')}")
            print(f"  Catégorie: {item.get('category', 'N/A')}")
            print(f"  Label: {item['label']}")
            print(f"  Nombre de mots OCR: {len(item.get('ocr_words', []))}")
            print(f"  Nombre de paragraphes OCR: {len(item.get('ocr_paragraphs', []))}")
            
            # Check first few OCR words
            if item.get('ocr_words'):
                print(f"  Premiers mots: {item['ocr_words'][:5]}")
            
            try:
                converted_image = self._convert_to_rgb(item['image'])
                print(f"  Taille image convertie: {converted_image.size}")
            except Exception as e:
                print(f"  Erreur conversion image: {e}")

    def _convert_to_rgb(self, image):
        """Convert various image formats to RGB PIL Image"""
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)
            elif image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            elif image.shape[2] == 4:
                image = image[:, :, :3]  # Handle RGBA
            
            if image.dtype != np.uint8:
                img_min, img_max = image.min(), image.max()
                if img_max > img_min:
                    image = ((image - img_min) / (img_max - img_min) * 255)
                else:
                    image = np.zeros_like(image)
                image = image.astype(np.uint8)
            return Image.fromarray(image)
        
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert('RGB')
        if isinstance(image, str):
            return Image.open(image).convert('RGB')
        
        try:
            return Image.fromarray(np.array(image)).convert('RGB')
        except Exception as e:
            print(f"Impossible de convertir l'image: {e}")
            raise

    def _prepare_layoutlmv3_inputs(self, item):
        """
        Prepare inputs for LayoutLMv3 (student) with OCR data
        """
        # Extract OCR text and boxes
        words = item.get('ocr_words', [])
        boxes = item.get('word_boxes', [])
        
        # Ensure we have matching words and boxes
        if len(words) != len(boxes):
            min_len = min(len(words), len(boxes))
            words = words[:min_len]
            boxes = boxes[:min_len]
        
        # Convert text to single string for tokenization
        text = " ".join(words) if words else "document"
        
        # Normalize boxes to 1000 scale (LayoutLMv3 expects this)
        normalized_boxes = []
        if boxes:
            img_width = item.get('width', 1000)
            img_height = item.get('height', 1000)
            
            for box in boxes:
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    # Normalize to 1000 scale
                    norm_box = [
                        min(max(int(1000 * x1 / img_width), 0), 1000),
                        min(max(int(1000 * y1 / img_height), 0), 1000),
                        min(max(int(1000 * x2 / img_width), 0), 1000),
                        min(max(int(1000 * y2 / img_height), 0), 1000)
                    ]
                    normalized_boxes.append(norm_box)
        
        # Process with simplified LayoutLMv3 (no pytesseract needed)
        try:
            inputs = self.student_processor(
                words=words if words else ["document"],
                boxes=normalized_boxes if normalized_boxes else None,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH
            )
                
        except Exception as e:
            print(f"Erreur dans le traitement LayoutLMv3, fallback simple: {e}")
            # Simple fallback
            inputs = self.student_processor(
                words=["document"],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH
            )
        
        return inputs

    def _prepare_dit_inputs(self, item):
        """
        Prepare inputs for DiT (teacher) - image only
        """
        image = self._convert_to_rgb(item['image'])
        
        # DiT processes image only
        inputs = self.teacher_processor(
            image,
            return_tensors="pt"
        )
        
        return inputs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            
            # Prepare inputs for both models
            teacher_inputs = self._prepare_dit_inputs(item)
            student_inputs = self._prepare_layoutlmv3_inputs(item)
            
            # Process label
            label = item['label']
            label_index = label if label in self.label_list else 0
            
            # Squeeze batch dimension
            teacher_inputs = {k: v.squeeze(0) for k, v in teacher_inputs.items()}
            student_inputs = {k: v.squeeze(0) for k, v in student_inputs.items()}
            
            return {
                'teacher_inputs': teacher_inputs,
                'student_inputs': student_inputs,
                'label': torch.tensor(label_index, dtype=torch.long)
            }
            
        except Exception as e:
            print(f"!!!!!! ERREUR DANS __getitem__ (index {idx}) !!!!!!!!")
            print(f"Exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise e


def load_data(teacher_processor, student_processor):
    """
    Load and prepare RVL-CDIP dataset with enhanced features
    """
    print(f"Chargement du dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME)
    
    # Create enhanced datasets
    train_dataset = RVLCDIPEnhancedDataset(
        dataset['train'],
        teacher_processor,
        student_processor
    )
    
    val_dataset = RVLCDIPEnhancedDataset(
        dataset['validation'],
        teacher_processor,
        student_processor
    )
    
    print("Création des DataLoaders...")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print("DataLoaders créés.")
    return train_loader, val_loader


def get_dataset_info(dataset_name=DATASET_NAME):
    """
    Get information about the dataset
    """
    print(f"Informations sur le dataset {dataset_name}:")
    dataset = load_dataset(dataset_name)
    
    for split_name, split_data in dataset.items():
        print(f"\n{split_name}:")
        print(f"  Taille: {len(split_data)}")
        
        if len(split_data) > 0:
            sample = split_data[0]
            print(f"  Colonnes: {list(sample.keys())}")
            
            # Check OCR data availability
            if 'ocr_words' in sample:
                print(f"  Mots OCR (échantillon): {len(sample['ocr_words'])}")
            if 'word_boxes' in sample:
                print(f"  Boxes mots (échantillon): {len(sample['word_boxes'])}")


# Create a compatible wrapper class for continual_learning.py
class RVLCDIPDataset:
    """
    Wrapper class for backward compatibility with continual_learning.py
    Adapts RVLCDIPEnhancedDataset to the expected interface
    """
    
    def __init__(self, root_dir: str, split: str = 'train', max_samples_per_class: int = None):
        from datasets import load_dataset
        from models import load_teacher_model, load_trained_student_model
        
        # Load dataset from Hugging Face (ignore root_dir since we use HF dataset)
        self.dataset_name = root_dir if root_dir.startswith("HAMMALE/") else "HAMMALE/rvl_cdip_OCR"
        self.split = split
        self.max_samples_per_class = max_samples_per_class
        
        # Load the HF dataset
        dataset = load_dataset(self.dataset_name)
        
        # Get the appropriate split
        if split in dataset:
            self.dataset_split = dataset[split]
        else:
            # Fallback to train if split not found
            self.dataset_split = dataset['train']
            print(f"Warning: Split '{split}' not found, using 'train' split")
        
        # Load processors (needed for RVLCDIPEnhancedDataset)
        try:
            # Lazy loading to avoid circular imports during gradio startup
            teacher_model, teacher_processor = load_teacher_model()
            student_model, student_processor = load_trained_student_model()
            
            # Create the underlying enhanced dataset
            self.enhanced_dataset = RVLCDIPEnhancedDataset(
                self.dataset_split, teacher_processor, student_processor
            )
            
            # Extract labels for compatibility
            self.labels = [item['label'] for item in self.dataset_split]
            print(f"✅ Successfully initialized RVLCDIPDataset with {len(self.labels)} samples")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load models for dataset initialization: {e}")
            print("   Using minimal fallback mode...")
            # Create a minimal fallback
            self.enhanced_dataset = None
            self.labels = [item['label'] for item in self.dataset_split]
    
    def __len__(self):
        return len(self.dataset_split)
    
    def __getitem__(self, idx):
        if self.enhanced_dataset:
            return self.enhanced_dataset[idx]
        else:
            # Fallback implementation
            import torch
            item = self.dataset_split[idx]
            return {
                'teacher_inputs': {'pixel_values': torch.zeros(3, 224, 224)},  # Dummy
                'student_inputs': {'input_ids': torch.zeros(512, dtype=torch.long)},  # Dummy  
                'label': torch.tensor(item['label'], dtype=torch.long)
            }

if __name__ == "__main__":
    # Test dataset loading
    get_dataset_info() 