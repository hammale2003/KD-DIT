import torch

# Device configuration
def configure_device():
    """Configure and return the best available device"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU(s) disponible(s): {gpu_count}")
        
        available_gpus = [
            i for i in range(gpu_count) 
            if torch.cuda.get_device_properties(i).total_memory > torch.cuda.memory_allocated(i)
        ]
        
        if available_gpus:
            device = torch.device(f"cuda:{available_gpus[0]}")
            print(f"Utilisation du GPU: {device}")
        else:
            print("Aucun GPU disponible avec de la mémoire libre. Utilisation du CPU.")
            device = torch.device("cpu")
    else:
        print("Aucun GPU disponible. Utilisation du CPU.")
        device = torch.device("cpu")
    
    return device

# Training hyperparameters - OPTIMIZED FOR BETTER PERFORMANCE
BATCH_SIZE = 16  # Keep current - good for memory
NUM_EPOCHS = 10  # Increase for better convergence
LEARNING_RATE = 2e-4  # Slightly lower for better stability
ALPHA = 0.8  # Increase distillation weight (was 0.7)
TEMPERATURE = 3.0  # Lower temperature for sharper distributions (was 4.0)
NUM_CLASSES = 16  # RVL-CDIP classes

# Advanced training parameters
WARMUP_EPOCHS = 2  # Learning rate warmup
WEIGHT_DECAY = 1e-4  # L2 regularization 
DROPOUT_RATE = 0.1  # Dropout for regularization
GRADIENT_CLIP_VALUE = 1.0  # Gradient clipping
LABEL_SMOOTHING = 0.1  # Label smoothing for better generalization

# Model paths
TEACHER_MODEL_NAME = "microsoft/dit-large-finetuned-rvlcdip"
STUDENT_MODEL_NAME = "microsoft/layoutlmv3-base"  # We'll modify this to create tiny version

# File paths
CHECKPOINT_PATH = "latest_checkpoint.pth"
BEST_MODEL_PATH = "student_modell.pth"

# Dataset configuration
DATASET_NAME = "HAMMALE/rvl_cdip_OCR"
MAX_LENGTH = 512

# Device
DEVICE = configure_device() 
