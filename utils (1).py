import torch
import random
import numpy as np
import os
from typing import Dict, Any
import json


def set_random_seed(seed: int = 42):
    """
    Set random seeds for reproducibility (alias for set_seed)
    """
    set_seed(seed)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_system_info():
    """
    Log system information including GPU memory
    """
    print("💻 System Information:")
    get_gpu_memory_info()
    
    # Add CPU info
    try:
        import psutil
        print(f"CPU cores: {psutil.cpu_count()}")
        print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    except ImportError:
        print("psutil not available for system info")


def save_training_log(training_summary: Dict[str, Any], filepath: str):
    """
    Save training summary to JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(training_summary, f, indent=2, default=str)
    print(f"Training summary saved to {filepath}")


def get_gpu_memory_info():
    """
    Get GPU memory information
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB")
            print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.1f} GB")
    else:
        print("CUDA not available")


def count_parameters(model, trainable_only=True):
    """
    Count model parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def save_config(config_dict: Dict[str, Any], filepath: str):
    """
    Save configuration to JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def format_time(seconds):
    """
    Format seconds into human readable time
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"


def create_dir_if_not_exists(path: str):
    """
    Create directory if it doesn't exist
    """
    os.makedirs(path, exist_ok=True)


def log_experiment(config, results, log_file="experiments.log"):
    """
    Log experiment configuration and results
    """
    import datetime
    
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "results": results
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_model_summary(model, input_size=None):
    """
    Print model summary
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model, trainable_only=False):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    
    if input_size:
        print(f"Input size: {input_size}")
    
    # Model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_all_mb:.2f} MB")


def check_dataset_features(dataset_sample):
    """
    Check what features are available in a dataset sample
    """
    print("Dataset features:")
    for key, value in dataset_sample.items():
        if isinstance(value, (list, tuple)):
            print(f"  {key}: {type(value)} (length: {len(value)})")
            if len(value) > 0:
                print(f"    First item type: {type(value[0])}")
        else:
            print(f"  {key}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"    Shape: {value.shape}")
            elif hasattr(value, 'size'):
                print(f"    Size: {value.size}")


def verify_model_compatibility(teacher_model, student_model, sample_batch):
    """
    Verify that teacher and student models can process the same data
    """
    print("Verifying model compatibility...")
    
    try:
        # Test teacher
        teacher_inputs = {k: v.cuda() for k, v in sample_batch['teacher_inputs'].items()}
        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs)
        print(f"✓ Teacher forward pass successful: {teacher_outputs.logits.shape}")
        
        # Test student
        student_inputs = {k: v.cuda() for k, v in sample_batch['student_inputs'].items()}
        with torch.no_grad():
            student_outputs = student_model(**student_inputs)
        print(f"✓ Student forward pass successful: {student_outputs.logits.shape}")
        
        # Check output compatibility
        if teacher_outputs.logits.shape[-1] == student_outputs.logits.shape[-1]:
            print("✓ Models have compatible output dimensions")
        else:
            print(f"✗ Output dimension mismatch: Teacher {teacher_outputs.logits.shape[-1]} vs Student {student_outputs.logits.shape[-1]}")
            
        return True
        
    except Exception as e:
        print(f"✗ Model compatibility check failed: {e}")
        return False 