import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_distillation_loss(student_logits, teacher_logits, labels, alpha, temperature):
    """
    Calculate knowledge distillation loss combining:
    1. Cross-entropy loss with true labels
    2. KL divergence loss with teacher soft targets
    
    Args:
        student_logits: Raw logits from student model
        teacher_logits: Raw logits from teacher model
        labels: True labels (ground truth)
        alpha: Weight for distillation loss (0-1)
        temperature: Temperature for softmax (higher = softer distributions)
    
    Returns:
        total_loss, ce_loss, kd_loss
    """
    # Cross-entropy loss with true labels
    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
    
    # Knowledge distillation loss (KL divergence)
    # Apply temperature to soften the probability distributions
    student_logits_temp = student_logits / temperature
    teacher_logits_temp = teacher_logits / temperature
    
    # Calculate KL divergence between student and teacher distributions
    # KL(P||Q) where P is teacher (target) and Q is student (prediction)
    kd_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits_temp, dim=1),
        F.softmax(teacher_logits_temp, dim=1)
    ) * (temperature ** 2)  # Scale by temperature squared
    
    # Combine losses: weighted sum of distillation and supervised losses
    total_loss = alpha * kd_loss + (1 - alpha) * ce_loss
    
    return total_loss, ce_loss, kd_loss


def calculate_accuracy(logits, labels):
    """
    Calculate classification accuracy
    """
    with torch.no_grad():
        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
    return accuracy, correct, total


def print_loss_info(total_loss, ce_loss, kd_loss, accuracy, step_type="Train"):
    """
    Print formatted loss information
    """
    print(f"{step_type} - Total: {total_loss:.4f}, CE: {ce_loss:.4f}, "
          f"KD: {kd_loss:.4f}, Acc: {accuracy:.4f}")


class DistillationLossTracker:
    """
    Track and compute running averages of losses during training
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters"""
        self.total_loss = 0.0
        self.total_ce_loss = 0.0
        self.total_kd_loss = 0.0
        self.total_correct = 0
        self.total_samples = 0
        
    def update(self, loss, ce_loss, kd_loss, correct, batch_size):
        """Update running totals"""
        self.total_loss += loss * batch_size
        self.total_ce_loss += ce_loss * batch_size
        self.total_kd_loss += kd_loss * batch_size
        self.total_correct += correct
        self.total_samples += batch_size
    
    def get_averages(self):
        """Get average losses and accuracy"""
        if self.total_samples == 0:
            return 0.0, 0.0, 0.0, 0.0
            
        avg_loss = self.total_loss / self.total_samples
        avg_ce_loss = self.total_ce_loss / self.total_samples
        avg_kd_loss = self.total_kd_loss / self.total_samples
        accuracy = self.total_correct / self.total_samples
        
        return avg_loss, avg_ce_loss, avg_kd_loss, accuracy


# Additional loss functions for experimentation

def focal_loss(logits, labels, alpha=1.0, gamma=2.0):
    """
    Focal loss for handling class imbalance
    """
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def label_smoothing_loss(logits, labels, smoothing=0.1):
    """
    Label smoothing cross-entropy loss
    """
    num_classes = logits.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    
    return F.kl_div(F.log_softmax(logits, dim=1), true_dist, reduction='batchmean')


def cosine_similarity_loss(student_features, teacher_features):
    """
    Cosine similarity loss for feature matching
    """
    student_norm = F.normalize(student_features, p=2, dim=1)
    teacher_norm = F.normalize(teacher_features, p=2, dim=1)
    
    cosine_sim = F.cosine_similarity(student_norm, teacher_norm, dim=1)
    loss = 1 - cosine_sim.mean()  # Convert similarity to loss
    
    return loss


def calculate_advanced_distillation_loss(student_logits, teacher_logits, labels, alpha, temperature, 
                                       label_smoothing=0.0, use_focal=False, focal_gamma=2.0):
    """
    Advanced knowledge distillation loss with additional regularization techniques:
    1. Label smoothing for better generalization
    2. Optional focal loss for hard examples
    3. Enhanced KL divergence with temperature scaling
    
    Args:
        student_logits: Raw logits from student model
        teacher_logits: Raw logits from teacher model
        labels: True labels (ground truth)
        alpha: Weight for distillation loss (0-1)
        temperature: Temperature for softmax (higher = softer distributions)
        label_smoothing: Label smoothing factor (0-1)
        use_focal: Whether to use focal loss instead of CE
        focal_gamma: Focal loss gamma parameter
    
    Returns:
        total_loss, ce_loss, kd_loss
    """
    # Choose between standard CE, label smoothing, or focal loss
    if use_focal:
        ce_loss = focal_loss(student_logits, labels, gamma=focal_gamma)
    elif label_smoothing > 0:
        ce_loss = label_smoothing_loss(student_logits, labels, smoothing=label_smoothing)
    else:
        ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
    
    # Enhanced knowledge distillation loss
    student_logits_temp = student_logits / temperature
    teacher_logits_temp = teacher_logits / temperature
    
    # Soft targets from teacher
    soft_teacher = F.softmax(teacher_logits_temp, dim=1)
    log_soft_student = F.log_softmax(student_logits_temp, dim=1)
    
    # KL divergence with temperature scaling
    kd_loss = nn.KLDivLoss(reduction='batchmean')(log_soft_student, soft_teacher) * (temperature ** 2)
    
    # Additional feature matching loss (cosine similarity)
    # Note: This would require feature extraction from intermediate layers
    feature_loss = 0.0  # Placeholder for now
    
    # Combine losses with dynamic weighting
    total_loss = alpha * kd_loss + (1 - alpha) * ce_loss + 0.1 * feature_loss
    
    return total_loss, ce_loss, kd_loss 