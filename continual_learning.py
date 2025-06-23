#!/usr/bin/env python3
"""
Continual Learning System for Knowledge Distillation
Implements sequential task learning with catastrophic forgetting evaluation
and mitigation techniques (Rehearsal, LwF)
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import json
import copy
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from collections import defaultdict
import os

from models import load_teacher_model
from dataset import RVLCDIPDataset
from config import DEVICE, NUM_EPOCHS, STUDENT_MODEL_NAME
from losses import calculate_distillation_loss, calculate_accuracy
from models import load_trained_student_model


class ContinualLearningConfig:
    """Configuration for continual learning experiments"""
    
    # Task division strategies
    TASK_DIVISION_STRATEGIES = {
        'class_split': 'Split by document classes',
        'temporal_split': 'Split by temporal distribution',
        'domain_split': 'Split by document domains'
    }
    
    # Mitigation techniques
    MITIGATION_TECHNIQUES = {
        'naive': 'No mitigation (baseline)',
        'rehearsal': 'Experience Replay/Rehearsal',
        'lwf': 'Learning without Forgetting',
        'ewc': 'Elastic Weight Consolidation',
        'combined': 'Rehearsal + LwF'
    }
    
    def __init__(self):
        # Task configuration
        self.num_tasks = 4  # Divide 16 classes into 4 tasks
        self.classes_per_task = 4
        self.task_strategy = 'class_split'
        
        # Training configuration
        self.epochs_per_task = 3
        self.learning_rate = 1e-4
        self.batch_size = 16
        
        # Rehearsal configuration
        self.rehearsal_memory_size = 100  # samples per previous task
        self.rehearsal_ratio = 0.3  # 30% rehearsal data in each batch
        
        # LwF configuration
        self.lwf_temperature = 3.0
        self.lwf_alpha = 0.5  # Balance between new and old knowledge
        
        # Evaluation configuration
        self.eval_on_all_tasks_every_epoch = True


class TaskManager:
    """Manages task creation and data splitting for continual learning"""
    
    def __init__(self, config: ContinualLearningConfig, dataset_path: str):
        self.config = config
        self.dataset_path = dataset_path
        self.class_names = [
            "letter", "form", "email", "handwritten", "advertisement", 
            "scientific report", "scientific publication", "specification", 
            "file folder", "news article", "budget", "invoice", 
            "presentation", "questionnaire", "resume", "memo"
        ]
        
        # Create tasks
        self.tasks = self._create_tasks()
        self.task_datasets = {}
        
    def _create_tasks(self) -> List[Dict]:
        """Create task definitions based on strategy"""
        tasks = []
        
        if self.config.task_strategy == 'class_split':
            # Split classes sequentially
            for task_id in range(self.config.num_tasks):
                start_class = task_id * self.config.classes_per_task
                end_class = min((task_id + 1) * self.config.classes_per_task, 16)
                
                task_classes = list(range(start_class, end_class))
                task_class_names = [self.class_names[i] for i in task_classes]
                
                tasks.append({
                    'task_id': task_id,
                    'name': f'Task_{task_id}_{task_class_names[0]}_to_{task_class_names[-1]}',
                    'classes': task_classes,
                    'class_names': task_class_names,
                    'description': f'Classes {start_class}-{end_class-1}: {", ".join(task_class_names)}'
                })
        
        return tasks
    
    def get_task_dataset(self, task_id: int, split: str = 'train') -> DataLoader:
        """Get dataset for specific task"""
        if (task_id, split) not in self.task_datasets:
            # Load full dataset
            full_dataset = RVLCDIPDataset(
                root_dir=self.dataset_path,
                split=split,
                max_samples_per_class=500 if split == 'train' else 100
            )
            
            # Filter for task classes
            task_classes = self.tasks[task_id]['classes']
            
            # Find indices for task classes
            task_indices = []
            for idx in range(len(full_dataset)):
                if full_dataset.labels[idx] in task_classes:
                    task_indices.append(idx)
            
            # Create subset
            task_subset = Subset(full_dataset, task_indices)
            
            # Create dataloader
            dataloader = DataLoader(
                task_subset,
                batch_size=self.config.batch_size,
                shuffle=(split == 'train'),
                num_workers=2
            )
            
            self.task_datasets[(task_id, split)] = dataloader
        
        return self.task_datasets[(task_id, split)]
    
    def get_rehearsal_dataset(self, completed_tasks: List[int], 
                            samples_per_task: int) -> Optional[DataLoader]:
        """Create rehearsal dataset from completed tasks"""
        if not completed_tasks:
            return None
        
        rehearsal_datasets = []
        
        for task_id in completed_tasks:
            task_dataset = self.get_task_dataset(task_id, 'train')
            
            # Sample random subset
            all_indices = list(range(len(task_dataset.dataset)))
            selected_indices = random.sample(
                all_indices, 
                min(samples_per_task, len(all_indices))
            )
            
            rehearsal_subset = Subset(task_dataset.dataset, selected_indices)
            rehearsal_datasets.append(rehearsal_subset)
        
        # Combine all rehearsal data
        combined_rehearsal = ConcatDataset(rehearsal_datasets)
        
        return DataLoader(
            combined_rehearsal,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )


class ContinualLearner:
    """Main continual learning system"""
    
    def __init__(self, config: ContinualLearningConfig, dataset_path: str):
        self.config = config
        self.task_manager = TaskManager(config, dataset_path)
        
        # Load models
        self.teacher_model, self.teacher_processor = load_teacher_model()
        self.student_model, self.student_processor = load_trained_student_model()
        
        # Store original model for reference
        self.original_student_state = copy.deepcopy(self.student_model.state_dict())
        
        # Tracking
        self.results = {
            'task_accuracies': defaultdict(list),  # accuracy[task_id][epoch]
            'forgetting_metrics': [],
            'learning_curve': [],
            'final_accuracies': {}
        }
        
        self.completed_tasks = []
        self.old_model_outputs = {}  # For LwF
        
    def train_task(self, task_id: int, mitigation: str = 'naive') -> Dict:
        """Train on a specific task with chosen mitigation technique"""
        print(f"\n=== Training Task {task_id}: {self.task_manager.tasks[task_id]['name']} ===")
        print(f"Mitigation: {self.config.MITIGATION_TECHNIQUES[mitigation]}")
        
        # Get task data
        task_dataloader = self.task_manager.get_task_dataset(task_id, 'train')
        
        # Setup optimizer
        optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Store model state before training (for LwF)
        if mitigation in ['lwf', 'combined'] and self.completed_tasks:
            self._store_old_model_outputs(task_dataloader)
        
        # Training loop
        for epoch in range(self.config.epochs_per_task):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs_per_task}")
            
            # Create training data for this epoch
            if mitigation in ['rehearsal', 'combined'] and self.completed_tasks:
                train_dataloader = self._create_rehearsal_dataloader(task_dataloader)
            else:
                train_dataloader = task_dataloader
            
            # Train epoch
            epoch_loss = self._train_epoch(
                train_dataloader, optimizer, task_id, mitigation
            )
            
            # Evaluate on all tasks
            if self.config.eval_on_all_tasks_every_epoch:
                self._evaluate_all_tasks(epoch, task_id)
            
            print(f"Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}")
        
        # Mark task as completed
        self.completed_tasks.append(task_id)
        
        # Final evaluation
        final_results = self._evaluate_all_tasks(self.config.epochs_per_task - 1, task_id)
        
        return final_results
    
    def _train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer,
                    current_task_id: int, mitigation: str) -> float:
        """Train for one epoch"""
        self.student_model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Task {current_task_id}")
        
        for batch in progress_bar:
            try:
                # Move to device
                teacher_inputs = {k: v.to(DEVICE) for k, v in batch['teacher_inputs'].items()}
                student_inputs = {k: v.to(DEVICE) for k, v in batch['student_inputs'].items()}
                labels = batch['label'].to(DEVICE)
                
                # Forward pass - Teacher
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**teacher_inputs)
                    teacher_logits = teacher_outputs.logits
                
                # Forward pass - Student
                student_outputs = self.student_model(**student_inputs)
                student_logits = student_outputs.logits
                
                # Calculate loss based on mitigation technique
                if mitigation == 'naive':
                    loss = self._calculate_naive_loss(student_logits, teacher_logits, labels)
                elif mitigation == 'rehearsal':
                    loss = self._calculate_rehearsal_loss(student_logits, teacher_logits, labels)
                elif mitigation == 'lwf':
                    loss = self._calculate_lwf_loss(student_logits, teacher_logits, labels, student_inputs)
                elif mitigation == 'combined':
                    loss = self._calculate_combined_loss(student_logits, teacher_logits, labels, student_inputs)
                else:
                    raise ValueError(f"Unknown mitigation technique: {mitigation}")
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _calculate_naive_loss(self, student_logits: torch.Tensor, 
                            teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard distillation loss without mitigation"""
        loss, _, _ = calculate_distillation_loss(
            student_logits, teacher_logits, labels, 
            alpha=0.7, temperature=3.0
        )
        return loss
    
    def _calculate_rehearsal_loss(self, student_logits: torch.Tensor,
                                teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Loss for rehearsal - same as naive since rehearsal is in data preparation"""
        return self._calculate_naive_loss(student_logits, teacher_logits, labels)
    
    def _calculate_lwf_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                           labels: torch.Tensor, student_inputs: Dict) -> torch.Tensor:
        """Learning without Forgetting loss"""
        # Standard distillation loss
        current_loss, _, _ = calculate_distillation_loss(
            student_logits, teacher_logits, labels,
            alpha=0.7, temperature=3.0
        )
        
        # LwF regularization loss
        if hasattr(self, 'old_model') and self.old_model is not None:
            with torch.no_grad():
                old_outputs = self.old_model(**student_inputs)
                old_logits = old_outputs.logits
            
            # KL divergence between current and old predictions
            old_probs = F.softmax(old_logits / self.config.lwf_temperature, dim=-1)
            current_log_probs = F.log_softmax(student_logits / self.config.lwf_temperature, dim=-1)
            
            lwf_loss = F.kl_div(current_log_probs, old_probs, reduction='batchmean')
            lwf_loss *= (self.config.lwf_temperature ** 2)
            
            # Combine losses
            total_loss = (1 - self.config.lwf_alpha) * current_loss + self.config.lwf_alpha * lwf_loss
        else:
            total_loss = current_loss
        
        return total_loss
    
    def _calculate_combined_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                               labels: torch.Tensor, student_inputs: Dict) -> torch.Tensor:
        """Combined rehearsal + LwF loss"""
        return self._calculate_lwf_loss(student_logits, teacher_logits, labels, student_inputs)
    
    def _create_rehearsal_dataloader(self, current_task_dataloader: DataLoader) -> DataLoader:
        """Create combined dataloader with rehearsal data"""
        # Get rehearsal data
        rehearsal_dataloader = self.task_manager.get_rehearsal_dataset(
            self.completed_tasks, self.config.rehearsal_memory_size
        )
        
        if rehearsal_dataloader is None:
            return current_task_dataloader
        
        # Combine current task and rehearsal data
        # This is a simplified version - in practice, you might want more sophisticated mixing
        combined_dataset = ConcatDataset([
            current_task_dataloader.dataset,
            rehearsal_dataloader.dataset
        ])
        
        return DataLoader(
            combined_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
    
    def _store_old_model_outputs(self, dataloader: DataLoader):
        """Store old model state for LwF"""
        self.old_model = copy.deepcopy(self.student_model)
        self.old_model.eval()
    
    def _evaluate_all_tasks(self, epoch: int, current_task: int) -> Dict:
        """Evaluate on all seen tasks"""
        results = {}
        
        for task_id in range(current_task + 1):
            task_acc = self._evaluate_task(task_id)
            results[f'task_{task_id}'] = task_acc
            
            # Store for tracking
            self.results['task_accuracies'][task_id].append(task_acc)
        
        # Calculate average accuracy
        avg_accuracy = np.mean(list(results.values()))
        results['average'] = avg_accuracy
        
        print(f"Task accuracies: {results}")
        
        return results
    
    def _evaluate_task(self, task_id: int) -> float:
        """Evaluate on specific task"""
        self.student_model.eval()
        
        test_dataloader = self.task_manager.get_task_dataset(task_id, 'test')
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                try:
                    student_inputs = {k: v.to(DEVICE) for k, v in batch['student_inputs'].items()}
                    labels = batch['label'].to(DEVICE)
                    
                    outputs = self.student_model(**student_inputs)
                    logits = outputs.logits
                    
                    _, correct, batch_size = calculate_accuracy(logits, labels)
                    total_correct += correct
                    total_samples += batch_size
                    
                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    continue
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return accuracy
    
    def calculate_forgetting_metrics(self) -> Dict:
        """Calculate catastrophic forgetting metrics"""
        forgetting_metrics = {}
        
        if len(self.completed_tasks) <= 1:
            return forgetting_metrics
        
        # For each completed task except the last one
        for task_id in self.completed_tasks[:-1]:
            task_accuracies = self.results['task_accuracies'][task_id]
            
            if len(task_accuracies) >= 2:
                # Maximum accuracy achieved on this task
                max_accuracy = max(task_accuracies)
                # Current accuracy on this task
                current_accuracy = task_accuracies[-1]
                
                # Forgetting = max_acc - current_acc
                forgetting = max_accuracy - current_accuracy
                forgetting_metrics[f'task_{task_id}_forgetting'] = forgetting
        
        # Average forgetting
        if forgetting_metrics:
            avg_forgetting = np.mean(list(forgetting_metrics.values()))
            forgetting_metrics['average_forgetting'] = avg_forgetting
        
        return forgetting_metrics
    
    def run_continual_learning_experiment(self, mitigation_techniques: List[str]) -> Dict:
        """Run full continual learning experiment with different mitigation techniques"""
        all_results = {}
        
        for technique in mitigation_techniques:
            print(f"\n{'='*60}")
            print(f"RUNNING EXPERIMENT WITH: {self.config.MITIGATION_TECHNIQUES[technique]}")
            print(f"{'='*60}")
            
            # Reset model to original state
            self.student_model.load_state_dict(self.original_student_state)
            self.completed_tasks = []
            self.results = {
                'task_accuracies': defaultdict(list),
                'forgetting_metrics': [],
                'learning_curve': [],
                'final_accuracies': {}
            }
            
            # Train on each task sequentially
            for task_id in range(self.config.num_tasks):
                task_results = self.train_task(task_id, technique)
                self.results['learning_curve'].append(task_results)
            
            # Calculate final metrics
            forgetting_metrics = self.calculate_forgetting_metrics()
            
            # Store results
            all_results[technique] = {
                'task_accuracies': dict(self.results['task_accuracies']),
                'forgetting_metrics': forgetting_metrics,
                'learning_curve': self.results['learning_curve'],
                'final_average_accuracy': self.results['learning_curve'][-1]['average']
            }
        
        return all_results
    
    def save_results(self, results: Dict, filepath: str):
        """Save experimental results"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {filepath}")


def create_continual_learning_plots(results: Dict, save_dir: str = "continual_learning_plots"):
    """Create visualization plots for continual learning results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    techniques = list(results.keys())
    
    # 1. Learning curves for each technique
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, technique in enumerate(techniques):
        if i >= len(axes):
            break
            
        ax = axes[i]
        technique_results = results[technique]
        
        # Plot accuracy over tasks
        for task_id in range(4):  # Assuming 4 tasks
            if task_id in technique_results['task_accuracies']:
                accuracies = technique_results['task_accuracies'][task_id]
                epochs = list(range(len(accuracies)))
                ax.plot(epochs, accuracies, marker='o', label=f'Task {task_id}')
        
        ax.set_title(f'{technique.upper()} - Task Accuracies')
        ax.set_xlabel('Epochs After Task Introduction')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Forgetting comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    forgetting_data = []
    technique_names = []
    
    for technique in techniques:
        forgetting_metrics = results[technique]['forgetting_metrics']
        if 'average_forgetting' in forgetting_metrics:
            forgetting_data.append(forgetting_metrics['average_forgetting'])
            technique_names.append(technique.upper())
    
    if forgetting_data:
        bars = ax.bar(technique_names, forgetting_data)
        ax.set_title('Average Catastrophic Forgetting by Technique')
        ax.set_ylabel('Average Forgetting (Accuracy Drop)')
        ax.set_xlabel('Mitigation Technique')
        
        # Add value labels on bars
        for bar, value in zip(bars, forgetting_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/forgetting_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Final accuracy comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    final_accuracies = [results[technique]['final_average_accuracy'] for technique in techniques]
    technique_names = [technique.upper() for technique in techniques]
    
    bars = ax.bar(technique_names, final_accuracies)
    ax.set_title('Final Average Accuracy by Technique')
    ax.set_ylabel('Final Average Accuracy')
    ax.set_xlabel('Mitigation Technique')
    
    # Add value labels
    for bar, value in zip(bars, final_accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{value:.3f}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/final_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {save_dir}/")


def main():
    """Main function to run continual learning experiments"""
    # Configuration
    config = ContinualLearningConfig()
    
    print("Continual Learning System for Knowledge Distillation")
    print("=" * 60)
    print(f"Number of tasks: {config.num_tasks}")
    print(f"Classes per task: {config.classes_per_task}")
    print(f"Epochs per task: {config.epochs_per_task}")
    print("Available mitigation techniques:")
    for key, desc in config.MITIGATION_TECHNIQUES.items():
        print(f"  - {key}: {desc}")


if __name__ == "__main__":
    main() 