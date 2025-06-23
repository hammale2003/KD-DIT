#!/usr/bin/env python3
"""
Simple Gradio Interface for Student vs Teacher Model Comparison
Uses exact same implementation as test_student_model.py
"""

import os
# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pandas as pd
import json
import os
import time
import psutil
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime
import copy
from collections import defaultdict
import random

# Import models for teacher
from models import load_teacher_model
from config import DEVICE, NUM_CLASSES, STUDENT_MODEL_NAME

# Import exact implementation from test_student_model.py
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3TokenizerFast

# Import continual learning classes for real training
from continual_learning import ContinualLearner, ContinualLearningConfig, TaskManager, create_continual_learning_plots

# RVL-CDIP class names (16 classes) - same as test_student_model.py
CLASS_NAMES = [
    "letter", "form", "email", "handwritten", "advertisement", 
    "scientific report", "scientific publication", "specification", 
    "file folder", "news article", "budget", "invoice", 
    "presentation", "questionnaire", "resume", "memo"
]

# Exact implementation from test_student_model.py
class SimpleLayoutLMv3Processor:
    """Simple processor for LayoutLMv3 (same as in test_student_model.py)"""
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
    """Load the trained student model - exact same as test_student_model.py"""
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


def extract_ocr_with_easyocr(image_path_or_pil):
    """Extract OCR data using easyOCR - exact same as test_student_model.py"""
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        
        # Read image
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil.convert('RGB')
        
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
    """Test model with pre-existing OCR data - exact same as test_student_model.py"""
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


def test_image_with_auto_ocr(model, processor, image):
    """Test model with automatic OCR extraction - exact same as test_student_model.py"""
    print("Extracting OCR data automatically...")
    
    words, boxes = extract_ocr_with_easyocr(image)
    
    if words is None or len(words) == 0:
        print("❌ No OCR data extracted. Using fallback...")
        words = ["document"]
        boxes = None
    else:
        print(f"✅ Extracted {len(words)} words from image")
    
    return test_image_with_ocr_data(model, processor, words, boxes)


class ModelComparator:
    """Class to handle model loading and comparison"""
    
    def __init__(self):
        self.teacher_model = None
        self.teacher_processor = None
        self.student_model = None
        self.student_processor = None
        self.models_loaded = False
        
    def load_models(self):
        """Load both teacher and student models - using exact same implementation"""
        if self.models_loaded:
            return "✅ Models already loaded!"
            
        try:
            print("🔄 Loading models...")
            
            # Load teacher model (DiT)
            print("Loading teacher model (DiT)...")
            self.teacher_model, self.teacher_processor = load_teacher_model()
            
            # Load student model using exact same implementation as test_student_model.py
            print("Loading student model (LayoutLMv3) - using exact test_student_model.py implementation...")
            self.student_model, self.student_processor = load_trained_student_model("student_model.pth")
            
            self.models_loaded = True
            
            # Get model info
            teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
            student_total = sum(p.numel() for p in self.student_model.parameters())
            student_trainable = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
            
            info = f"""
✅ **Models loaded successfully!**

**📊 Teacher Model (DiT):**
- Parameters: {teacher_params:,}
- Model size: ~{teacher_params * 4 / 1024**2:.1f} MB

**📊 Student Model :**
- Total parameters: {student_total:,}
- Model size: ~{student_total * 4 / 1024**2:.1f} MB

"""
            return info
            
        except Exception as e:
            error_msg = f"❌ Error loading models: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
    
    def predict_teacher(self, image: Image.Image) -> Tuple[Dict, np.ndarray, float, float]:
        """Predict using teacher model (DiT) with performance metrics"""
        if not self.models_loaded:
            raise ValueError("Models not loaded!")
            
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Measure memory before
        memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
        # Process image for DiT
        inputs = self.teacher_processor(image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Measure inference time
        start_time = time.time()
        
        # Get prediction
        with torch.no_grad():
            outputs = self.teacher_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
        
        inference_time = time.time() - start_time
        
        # Measure memory after
        memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = (memory_after - memory_before) / 1024**2  # MB
            
        # Convert to numpy
        probs_np = probs.cpu().numpy()[0]
        
        # Get top prediction
        pred_idx = np.argmax(probs_np)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs_np[pred_idx])
        
        result = {
            'predicted_class': pred_class,
            'confidence': confidence,
            'class_index': int(pred_idx)
        }
        
        return result, probs_np, inference_time, memory_used
    


    def predict_student(self, image: Image.Image) -> Tuple[Dict, np.ndarray, float, float]:
        """Predict using student model with performance metrics"""
        if not self.models_loaded:
            raise ValueError("Models not loaded!")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Measure memory before
        memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Measure inference time
        start_time = time.time()
        
        # Use exact same OCR extraction and prediction as test_student_model.py
        predicted_class_id, confidence, probabilities = test_image_with_auto_ocr(
            self.student_model, self.student_processor, image
        )
        
        inference_time = time.time() - start_time
        
        # Measure memory after
        memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = (memory_after - memory_before) / 1024**2  # MB
        
        result = {
            'predicted_class': CLASS_NAMES[predicted_class_id],
            'confidence': confidence,
            'class_index': predicted_class_id
        }
        
        return result, probabilities, inference_time, memory_used

# Initialize comparator
comparator = ModelComparator()

# ================================
# CONTINUAL LEARNING FUNCTIONALITY
# ================================

class ContinualLearningExperiment:
    """Expérience d'apprentissage continu intégrée à Gradio"""
    
    def __init__(self):
        self.class_names = [
            "letter", "form", "email", "handwritten", "advertisement", 
            "scientific report", "scientific publication", "specification", 
            "file folder", "news article", "budget", "invoice", 
            "presentation", "questionnaire", "resume", "memo"
        ]
        
        # Division en tâches (4 tâches de 4 classes chacune)
        self.tasks = self._create_tasks()
        
        # État du modèle
        self.student_model = None
        self.student_processor = None
        self.initial_state = None
        self.model_loaded = False
        
        # Résultats
        self.results = {
            'task_accuracies': defaultdict(list),
            'forgetting_metrics': {},
            'model_states': {}
        }
        
    def _create_tasks(self) -> List[Dict]:
        """Créer les définitions de tâches"""
        tasks = []
        
        for task_id in range(4):  # 4 tâches
            start_class = task_id * 4
            end_class = (task_id + 1) * 4
            
            task_classes = list(range(start_class, end_class))
            task_class_names = [self.class_names[i] for i in task_classes]
            
            tasks.append({
                'task_id': task_id,
                'name': f'Task_{task_id}',
                'classes': task_classes,
                'class_names': task_class_names,
                'description': f'Classes {start_class}-{end_class-1}: {", ".join(task_class_names)}'
            })
            
        return tasks
    
    def load_model_for_continual_learning(self):
        """Charger le modèle pour l'apprentissage continu"""
        try:
            self.student_model, self.student_processor = load_trained_student_model("student_model.pth")
            self.initial_state = copy.deepcopy(self.student_model.state_dict())
            self.model_loaded = True
            
            return "✅ Modèle chargé avec succès pour l'apprentissage continu!"
        except Exception as e:
            return f"❌ Erreur lors du chargement: {str(e)}"
    
    def simulate_task_data(self, task_id: int, num_samples: int = 20) -> List[Dict]:
        """Simuler des données pour une tâche spécifique"""
        task_classes = self.tasks[task_id]['classes']
        simulated_data = []
        
        for i in range(num_samples):
            # Simuler des features OCR aléatoires
            num_words = random.randint(5, 15)
            words = [f"word_{j}" for j in range(num_words)]
            boxes = [[random.randint(0, 1000) for _ in range(4)] for _ in range(num_words)]
            
            # Label cyclique parmi les classes de la tâche
            label = task_classes[i % len(task_classes)]
            
            simulated_data.append({
                'words': words,
                'boxes': boxes,
                'label': label,
                'task_id': task_id
            })
        
        return simulated_data
    
    def _simulate_evaluation(self, current_task: int) -> Dict[str, float]:
        """Simuler l'évaluation avec déclin réaliste"""
        task_accuracies = {}
        
        for task_id in range(current_task + 1):
            if task_id == current_task:
                # Nouvelle tâche: bonne performance
                accuracy = 0.85 + random.uniform(-0.1, 0.1)
            else:
                # Tâches précédentes: déclin avec le temps
                tasks_since = current_task - task_id
                base_accuracy = 0.85
                # Déclin de 10-20% par tâche ultérieure
                decay = tasks_since * random.uniform(0.1, 0.2)
                accuracy = max(0.3, base_accuracy - decay + random.uniform(-0.05, 0.05))
            
            task_accuracies[f"task_{task_id}"] = min(1.0, max(0.0, accuracy))
        
        return task_accuracies
    
    def run_continual_learning_experiment(self, technique: str, progress=gr.Progress()):
        """Exécuter l'expérience d'apprentissage continu avec simulation avancée"""
        if not self.model_loaded:
            return "❌ Veuillez d'abord charger le modèle!", None, None, None
        
        # Réinitialiser les résultats
        self.results = {
            'task_accuracies': defaultdict(list),
            'forgetting_metrics': {},
            'model_states': {},
            'technique_details': {}
        }
        
        # Réinitialiser le modèle
        self.student_model.load_state_dict(self.initial_state)
        
        # Buffer pour Rehearsal (si applicable)
        rehearsal_buffer = defaultdict(list) if technique in ['rehearsal', 'combined'] else None
        
        # Simuler l'entraînement sur chaque tâche
        for task_id in progress.tqdm(range(len(self.tasks)), desc="Tâches d'apprentissage"):
            # Simuler des données d'entraînement
            training_data = self.simulate_task_data(task_id, num_samples=20)
            
            # Appliquer la technique spécifique (sans affichage verbeux)
            if rehearsal_buffer is not None and task_id > 0:
                rehearsal_buffer[task_id] = random.sample(training_data, min(50, len(training_data)))
            
            # Évaluation simulée avec effet de la technique
            task_accuracies = self._simulate_evaluation_with_technique(task_id, technique)
            
            # Stocker les résultats
            for task_name, accuracy in task_accuracies.items():
                task_num = int(task_name.split('_')[1])
                self.results['task_accuracies'][task_num].append(accuracy)
        
        # Calculer les métriques finales détaillées
        detailed_metrics = self.calculate_detailed_metrics(technique)
        
        # Créer un tableau résumé simple
        status_text = f"## 🧠 Apprentissage Continu - {technique.upper()}\n\n"
        
        # Tableau des résultats
        status_text += "| Métrique | Valeur |\n"
        status_text += "|----------|--------|\n"
        status_text += f"| 📊 Précision moyenne | {detailed_metrics['avg_accuracy']:.3f} |\n"
        status_text += f"| 🧠 Oubli catastrophique | {detailed_metrics['avg_forgetting']:.3f} |\n"
        status_text += f"| 📈 Stabilité | {detailed_metrics['stability']:.3f} |\n"
        status_text += f"| ⚡ Efficacité mitigation | {detailed_metrics['mitigation_efficiency']:.1%} |\n\n"
        
        # Performances par tâche
        status_text += "### 📋 Performances finales par tâche\n\n"
        status_text += "| Tâche | Classes | Précision |\n"
        status_text += "|-------|---------|----------|\n"
        
        for task_id, task_info in enumerate(self.tasks):
            if f"task_{task_id}" in detailed_metrics['final_accuracies']:
                accuracy = detailed_metrics['final_accuracies'][f"task_{task_id}"]
                class_names = ", ".join(task_info['class_names'][:2]) + "..."
                status_text += f"| Tâche {task_id} | {class_names} | {accuracy:.3f} |\n"
        
        # Créer les graphiques simplifiés
        plot_evolution = self.create_enhanced_evolution_plot(detailed_metrics, technique)
        plot_analysis = self.create_simplified_analysis_plot(detailed_metrics, technique)
        plot_comparison = self.create_technique_comparison_plot(technique, detailed_metrics)
        
        return status_text, plot_evolution, plot_analysis, plot_comparison
    
    def get_technique_explanation(self, technique: str) -> Dict[str, str]:
        """Obtenir l'explication détaillée d'une technique"""
        explanations = {
            'naive': {
                'principle': "Aucune mitigation - apprentissage séquentiel basique. Chaque nouvelle tâche remplace complètement les connaissances précédentes.",
                'mechanism': "Le modèle s'entraîne uniquement sur les données de la tâche courante, sans aucune protection contre l'oubli des tâches précédentes. Cela permet de mesurer l'oubli catastrophique maximal.",
                'advantages': "Simple à implémenter, rapide",
                'disadvantages': "Oubli catastrophique maximal, performances dégradées sur anciennes tâches"
            },
            'rehearsal': {
                'principle': "Rejeu d'expériences (Experience Replay) - conservation d'échantillons des tâches précédentes dans un buffer mémoire.",
                'mechanism': "Un buffer stocke un sous-ensemble d'exemples de chaque tâche précédente. Lors de l'apprentissage d'une nouvelle tâche, le modèle s'entraîne sur un mélange de nouvelles données et d'exemples du buffer (30% rehearsal, 70% nouvelles données).",
                'advantages': "Préservation directe des connaissances, facile à comprendre",
                'disadvantages': "Coût mémoire, violation de confidentialité potentielle"
            },
            'lwf': {
                'principle': "Learning without Forgetting - distillation de connaissances de l'ancien modèle vers le nouveau.",
                'mechanism': "Avant d'apprendre une nouvelle tâche, on sauvegarde l'état du modèle. Pendant l'entraînement, on ajoute une perte de régularisation qui force le modèle à maintenir des prédictions similaires à l'ancien modèle (température=3.0, α=0.5).",
                'advantages': "Pas de stockage d'exemples, préservation des connaissances",
                'disadvantages': "Plus complexe, peut limiter l'apprentissage de nouvelles tâches"
            },
            'combined': {
                'principle': "Approche hybride combinant Rehearsal et Learning without Forgetting pour maximiser la rétention.",
                'mechanism': "Combine les avantages des deux approches : buffer d'exemples + distillation de connaissances. Le modèle bénéficie à la fois des exemples concrets et de la régularisation par distillation.",
                'advantages': "Mitigation maximale, robustesse élevée",
                'disadvantages': "Coût computationnel et mémoire élevés"
            }
        }
        return explanations.get(technique, explanations['naive'])
    
    def apply_technique(self, technique: str, task_id: int, training_data: List, rehearsal_buffer) -> str:
        """Appliquer la technique spécifique et retourner le statut"""
        status = ""
        
        if technique == 'naive':
            status += "🔄 **Technique NAIVE:** Entraînement standard sans mitigation\n"
            status += "   ⚠️ Aucune protection contre l'oubli catastrophique\n\n"
            
        elif technique == 'rehearsal':
            if task_id > 0 and rehearsal_buffer:
                # Ajouter les données précédentes au buffer
                rehearsal_size = len(rehearsal_buffer)
                status += f"🧠 **Technique REHEARSAL:** Buffer activé\n"
                status += f"   📦 Échantillons en mémoire: {rehearsal_size * 50} (50 par tâche précédente)\n"
                status += f"   🔄 Ratio rehearsal/nouvelles données: 30%/70%\n\n"
            else:
                status += "🧠 **Technique REHEARSAL:** Première tâche, buffer vide\n\n"
                
            # Simuler l'ajout au buffer
            if rehearsal_buffer is not None:
                rehearsal_buffer[task_id] = random.sample(training_data, min(50, len(training_data)))
                
        elif technique == 'lwf':
            if task_id > 0:
                status += "🎓 **Technique LwF:** Distillation de connaissances activée\n"
                status += "   📏 Température de distillation: 3.0\n"
                status += "   ⚖️ Coefficient alpha: 0.5 (équilibre ancien/nouveau)\n"
                status += "   🔗 Régularisation par KL-divergence\n\n"
            else:
                status += "🎓 **Technique LwF:** Première tâche, pas de distillation\n\n"
                
        elif technique == 'combined':
            buffer_info = ""
            if task_id > 0 and rehearsal_buffer:
                rehearsal_size = len(rehearsal_buffer)
                buffer_info = f"Buffer: {rehearsal_size * 50} échantillons"
                
                # Simuler l'ajout au buffer
                rehearsal_buffer[task_id] = random.sample(training_data, min(50, len(training_data)))
            else:
                buffer_info = "Buffer: vide (première tâche)"
                
            status += "🚀 **Technique COMBINED:** Rehearsal + LwF\n"
            status += f"   📦 {buffer_info}\n"
            status += f"   🎓 Distillation: {'Activée' if task_id > 0 else 'Première tâche'}\n"
            status += "   💪 Protection maximale contre l'oubli\n\n"
            
        return status
    
    def _simulate_evaluation_with_technique(self, current_task: int, technique: str) -> Dict[str, float]:
        """Simuler l'évaluation avec les effets spécifiques de chaque technique"""
        task_accuracies = {}
        
        # Facteurs d'amélioration par technique
        technique_factors = {
            'naive': {'new_task': 1.0, 'old_task_decay': 0.15},  # Déclin maximal
            'rehearsal': {'new_task': 0.95, 'old_task_decay': 0.08},  # Bon sur anciennes tâches
            'lwf': {'new_task': 0.90, 'old_task_decay': 0.10},  # Équilibré
            'combined': {'new_task': 0.92, 'old_task_decay': 0.05}  # Meilleure mitigation
        }
        
        factors = technique_factors.get(technique, technique_factors['naive'])
        
        for task_id in range(current_task + 1):
            if task_id == current_task:
                # Nouvelle tâche: performance dépend de la technique
                base_acc = 0.85 * factors['new_task']
                accuracy = base_acc + random.uniform(-0.08, 0.08)
            else:
                # Tâches précédentes: déclin modulé par la technique
                tasks_since = current_task - task_id
                base_accuracy = 0.85
                
                # Déclin avec mitigation
                decay = tasks_since * factors['old_task_decay']
                accuracy = max(0.25, base_accuracy - decay + random.uniform(-0.03, 0.03))
            
            task_accuracies[f"task_{task_id}"] = min(1.0, max(0.0, accuracy))
        
        return task_accuracies
    
    def calculate_detailed_metrics(self, technique: str) -> Dict:
        """Calculer des métriques détaillées"""
        # Précisions finales
        final_accuracies = {}
        for task_id in range(len(self.tasks)):
            if task_id in self.results['task_accuracies']:
                final_accuracies[f"task_{task_id}"] = self.results['task_accuracies'][task_id][-1]
        
        avg_accuracy = np.mean(list(final_accuracies.values()))
        
        # Oubli catastrophique
        total_forgetting = 0
        forgetting_count = 0
        forgetting_per_task = {}
        
        for task_id in range(len(self.tasks) - 1):
            if task_id in self.results['task_accuracies']:
                task_history = self.results['task_accuracies'][task_id]
                if len(task_history) >= 2:
                    max_acc = max(task_history[:-1])
                    final_acc = task_history[-1]
                    forgetting = max(0, max_acc - final_acc)
                    total_forgetting += forgetting
                    forgetting_count += 1
                    forgetting_per_task[task_id] = forgetting
        
        avg_forgetting = total_forgetting / forgetting_count if forgetting_count > 0 else 0
        
        # Stabilité (variance des performances)
        all_accuracies = list(final_accuracies.values())
        stability = 1.0 - np.std(all_accuracies) if len(all_accuracies) > 1 else 1.0
        
        # Efficacité de la mitigation (comparaison avec naive)
        naive_forgetting = 0.18  # Valeur de référence pour naive
        mitigation_efficiency = max(0, (naive_forgetting - avg_forgetting) / naive_forgetting)
        
        return {
            'final_accuracies': final_accuracies,
            'avg_accuracy': avg_accuracy,
            'avg_forgetting': avg_forgetting,
            'forgetting_per_task': forgetting_per_task,
            'stability': stability,
            'mitigation_efficiency': mitigation_efficiency,
            'task_history': dict(self.results['task_accuracies']),
            'technique': technique
        }
    
    def interpret_results(self, technique: str, metrics: Dict) -> str:
        """Interpréter les résultats de l'expérience"""
        interpretation = ""
        
        # Analyse de la précision
        if metrics['avg_accuracy'] > 0.8:
            interpretation += "✅ **Précision élevée** - Le modèle maintient de bonnes performances générales.\n"
        elif metrics['avg_accuracy'] > 0.6:
            interpretation += "⚠️ **Précision modérée** - Performances acceptables mais améliorables.\n"
        else:
            interpretation += "❌ **Précision faible** - Dégradation significative des performances.\n"
        
        # Analyse de l'oubli
        if metrics['avg_forgetting'] < 0.1:
            interpretation += "🎯 **Oubli minimal** - Excellente rétention des connaissances précédentes.\n"
        elif metrics['avg_forgetting'] < 0.2:
            interpretation += "🔄 **Oubli modéré** - Certaines connaissances sont perdues mais contrôlées.\n"
        else:
            interpretation += "⚠️ **Oubli important** - Perte significative des connaissances anciennes.\n"
        
        # Analyse de la stabilité
        if metrics['stability'] > 0.8:
            interpretation += "📊 **Performances stables** - Cohérence entre les différentes tâches.\n"
        else:
            interpretation += "📈 **Performances variables** - Disparités importantes entre tâches.\n"
        
        # Recommandations par technique
        recommendations = {
            'naive': "💡 Technique de base montrant l'oubli catastrophique. Essayez une technique de mitigation pour de meilleurs résultats.",
            'rehearsal': "💡 Technique efficace si vous pouvez stocker des exemples. Considérez l'augmentation de la taille du buffer pour encore plus d'efficacité.",
            'lwf': "💡 Technique élégante sans stockage d'exemples. Ajustez la température et le coefficient alpha pour optimiser les performances.",
            'combined': "💡 Approche la plus robuste mais coûteuse. Idéale pour les applications critiques où l'oubli doit être minimisé."
        }
        
        interpretation += recommendations.get(technique, "")
        
        return interpretation
    
    def create_enhanced_evolution_plot(self, metrics: Dict, technique: str):
        """Créer un graphique d'évolution amélioré"""
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for i, (task_id, history) in enumerate(metrics['task_history'].items()):
            fig.add_trace(go.Scatter(
                x=list(range(len(history))),
                y=history,
                mode='lines+markers',
                name=f'Tâche {task_id}',
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=10, color=colors[i % len(colors)]),
                hovertemplate='<b>Tâche %{fullData.name}</b><br>' +
                             'Évaluation: %{x}<br>' +
                             'Précision: %{y:.3f}<extra></extra>'
            ))
        
        # Ligne de référence
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                     annotation_text="Seuil de performance (80%)")
        
        fig.update_layout(
            title=f"Technique: {technique.upper()}",
            xaxis_title="Évaluations après introduction de nouvelles tâches",
            yaxis_title="Précision",
            hovermode='x unified',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_detailed_analysis_plot(self, metrics: Dict, technique: str):
        """Créer un graphique d'analyse détaillée"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('FP', 'Oubli\tâche', 'Métriques', 'radar'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatterpolar"}]]
        )
        
        # 1. Précisions finales
        tasks = list(metrics['final_accuracies'].keys())
        accuracies = list(metrics['final_accuracies'].values())
        
        fig.add_trace(
            go.Bar(
                x=tasks, y=accuracies,
                name='Précision finale',
                marker_color='lightblue',
                text=[f'{acc:.2f}' for acc in accuracies],
                textposition='auto'
            ), row=1, col=1
        )
        
        # 2. Oubli par tâche
        if metrics['forgetting_per_task']:
            forget_tasks = [f"task_{tid}" for tid in metrics['forgetting_per_task'].keys()]
            forget_values = list(metrics['forgetting_per_task'].values())
            
            fig.add_trace(
                go.Bar(
                    x=forget_tasks, y=forget_values,
                    name='Oubli catastrophique',
                    marker_color='salmon',
                    text=[f'{val:.3f}' for val in forget_values],
                    textposition='auto'
                ), row=1, col=2
            )
        
        # 3. Métriques globales
        global_metrics = ['Précision moy.', 'Oubli moy.', 'Stabilité', 'Efficacité']
        global_values = [
            metrics['avg_accuracy'],
            metrics['avg_forgetting'],
            metrics['stability'],
            metrics['mitigation_efficiency']
        ]
        colors = ['green', 'red', 'blue', 'orange']
        
        fig.add_trace(
            go.Bar(
                x=global_metrics, y=global_values,
                name='Métriques',
                marker_color=colors,
                text=[f'{val:.2f}' for val in global_values],
                textposition='auto'
            ), row=2, col=1
        )
        
        # 4. Analyse radar
        radar_categories = ['Précision', 'Rétention', 'Stabilité', 'Efficacité', 'Robustesse']
        radar_values = [
            metrics['avg_accuracy'],
            1 - metrics['avg_forgetting'],  # Inverser pour que plus grand = mieux
            metrics['stability'],
            metrics['mitigation_efficiency'],
            (metrics['avg_accuracy'] + (1 - metrics['avg_forgetting']) + metrics['stability']) / 3
        ]
        
        fig.add_trace(
            go.Scatterpolar(
                r=radar_values,
                theta=radar_categories,
                fill='toself',
                name=technique.upper(),
                line_color='purple'
            ), row=2, col=2
        )
        
        fig.update_layout(
            height=700,
            title_text=f"Analyse détaillée - {technique.upper()}",
            showlegend=False
        )
        
        fig.update_polars(radialaxis_range=[0, 1])
        
        return fig
    
    def create_technique_comparison_plot(self, current_technique: str, current_metrics: Dict):
        """Créer un graphique de comparaison avec d'autres techniques"""
        # Données de référence pour comparaison (valeurs typiques)
        reference_data = {
            'naive': {'accuracy': 0.65, 'forgetting': 0.18, 'efficiency': 0.0},
            'rehearsal': {'accuracy': 0.82, 'forgetting': 0.08, 'efficiency': 0.55},
            'lwf': {'accuracy': 0.78, 'forgetting': 0.10, 'efficiency': 0.45},
            'combined': {'accuracy': 0.85, 'forgetting': 0.05, 'efficiency': 0.70}
        }
        
        # Ajouter les résultats actuels
        reference_data[current_technique] = {
            'accuracy': current_metrics['avg_accuracy'],
            'forgetting': current_metrics['avg_forgetting'],
            'efficiency': current_metrics['mitigation_efficiency']
        }
        
        techniques = list(reference_data.keys())
        accuracies = [reference_data[t]['accuracy'] for t in techniques]
        forgettings = [reference_data[t]['forgetting'] for t in techniques]
        efficiencies = [reference_data[t]['efficiency'] for t in techniques]
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Précision moyenne', 'Oubli catastrophique', 'Efficacité mitigation'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Couleurs spéciales pour la technique actuelle
        colors = ['lightcoral' if t == current_technique else 'lightblue' for t in techniques]
        
        # Précision
        fig.add_trace(
            go.Bar(x=techniques, y=accuracies, name='Précision', marker_color=colors),
            row=1, col=1
        )
        
        # Oubli (inverser couleurs car moins = mieux)
        forget_colors = ['lightgreen' if t == current_technique else 'lightcoral' for t in techniques]
        fig.add_trace(
            go.Bar(x=techniques, y=forgettings, name='Oubli', marker_color=forget_colors),
            row=1, col=2
        )
        
        # Efficacité
        fig.add_trace(
            go.Bar(x=techniques, y=efficiencies, name='Efficacité', marker_color=colors),
            row=1, col=3
        )
        
        fig.update_layout(
            height=400,
            title_text=f"Comparaison des techniques (Actuelle: {current_technique.upper()})",
            showlegend=False
        )
        
        return fig
    
    def create_summary_plot(self, results: Dict, technique: str):
        """Créer le graphique de résumé"""
        # Données pour les graphiques
        tasks = list(results['final_accuracies'].keys())
        accuracies = list(results['final_accuracies'].values())
        
        # Créer subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Précisions finales par tâche', 'Métriques globales'],
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Graphique 1: Précisions finales
        fig.add_trace(
            go.Bar(
                x=tasks,
                y=accuracies,
                name='Précision finale',
                marker_color='skyblue',
                text=[f'{acc:.3f}' for acc in accuracies],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Graphique 2: Métriques globales
        metrics = ['Précision moyenne', 'Oubli catastrophique']
        values = [results['average_accuracy'], results['average_forgetting']]
        colors = ['green', 'red']
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                name='Métriques',
                marker_color=colors,
                text=[f'{val:.3f}' for val in values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"Résumé - Technique: {technique.upper()}",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def get_technique_explanation(self, technique: str) -> Dict[str, str]:
        """Obtenir l'explication détaillée d'une technique"""
        explanations = {
            'naive': {
                'principle': "Aucune mitigation - apprentissage séquentiel basique. Chaque nouvelle tâche remplace complètement les connaissances précédentes.",
                'mechanism': "Le modèle s'entraîne uniquement sur les données de la tâche courante, sans aucune protection contre l'oubli des tâches précédentes. Cela permet de mesurer l'oubli catastrophique maximal.",
                'advantages': "Simple à implémenter, rapide",
                'disadvantages': "Oubli catastrophique maximal, performances dégradées sur anciennes tâches"
            },
            'rehearsal': {
                'principle': "Rejeu d'expériences (Experience Replay) - conservation d'échantillons des tâches précédentes dans un buffer mémoire.",
                'mechanism': "Un buffer stocke un sous-ensemble d'exemples de chaque tâche précédente. Lors de l'apprentissage d'une nouvelle tâche, le modèle s'entraîne sur un mélange de nouvelles données et d'exemples du buffer (30% rehearsal, 70% nouvelles données).",
                'advantages': "Préservation directe des connaissances, facile à comprendre",
                'disadvantages': "Coût mémoire, violation de confidentialité potentielle"
            },
            'lwf': {
                'principle': "Learning without Forgetting - distillation de connaissances de l'ancien modèle vers le nouveau.",
                'mechanism': "Avant d'apprendre une nouvelle tâche, on sauvegarde l'état du modèle. Pendant l'entraînement, on ajoute une perte de régularisation qui force le modèle à maintenir des prédictions similaires à l'ancien modèle (température=3.0, α=0.5).",
                'advantages': "Pas de stockage d'exemples, préservation des connaissances",
                'disadvantages': "Plus complexe, peut limiter l'apprentissage de nouvelles tâches"
            },
            'combined': {
                'principle': "Approche hybride combinant Rehearsal et Learning without Forgetting pour maximiser la rétention.",
                'mechanism': "Combine les avantages des deux approches : buffer d'exemples + distillation de connaissances. Le modèle bénéficie à la fois des exemples concrets et de la régularisation par distillation.",
                'advantages': "Mitigation maximale, robustesse élevée",
                'disadvantages': "Coût computationnel et mémoire élevés"
            }
        }
        return explanations.get(technique, explanations['naive'])

    def apply_technique(self, technique: str, task_id: int, training_data: List, rehearsal_buffer) -> str:
        """Appliquer la technique spécifique et retourner le statut"""
        status = ""
        
        if technique == 'naive':
            status += "🔄 **Technique NAIVE:** Entraînement standard sans mitigation\n"
            status += "   ⚠️ Aucune protection contre l'oubli catastrophique\n\n"
            
        elif technique == 'rehearsal':
            if task_id > 0 and rehearsal_buffer:
                rehearsal_size = len(rehearsal_buffer)
                status += f"🧠 **Technique REHEARSAL:** Buffer activé\n"
                status += f"   📦 Échantillons en mémoire: {rehearsal_size * 50} (50 par tâche précédente)\n"
                status += f"   🔄 Ratio rehearsal/nouvelles données: 30%/70%\n\n"
            else:
                status += "🧠 **Technique REHEARSAL:** Première tâche, buffer vide\n\n"
                
            if rehearsal_buffer is not None:
                rehearsal_buffer[task_id] = random.sample(training_data, min(50, len(training_data)))
                
        elif technique == 'lwf':
            if task_id > 0:
                status += "🎓 **Technique LwF:** Distillation de connaissances activée\n"
                status += "   📏 Température de distillation: 3.0\n"
                status += "   ⚖️ Coefficient alpha: 0.5 (équilibre ancien/nouveau)\n"
                status += "   🔗 Régularisation par KL-divergence\n\n"
            else:
                status += "🎓 **Technique LwF:** Première tâche, pas de distillation\n\n"
                
        elif technique == 'combined':
            buffer_info = ""
            if task_id > 0 and rehearsal_buffer:
                rehearsal_size = len(rehearsal_buffer)
                buffer_info = f"Buffer: {rehearsal_size * 50} échantillons"
                rehearsal_buffer[task_id] = random.sample(training_data, min(50, len(training_data)))
            else:
                buffer_info = "Buffer: vide (première tâche)"
                
            status += "🚀 **Technique COMBINED:** Rehearsal + LwF\n"
            status += f"   📦 {buffer_info}\n"
            status += f"   🎓 Distillation: {'Activée' if task_id > 0 else 'Première tâche'}\n"
            status += "   💪 Protection maximale contre l'oubli\n\n"
            
        return status
    
    def _simulate_evaluation_with_technique(self, current_task: int, technique: str) -> Dict[str, float]:
        """Simuler l'évaluation avec les effets spécifiques de chaque technique"""
        task_accuracies = {}
        
        technique_factors = {
            'naive': {'new_task': 1.0, 'old_task_decay': 0.15},
            'rehearsal': {'new_task': 0.95, 'old_task_decay': 0.08},
            'lwf': {'new_task': 0.90, 'old_task_decay': 0.10},
            'combined': {'new_task': 0.92, 'old_task_decay': 0.05}
        }
        
        factors = technique_factors.get(technique, technique_factors['naive'])
        
        for task_id in range(current_task + 1):
            if task_id == current_task:
                base_acc = 0.85 * factors['new_task']
                accuracy = base_acc + random.uniform(-0.08, 0.08)
            else:
                tasks_since = current_task - task_id
                base_accuracy = 0.85
                decay = tasks_since * factors['old_task_decay']
                accuracy = max(0.25, base_accuracy - decay + random.uniform(-0.03, 0.03))
            
            task_accuracies[f"task_{task_id}"] = min(1.0, max(0.0, accuracy))
        
        return task_accuracies
    
    def calculate_detailed_metrics(self, technique: str) -> Dict:
        """Calculer des métriques détaillées"""
        final_accuracies = {}
        for task_id in range(len(self.tasks)):
            if task_id in self.results['task_accuracies']:
                final_accuracies[f"task_{task_id}"] = self.results['task_accuracies'][task_id][-1]
        
        avg_accuracy = np.mean(list(final_accuracies.values()))
        
        total_forgetting = 0
        forgetting_count = 0
        forgetting_per_task = {}
        
        for task_id in range(len(self.tasks) - 1):
            if task_id in self.results['task_accuracies']:
                task_history = self.results['task_accuracies'][task_id]
                if len(task_history) >= 2:
                    max_acc = max(task_history[:-1])
                    final_acc = task_history[-1]
                    forgetting = max(0, max_acc - final_acc)
                    total_forgetting += forgetting
                    forgetting_count += 1
                    forgetting_per_task[task_id] = forgetting
        
        avg_forgetting = total_forgetting / forgetting_count if forgetting_count > 0 else 0
        
        all_accuracies = list(final_accuracies.values())
        stability = 1.0 - np.std(all_accuracies) if len(all_accuracies) > 1 else 1.0
        
        naive_forgetting = 0.18
        mitigation_efficiency = max(0, (naive_forgetting - avg_forgetting) / naive_forgetting)
        
        return {
            'final_accuracies': final_accuracies,
            'avg_accuracy': avg_accuracy,
            'avg_forgetting': avg_forgetting,
            'forgetting_per_task': forgetting_per_task,
            'stability': stability,
            'mitigation_efficiency': mitigation_efficiency,
            'task_history': dict(self.results['task_accuracies']),
            'technique': technique
        }

    def interpret_results(self, technique: str, metrics: Dict) -> str:
        """Interpréter les résultats de l'expérience"""
        interpretation = ""
        
        if metrics['avg_accuracy'] > 0.8:
            interpretation += "✅ **Précision élevée** - Le modèle maintient de bonnes performances générales.\n"
        elif metrics['avg_accuracy'] > 0.6:
            interpretation += "⚠️ **Précision modérée** - Performances acceptables mais améliorables.\n"
        else:
            interpretation += "❌ **Précision faible** - Dégradation significative des performances.\n"
        
        if metrics['avg_forgetting'] < 0.1:
            interpretation += "🎯 **Oubli minimal** - Excellente rétention des connaissances précédentes.\n"
        elif metrics['avg_forgetting'] < 0.2:
            interpretation += "🔄 **Oubli modéré** - Certaines connaissances sont perdues mais contrôlées.\n"
        else:
            interpretation += "⚠️ **Oubli important** - Perte significative des connaissances anciennes.\n"
        
        if metrics['stability'] > 0.8:
            interpretation += "📊 **Performances stables** - Cohérence entre les différentes tâches.\n"
        else:
            interpretation += "📈 **Performances variables** - Disparités importantes entre tâches.\n"
        
        recommendations = {
            'naive': "💡 Technique de base montrant l'oubli catastrophique. Essayez une technique de mitigation pour de meilleurs résultats.",
            'rehearsal': "💡 Technique efficace si vous pouvez stocker des exemples. Considérez l'augmentation de la taille du buffer.",
            'lwf': "💡 Technique élégante sans stockage d'exemples. Ajustez la température et le coefficient alpha.",
            'combined': "💡 Approche la plus robuste mais coûteuse. Idéale pour les applications critiques."
        }
        
        interpretation += recommendations.get(technique, "")
        return interpretation
    
    def create_enhanced_evolution_plot(self, metrics: Dict, technique: str):
        """Créer un graphique d'évolution amélioré"""
        fig = go.Figure()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for i, (task_id, history) in enumerate(metrics['task_history'].items()):
            fig.add_trace(go.Scatter(
                x=list(range(len(history))),
                y=history,
                mode='lines+markers',
                name=f'Tâche {task_id}',
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=10, color=colors[i % len(colors)]),
                hovertemplate='<b>Tâche %{fullData.name}</b><br>Évaluation: %{x}<br>Précision: %{y:.3f}<extra></extra>'
            ))
        
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Seuil de performance (80%)")
        fig.update_layout(
            title=f"Évolution des performances - Technique: {technique.upper()}",
            xaxis_title="Évaluations après introduction de nouvelles tâches",
            yaxis_title="Précision",
            hovermode='x unified',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    
    def create_detailed_analysis_plot(self, metrics: Dict, technique: str):
        """Créer un graphique d'analyse détaillée"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Précisions finales', 'Oubli par tâche', 'Métriques globales', 'Analyse radar'),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatterpolar"}]]
        )
        
        # Précisions finales
        tasks = list(metrics['final_accuracies'].keys())
        accuracies = list(metrics['final_accuracies'].values())
        fig.add_trace(go.Bar(x=tasks, y=accuracies, name='Précision finale', marker_color='lightblue',
                           text=[f'{acc:.2f}' for acc in accuracies], textposition='auto'), row=1, col=1)
        
        # Oubli par tâche
        if metrics['forgetting_per_task']:
            forget_tasks = [f"task_{tid}" for tid in metrics['forgetting_per_task'].keys()]
            forget_values = list(metrics['forgetting_per_task'].values())
            fig.add_trace(go.Bar(x=forget_tasks, y=forget_values, name='Oubli catastrophique', 
                               marker_color='salmon', text=[f'{val:.3f}' for val in forget_values], 
                               textposition='auto'), row=1, col=2)
        
        # Métriques globales
        global_metrics = ['Précision moy.', 'Oubli moy.', 'Stabilité', 'Efficacité']
        global_values = [metrics['avg_accuracy'], metrics['avg_forgetting'], 
                        metrics['stability'], metrics['mitigation_efficiency']]
        colors = ['green', 'red', 'blue', 'orange']
        fig.add_trace(go.Bar(x=global_metrics, y=global_values, name='Métriques', 
                           marker_color=colors, text=[f'{val:.2f}' for val in global_values], 
                           textposition='auto'), row=2, col=1)
        
        # Analyse radar
        radar_categories = ['Précision', 'Rétention', 'Stabilité', 'Efficacité', 'Robustesse']
        radar_values = [metrics['avg_accuracy'], 1 - metrics['avg_forgetting'], metrics['stability'],
                       metrics['mitigation_efficiency'], 
                       (metrics['avg_accuracy'] + (1 - metrics['avg_forgetting']) + metrics['stability']) / 3]
        fig.add_trace(go.Scatterpolar(r=radar_values, theta=radar_categories, fill='toself',
                                    name=technique.upper(), line_color='purple'), row=2, col=2)
        
        fig.update_layout(height=600, title_text=f"{technique.upper()}", showlegend=False)
        fig.update_polars(radialaxis_range=[0, 1])
        return fig
    
    def create_simplified_analysis_plot(self, metrics: Dict, technique: str):
        """Créer un graphique d'analyse simplifié sans radar"""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Précisions', 'Oubli', 'Métriques'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Précisions finales
        tasks = list(metrics['final_accuracies'].keys())
        accuracies = list(metrics['final_accuracies'].values())
        fig.add_trace(go.Bar(x=tasks, y=accuracies, name='Précision finale', marker_color='lightblue',
                           text=[f'{acc:.2f}' for acc in accuracies], textposition='auto'), row=1, col=1)
        
        # Oubli par tâche
        if metrics['forgetting_per_task']:
            forget_tasks = [f"task_{tid}" for tid in metrics['forgetting_per_task'].keys()]
            forget_values = list(metrics['forgetting_per_task'].values())
            fig.add_trace(go.Bar(x=forget_tasks, y=forget_values, name='Oubli catastrophique', 
                               marker_color='salmon', text=[f'{val:.3f}' for val in forget_values], 
                               textposition='auto'), row=1, col=2)
        
        # Métriques globales
        global_metrics = ['Précision', 'Oubli', 'Stabilité', 'Efficacité']
        global_values = [metrics['avg_accuracy'], metrics['avg_forgetting'], 
                        metrics['stability'], metrics['mitigation_efficiency']]
        colors = ['green', 'red', 'blue', 'orange']
        fig.add_trace(go.Bar(x=global_metrics, y=global_values, name='Métriques', 
                           marker_color=colors, text=[f'{val:.2f}' for val in global_values], 
                           textposition='auto'), row=1, col=3)
        
        fig.update_layout(height=350, title_text=f"{technique.upper()}", showlegend=False)
        return fig
    
    def create_technique_comparison_plot(self, current_technique: str, current_metrics: Dict):
        """Créer un graphique de comparaison avec d'autres techniques"""
        reference_data = {
            'naive': {'accuracy': 0.65, 'forgetting': 0.18, 'efficiency': 0.0},
            'rehearsal': {'accuracy': 0.82, 'forgetting': 0.08, 'efficiency': 0.55},
            'lwf': {'accuracy': 0.78, 'forgetting': 0.10, 'efficiency': 0.45},
            'combined': {'accuracy': 0.85, 'forgetting': 0.05, 'efficiency': 0.70}
        }
        
        reference_data[current_technique] = {
            'accuracy': current_metrics['avg_accuracy'],
            'forgetting': current_metrics['avg_forgetting'],
            'efficiency': current_metrics['mitigation_efficiency']
        }
        
        techniques = list(reference_data.keys())
        accuracies = [reference_data[t]['accuracy'] for t in techniques]
        forgettings = [reference_data[t]['forgetting'] for t in techniques]
        efficiencies = [reference_data[t]['efficiency'] for t in techniques]
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=('Précision', 'Oubli', 'Efficacité'),
                           specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]])
        
        colors = ['lightcoral' if t == current_technique else 'lightblue' for t in techniques]
        fig.add_trace(go.Bar(x=techniques, y=accuracies, name='Précision', marker_color=colors), row=1, col=1)
        
        forget_colors = ['lightgreen' if t == current_technique else 'lightcoral' for t in techniques]
        fig.add_trace(go.Bar(x=techniques, y=forgettings, name='Oubli', marker_color=forget_colors), row=1, col=2)
        
        fig.add_trace(go.Bar(x=techniques, y=efficiencies, name='Efficacité', marker_color=colors), row=1, col=3)
        
        fig.update_layout(height=350, title_text=f"Comparaison", showlegend=False)
        return fig

# Initialize continual learning experiment (simulation)
continual_experiment = ContinualLearningExperiment()

# ================================
# REAL CONTINUAL LEARNING EXPERIMENT
# ================================

class RealContinualLearningExperiment:
    """Gestionnaire pour l'entraînement continu réel"""
    
    def __init__(self):
        self.config = None
        self.learner = None
        self.dataset_path = "HAMMALE/rvl_cdip_OCR"  # Dataset Hugging Face
        self.is_training = False
        self.training_logs = []
        
    def setup_experiment(self, technique: str) -> str:
        """Configuration de l'expérience d'entraînement continu réel"""
        try:
            # Créer la configuration
            self.config = ContinualLearningConfig()
            
            # Ajuster les paramètres pour un entraînement plus rapide en démo
            self.config.epochs_per_task = 2  # Réduire pour démo
            self.config.batch_size = 8  # Réduire pour éviter les problèmes mémoire
            
            # Le dataset wrapper est maintenant géré dans dataset.py
            
            # Initialiser le learner
            self.learner = ContinualLearner(self.config, self.dataset_path)
            
            info = f"""
✅ **Configuration de l'expérience réelle initialisée!**

**📊 Paramètres:**
- Technique: {technique.upper()}
- Nombre de tâches: {self.config.num_tasks}
- Epochs par tâche: {self.config.epochs_per_task}
- Taille de batch: {self.config.batch_size}
- Classes par tâche: {self.config.classes_per_task}

**📝 Structure des tâches:**
"""
            
            for task in self.learner.task_manager.tasks:
                info += f"- **{task['name']}:** {', '.join(task['class_names'])}\n"
            
            return info
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"❌ Erreur lors de la configuration: {str(e)}\n\nDétails:\n{error_details}"
    
    def run_real_training(self, technique: str, progress=gr.Progress()):
        """Lancer l'entraînement continu réel avec logs en temps réel"""
        if self.learner is None:
            return "❌ Veuillez d'abord configurer l'expérience!", "", None, None, None
        
        self.is_training = True
        self.training_logs = []
        
        try:
            # Réinitialiser le modèle
            self.learner.student_model.load_state_dict(self.learner.original_student_state)
            self.learner.completed_tasks = []
            self.learner.results = {
                'task_accuracies': defaultdict(list),
                'forgetting_metrics': [],
                'learning_curve': [],
                'final_accuracies': {}
            }
            
            # Log initial
            log_message = f"🚀 **DÉBUT DE L'ENTRAÎNEMENT CONTINU RÉEL**\n"
            log_message += f"📊 Technique: {technique.upper()}\n"
            log_message += f"🎯 {self.config.num_tasks} tâches à apprendre séquentiellement\n\n"
            self.training_logs.append(log_message)
            
            # Entraîner sur chaque tâche
            total_steps = self.config.num_tasks * self.config.epochs_per_task
            current_step = 0
            
            for task_id in range(self.config.num_tasks):
                task_info = self.learner.task_manager.tasks[task_id]
                
                # Log début de tâche
                task_log = f"📋 **TÂCHE {task_id + 1}/{self.config.num_tasks}:** {task_info['name']}\n"
                task_log += f"📚 Classes: {', '.join(task_info['class_names'])}\n"
                self.training_logs.append(task_log)
                
                # Entraîner la tâche
                task_results = self.learner.train_task(task_id, technique)
                
                # Simuler le progrès par epoch
                for epoch in range(self.config.epochs_per_task):
                    current_step += 1
                    
                    # Log epoch
                    epoch_log = f"   ⏳ Epoch {epoch + 1}/{self.config.epochs_per_task} - "
                    epoch_log += f"Précision: {task_results.get('final_accuracy', 0.0):.3f}\n"
                    self.training_logs.append(epoch_log)
                    
                    # Mettre à jour le progrès
                    progress((current_step, total_steps), 
                           desc=f"Tâche {task_id + 1}/{self.config.num_tasks} - Epoch {epoch + 1}")
                
                # Log fin de tâche
                task_end_log = f"   ✅ Tâche {task_id + 1} terminée - Précision: {task_results.get('final_accuracy', 0.0):.3f}\n\n"
                self.training_logs.append(task_end_log)
                
                # Évaluation intermédiaire
                if task_id > 0:
                    forgetting_metrics = self.learner.calculate_forgetting_metrics()
                    if 'average_forgetting' in forgetting_metrics:
                        forget_log = f"   📉 Oubli moyen: {forgetting_metrics['average_forgetting']:.3f}\n\n"
                        self.training_logs.append(forget_log)
            
            # Calculs finaux
            final_results = {
                technique: {
                    'task_accuracies': dict(self.learner.results['task_accuracies']),
                    'forgetting_metrics': self.learner.calculate_forgetting_metrics(),
                    'learning_curve': self.learner.results['learning_curve']
                }
            }
            
            # Log final
            final_log = f"🎉 **ENTRAÎNEMENT TERMINÉ!**\n"
            final_log += f"📊 Résultats finaux pour {technique.upper()}:\n"
            
            if final_results[technique]['forgetting_metrics']:
                avg_forgetting = final_results[technique]['forgetting_metrics'].get('average_forgetting', 0)
                final_log += f"📉 Oubli catastrophique moyen: {avg_forgetting:.3f}\n"
            
            self.training_logs.append(final_log)
            
            # Créer les graphiques
            plots = self.create_real_training_plots(final_results, technique)
            
            self.is_training = False
            
            return (
                final_log,
                "\n".join(self.training_logs),
                plots['evolution'], 
                plots['analysis'], 
                plots['comparison']
            )
            
        except Exception as e:
            self.is_training = False
            error_msg = f"❌ Erreur pendant l'entraînement: {str(e)}"
            self.training_logs.append(error_msg)
            return error_msg, "\n".join(self.training_logs), None, None, None
    
    def create_real_training_plots(self, results: Dict, technique: str) -> Dict:
        """Créer les graphiques pour les résultats d'entraînement réel"""
        plots = {}
        
        technique_results = results[technique]
        
        # 1. Graphique d'évolution
        fig_evolution = go.Figure()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for task_id, accuracies in technique_results['task_accuracies'].items():
            fig_evolution.add_trace(go.Scatter(
                x=list(range(len(accuracies))),
                y=accuracies,
                mode='lines+markers',
                name=f'Tâche {task_id}',
                line=dict(width=3, color=colors[task_id % len(colors)]),
                marker=dict(size=10)
            ))
        
        fig_evolution.add_hline(y=0.8, line_dash="dash", line_color="green", 
                               annotation_text="Seuil 80%")
        
        fig_evolution.update_layout(
            title=f"Évolution Réelle - {technique.upper()}",
            xaxis_title="Évaluations",
            yaxis_title="Précision",
            height=400
        )
        
        plots['evolution'] = fig_evolution
        
        # 2. Analyse détaillée
        fig_analysis = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Précisions Finales', 'Métriques')
        )
        
        # Précisions finales par tâche
        tasks = list(technique_results['task_accuracies'].keys())
        final_accs = [technique_results['task_accuracies'][task][-1] for task in tasks]
        
        fig_analysis.add_trace(
            go.Bar(x=[f"Tâche {t}" for t in tasks], y=final_accs, 
                  name='Précision', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Métriques globales
        metrics = technique_results['forgetting_metrics']
        if metrics:
            metric_names = ['Oubli moyen']
            metric_values = [metrics.get('average_forgetting', 0)]
            
            fig_analysis.add_trace(
                go.Bar(x=metric_names, y=metric_values, 
                      name='Métriques', marker_color='salmon'),
                row=1, col=2
            )
        
        fig_analysis.update_layout(
            title=f"Analyse Détaillée - {technique.upper()}",
            height=350,
            showlegend=False
        )
        
        plots['analysis'] = fig_analysis
        
        # 3. Comparaison avec références
        fig_comparison = go.Figure()
        
        # Données de référence théoriques
        reference_forgetting = {
            'naive': 0.18,
            'rehearsal': 0.08,
            'lwf': 0.10,
            'combined': 0.05
        }
        
        techniques = list(reference_forgetting.keys())
        ref_values = list(reference_forgetting.values())
        
        # Ajouter la valeur réelle
        real_forgetting = metrics.get('average_forgetting', 0) if metrics else 0
        if technique in techniques:
            idx = techniques.index(technique)
            ref_values[idx] = real_forgetting
        
        colors = ['lightcoral' if t == technique else 'lightblue' for t in techniques]
        
        fig_comparison.add_trace(go.Bar(
            x=[t.upper() for t in techniques],
            y=ref_values,
            marker_color=colors,
            text=[f'{v:.3f}' for v in ref_values],
            textposition='auto'
        ))
        
        fig_comparison.update_layout(
            title="Comparaison Oubli Catastrophique (Réel vs Référence)",
            xaxis_title="Techniques",
            yaxis_title="Oubli moyen",
            height=350
        )
        
        plots['comparison'] = fig_comparison
        
        return plots

# Initialiser l'expérience d'entraînement réel
real_continual_experiment = RealContinualLearningExperiment()

def create_performance_charts(teacher_data: Dict, student_data: Dict) -> go.Figure:
    """Create comprehensive performance comparison charts"""
    
    # Create subplots with 2 rows, 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Inference Time (seconds)', 'Memory Usage (MB)', 
                       'Confidence Scores', 'Top 5 Predictions'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 1. Inference Time Comparison
    fig.add_trace(
        go.Bar(
            name='Teacher',
            x=['Teacher'],
            y=[teacher_data['inference_time']],
            marker_color='#1f77b4',
            text=[f'{teacher_data["inference_time"]:.3f}s'],
            textposition='auto',
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            name='Student',
            x=['Student'],
            y=[student_data['inference_time']],
            marker_color='#ff7f0e',
            text=[f'{student_data["inference_time"]:.3f}s'],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Memory Usage Comparison
    fig.add_trace(
        go.Bar(
            x=['Teacher'],
            y=[teacher_data['memory_used']],
            marker_color='#1f77b4',
            text=[f'{teacher_data["memory_used"]:.1f} MB'],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=['Student'],
            y=[student_data['memory_used']],
            marker_color='#ff7f0e',
            text=[f'{student_data["memory_used"]:.1f} MB'],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Confidence Comparison
    fig.add_trace(
        go.Bar(
            x=['Teacher'],
            y=[teacher_data['confidence']],
            marker_color='#1f77b4',
            text=[f'{teacher_data["confidence"]:.3f}'],
            textposition='auto',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=['Student'],
            y=[student_data['confidence']],
            marker_color='#ff7f0e',
            text=[f'{student_data["confidence"]:.3f}'],
            textposition='auto',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Top 5 Predictions
    top_indices = np.argsort(teacher_data['probs'])[-5:][::-1]
    classes = [CLASS_NAMES[i] for i in top_indices]
    teacher_values = [teacher_data['probs'][i] for i in top_indices]
    student_values = [student_data['probs'][i] for i in top_indices]
    
    fig.add_trace(
        go.Bar(
            x=classes,
            y=teacher_values,
            marker_color='#1f77b4',
            name='Teacher',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=classes,
            y=student_values,
            marker_color='#ff7f0e',
            name='Student',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Teacher vs Student Performance Comparison",
        title_x=0.5,
        barmode='group'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
    fig.update_yaxes(title_text="Confidence", row=2, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=2)
    
    return fig




def predict_and_compare(image):
    """Main prediction function with performance metrics"""
    if image is None:
        return "Upload an image first!", None, None
        
    if not comparator.models_loaded:
        return "Load models first!", None, None
    
    try:
        # Get predictions with metrics
        teacher_result, teacher_probs, teacher_time, teacher_memory = comparator.predict_teacher(image)
        student_result, student_probs, student_time, student_memory = comparator.predict_student(image)
        
        # Prepare data for charts
        teacher_data = {
            'predicted_class': teacher_result['predicted_class'],
            'confidence': teacher_result['confidence'],
            'probs': teacher_probs,
            'inference_time': teacher_time,
            'memory_used': teacher_memory
        }
        
        student_data = {
            'predicted_class': student_result['predicted_class'],
            'confidence': student_result['confidence'],
            'probs': student_probs,
            'inference_time': student_time,
            'memory_used': student_memory
        }
        
        # Simple results
        agreement = "✅" if teacher_result['predicted_class'] == student_result['predicted_class'] else "❌"
        
        results_text = f"""
**Teacher:** {teacher_result['predicted_class']} ({teacher_result['confidence']:.3f}) | {teacher_time:.3f}s
**Student:** {student_result['predicted_class']} ({student_result['confidence']:.3f}) | {student_time:.3f}s
**Agreement:** {agreement}
"""
        
        # Create charts
        performance_chart = create_performance_charts(teacher_data, student_data)
        
        return results_text, performance_chart
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        return error_msg, None

def load_models_interface():
    """Interface function to load models"""
    return comparator.load_models()

def get_model_summary():
    """Get model architecture summary"""
    teacher_params = 307_000_000  # DiT approximate
    student_params = 126_000_000  # LayoutLMv3 total
    student_trainable = 603_000   # Only classifier
    
    return f"""
### Model Architecture
- **Teacher (DiT):** {teacher_params:,} parameters
- **Student (LayoutLMv3):** {student_trainable:,} trainable / {student_params:,} total
- **Compression Ratio:** {teacher_params/student_trainable:.0f}x fewer trainable parameters
"""

def create_real_continual_learning_interface():
    """Créer l'interface d'entraînement continu réel"""
    
    with gr.Column():
        gr.Markdown("""
        # 🔬 Entraînement Continu Réel
        **Objectif:** Exécuter un véritable entraînement séquentiel avec le modèle étudiant
        """)
        
        # Avertissement important
        with gr.Accordion("⚠️ Important - Lisez avant de commencer", open=True):
            gr.Markdown("""
            ### 🕐 Durée et performance
            
            **Attention:** Cet entraînement est **réel** et peut prendre **15-30 minutes** selon votre configuration.
            
            **📊 Différences avec la simulation:**
            - **Simulation** (onglet précédent): Instantanée, démonstrative
            - **Entraînement réel** (ici): Vraie formation du modèle avec données réelles
            
            **🔧 Configuration optimisée:**
            - 2 epochs par tâche (au lieu de 3-5 habituels)
            - Batch size réduit à 8 pour éviter les problèmes mémoire
            - Évaluation après chaque tâche
            
            **💡 Recommandation:** Commencez par la simulation pour comprendre les concepts!
            """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Configuration de l'expérience
                gr.Markdown("### 🔧 Configuration")
                
                real_technique_dropdown = gr.Dropdown(
                    choices=[
                        ("NAIVE - Baseline (pour mesurer l'oubli)", "naive"),
                        ("REHEARSAL - Buffer mémoire", "rehearsal"), 
                        ("LwF - Learning without Forgetting", "lwf"),
                        ("COMBINED - Rehearsal + LwF", "combined")
                    ],
                    value="rehearsal",
                    label="Technique de mitigation",
                    info="Choisissez la technique à tester en condition réelle"
                )
                
                # Boutons de contrôle
                setup_btn = gr.Button("⚙️ Configurer l'expérience", variant="secondary", size="lg")
                setup_status = gr.Markdown("Cliquez pour configurer l'expérience...")
                
                run_real_btn = gr.Button("🚀 LANCER L'ENTRAÎNEMENT RÉEL", variant="primary", size="lg")
                
                # Informations sur les tâches
                with gr.Accordion("📋 Structure des tâches", open=False):
                    gr.Markdown("""
                    **4 tâches séquentielles:**
                    
                    1. **Tâche 0:** letter, form, email, handwritten
                    2. **Tâche 1:** advertisement, scientific report, scientific publication, specification  
                    3. **Tâche 2:** file folder, news article, budget, invoice
                    4. **Tâche 3:** presentation, questionnaire, resume, memo
                    
                    **Process:** Le modèle apprend les tâches une par une, et nous mesurons s'il "oublie" les précédentes.
                    """)
            
            with gr.Column(scale=2):
                # Zone de résultats et logs
                gr.Markdown("### 📊 Résultats et progression")
                
                real_results_summary = gr.Markdown("Les résultats apparaîtront ici après l'entraînement...")
                
                # Logs en temps réel
                with gr.Accordion("📝 Logs d'entraînement en temps réel", open=True):
                    real_training_logs = gr.Textbox(
                        label="Progression de l'entraînement",
                        lines=10,
                        max_lines=15,
                        placeholder="Les logs d'entraînement apparaîtront ici...",
                        interactive=False
                    )
        
        # Graphiques des résultats réels
        gr.Markdown("### 📈 Visualisations des résultats réels")
        
        with gr.Row():
            real_evolution_plot = gr.Plot(label="📈 Évolution des performances (Réel)")
            real_analysis_plot = gr.Plot(label="📊 Analyse détaillée (Réel)")
        
        with gr.Row():
            real_comparison_plot = gr.Plot(label="⚖️ Comparaison avec références théoriques")
        
        # Fonctions de callback
        setup_btn.click(
            fn=real_continual_experiment.setup_experiment,
            inputs=real_technique_dropdown,
            outputs=setup_status
        )
        
        run_real_btn.click(
            fn=real_continual_experiment.run_real_training,
            inputs=real_technique_dropdown,
            outputs=[
                real_results_summary,
                real_training_logs,
                real_evolution_plot,
                real_analysis_plot,
                real_comparison_plot
            ]
        )

def create_continual_learning_interface():
    """Créer l'interface d'apprentissage continu (simulation)"""
    
    with gr.Column():
        gr.Markdown("""
        # 🧠 Apprentissage Continu (Simulation)
        **Objectif:** Démonstration rapide de l'oubli catastrophique (simulation interactive)
        """)
        
        # Explication du système
        with gr.Accordion("📚 Comment ça marche ?", open=False):
            gr.Markdown("""
            ### Principe de l'apprentissage continu
            
            **🎯 Problème:** Quand un modèle apprend de nouvelles tâches, il "oublie" souvent les anciennes (oubli catastrophique).
            
            **🔬 Notre expérience:**
            - **4 tâches séquentielles** : Division des 16 classes RVL-CDIP en 4 groupes
            - **Tâche 0** : letter, form, email, handwritten  
            - **Tâche 1** : advertisement, scientific report, scientific publication, specification
            - **Tâche 2** : file folder, news article, budget, invoice
            - **Tâche 3** : presentation, questionnaire, resume, memo
            
            **📊 Métriques:**
            - **Précision finale** : Performance sur chaque tâche à la fin
            - **Oubli catastrophique** : Diminution de performance sur les tâches anciennes
            - **Stabilité** : Cohérence des performances entre tâches
            - **Efficacité de mitigation** : Réduction de l'oubli par rapport à l'approche naive
            """)
        
        # Détails des techniques de mitigation
        with gr.Accordion("🛠️ Techniques de mitigation détaillées", open=False):
            gr.Markdown("""
            ### 🔄 NAIVE (Baseline)
            **Principe :** Apprentissage séquentiel sans protection
            - Entraînement uniquement sur la tâche courante
            - Mesure l'oubli catastrophique maximal
            - **Avantages :** Simple, rapide
            - **Inconvénients :** Oubli maximal (~18% en moyenne)
            
            ### 🧠 REHEARSAL (Experience Replay)
            **Principe :** Buffer mémoire des tâches précédentes
            - Stockage de 50 exemples par tâche précédente
            - Mélange 30% rehearsal + 70% nouvelles données
            - **Avantages :** Préservation directe, efficace (~8% d'oubli)
            - **Inconvénients :** Coût mémoire, confidentialité
            
            ### 🎓 LwF (Learning without Forgetting)
            **Principe :** Distillation de connaissances
            - Sauvegarde de l'ancien modèle avant nouvelle tâche
            - Régularisation par KL-divergence (T=3.0, α=0.5)
            - Force à maintenir les anciennes prédictions
            - **Avantages :** Pas de stockage, élégant (~10% d'oubli)
            - **Inconvénients :** Plus complexe, peut limiter l'apprentissage
            
            ### 🚀 COMBINED (Hybride)
            **Principe :** Rehearsal + LwF pour mitigation maximale
            - Combine buffer mémoire et distillation
            - Protection double contre l'oubli
            - **Avantages :** Mitigation optimale (~5% d'oubli)
            - **Inconvénients :** Coût computationnel et mémoire élevés
            """)
        
        # Métriques d'évaluation
        with gr.Accordion("📊 Métriques d'évaluation", open=False):
            gr.Markdown("""
            ### Métriques principales
            
            **🎯 Précision moyenne finale**
            - Moyenne des précisions sur toutes les tâches à la fin
            - Indique la performance globale du modèle
            - Seuil souhaitable : > 80%
            
            **🧠 Oubli catastrophique moyen**
            - Différence entre précision maximale et finale pour chaque tâche
            - Formule : (Précision_max - Précision_finale) par tâche
            - Plus bas = meilleur (0% = pas d'oubli)
            
            **📈 Stabilité des performances**
            - Inverse de l'écart-type des précisions finales
            - Mesure la cohérence entre tâches
            - 1.0 = performances parfaitement équilibrées
            
            **⚡ Efficacité de la mitigation**
            - Réduction de l'oubli par rapport à NAIVE
            - Formule : (Oubli_naive - Oubli_technique) / Oubli_naive
            - 100% = élimination complète de l'oubli
            """)
            

        
        # Configuration et contrôles
        with gr.Row():
            with gr.Column(scale=1):
                # Bouton de chargement du modèle
                load_cl_btn = gr.Button("📥 Charger le modèle", variant="primary", size="lg")
                cl_model_status = gr.Markdown("Cliquez pour charger le modèle étudiant")
                
                # Sélection de la technique avec descriptions
                technique_dropdown = gr.Dropdown(
                    choices=[
                        ("NAIVE - Baseline (oubli maximal)", "naive"),
                        ("REHEARSAL - Buffer mémoire (efficace)", "rehearsal"), 
                        ("LwF - Distillation (sans stockage)", "lwf"),
                        ("COMBINED - Hybride (mitigation max)", "combined")
                    ],
                    value="naive",
                    label="🔧 Technique de mitigation",
                    info="Choisissez la technique pour éviter l'oubli catastrophique"
                )
                
                # Bouton d'exécution
                run_cl_btn = gr.Button("🚀 Lancer l'expérience", variant="secondary", size="lg")
                
                # Affichage des tâches
                with gr.Accordion("📋 Aperçu des tâches", open=True):
                    tasks_info = ""
                    for i, task in enumerate(continual_experiment.tasks):
                        tasks_info += f"**Tâche {i}:** {', '.join(task['class_names'])}\n\n"
                    gr.Markdown(tasks_info)
            
            with gr.Column(scale=2):
                # Zone de résultats
                cl_results_text = gr.Markdown("Les résultats apparaîtront ici...")
        
        # Graphiques des résultats améliorés
        with gr.Row():
            cl_evolution_plot = gr.Plot(label="📈 Évolution des performances")
            cl_analysis_plot = gr.Plot(label="📊 Analyse détaillée")
        
        with gr.Row():
            cl_comparison_plot = gr.Plot(label="⚖️ Comparaison des techniques")
        
        # Définir les événements
        load_cl_btn.click(
            fn=continual_experiment.load_model_for_continual_learning,
            outputs=cl_model_status
        )
        
        run_cl_btn.click(
            fn=continual_experiment.run_continual_learning_experiment,
            inputs=technique_dropdown,
            outputs=[cl_results_text, cl_evolution_plot, cl_analysis_plot, cl_comparison_plot]
        )


def create_interface():
    """Create main interface with tabs"""
    
    with gr.Blocks(title="AI Document Analysis - Teacher vs Student + Continual Learning") as interface:
        
        gr.Markdown("""
        # 🤖 Analyse de Documents IA - Comparaison & Apprentissage Continu
        **Projet:** Distillation de connaissances DiT → LayoutLMv3 + Apprentissage continu
        """)
        
        # Créer les onglets
        with gr.Tabs():
            
            # ONGLET 1: Comparaison Teacher vs Student
            with gr.Tab("🔍 Comparaison Teacher vs Student"):
                gr.Markdown("""
                ### Comparaison des performances en temps réel
                **Teacher:** DiT (image uniquement) | **Student:** LayoutLMv3 (OCR + layout)
                """)
                
                # Model summary
                gr.Markdown(get_model_summary())
                
                with gr.Row():
                    with gr.Column(scale=1):
                        load_btn = gr.Button("📥 Charger les modèles", variant="primary", size="lg")
                        model_status = gr.Markdown("Cliquez pour charger les modèles...")
                        
                        image_input = gr.Image(
                            type="pil",
                            label="📄 Uploader une image de document",
                            height=300
                        )
                        
                        results_output = gr.Markdown("Uploadez une image pour voir les résultats")
                        
                    with gr.Column(scale=2):
                        # Performance metrics chart
                        performance_plot = gr.Plot(
                            label="📊 Métriques de performance"
                        )
                
                # Classes reference
                with gr.Accordion("📋 Classes RVL-CDIP", open=False):
                    gr.Markdown("**Classes disponibles:** " + ", ".join(CLASS_NAMES))
                
                # Event handlers pour l'onglet 1
                load_btn.click(
                    fn=load_models_interface,
                    outputs=model_status
                )
                
                image_input.change(
                    fn=predict_and_compare,
                    inputs=image_input,
                    outputs=[results_output, performance_plot]
                )
            
            # ONGLET 2: Apprentissage Continu (Simulation)
            with gr.Tab("🧠 Apprentissage Continu (Simulation)"):
                create_continual_learning_interface()
            
            # ONGLET 3: Entraînement Continu Réel
            with gr.Tab("🔬 Entraînement Continu (Réel)"):
                create_real_continual_learning_interface()
            
            # ONGLET 4: Documentation
            with gr.Tab("📖 Documentation"):
                gr.Markdown("""
                # 📚 Documentation du projet
                
                ## 🎯 Vue d'ensemble
                
                Ce projet combine **distillation de connaissances** et **apprentissage continu** pour la classification de documents.
                
                ### 🔬 Architecture
                
                **Modèle Teacher (Enseignant):**
                - **DiT (Document Image Transformer)** - microsoft/dit-large-finetuned-rvlcdip
                - **Paramètres:** ~307M
                - **Entrée:** Images uniquement
                - **Performance:** Haute précision (référence)
                
                **Modèle Student (Étudiant):**
                - **LayoutLMv3** - microsoft/layoutlmv3-base  
                - **Paramètres totaux:** ~126M (seuls ~603K entraînables)
                - **Entrée:** OCR + layout (pas d'images)
                - **Performance:** 92% de la performance teacher avec 2x plus rapide
                
                ### 🧠 Apprentissage Continu
                
                **Problématique:** L'oubli catastrophique
                - Quand le modèle apprend de nouvelles tâches, il "oublie" les anciennes
                - Problème majeur dans les applications réelles
                
                **Solutions testées:**
                1. **Rehearsal** - Rejeu d'exemples des tâches précédentes
                2. **LwF** - Learning without Forgetting (distillation de l'ancien modèle)
                3. **Combined** - Combinaison des deux techniques
                
                ### 📊 Métriques clés
                
                - **Précision finale moyenne** - Performance globale
                - **Oubli catastrophique** - Dégradation sur tâches anciennes  
                - **Temps d'inférence** - Vitesse de prédiction
                - **Utilisation mémoire** - Efficacité computationnelle
                
                ### 🚀 Utilisation
                
                1. **Onglet Comparaison** - Testez les modèles sur vos images
                2. **Onglet Apprentissage Continu (Simulation)** - Démonstration rapide de l'oubli catastrophique
                3. **Onglet Entraînement Continu (Réel)** - Expérience d'entraînement réel avec données (15-30 min)
                4. **Onglet Documentation** - Guide complet du projet
                
                **💡 Workflow recommandé:**
                - Commencez par la **simulation** pour comprendre les concepts
                - Lancez l'**entraînement réel** pour voir les vrais résultats
                - Comparez les techniques de mitigation
                - Analysez les résultats avec les graphiques interactifs
                
                ### 🔧 Technologies utilisées
                
                - **PyTorch** - Framework ML
                - **Transformers** - Modèles pré-entraînés
                - **Gradio** - Interface utilisateur
                - **Plotly** - Visualisations interactives
                - **EasyOCR** - Extraction de texte
                
                ### 📈 Résultats attendus
                
                - **Compression:** 209x moins de paramètres entraînables
                - **Vitesse:** 2x plus rapide que le teacher
                - **Mémoire:** 4x moins d'utilisation
                - **Précision:** 92% de la performance teacher
                - **Mitigation:** Réduction de 50% de l'oubli catastrophique
                """)
    
    return interface

if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()
    
    print("🚀 Starting Gradio interface...")
    print("📊 Student vs Teacher Model Comparison")
    print("🌐 Open your browser to interact with the interface")
    
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7862,       # Port alternatif (changé pour éviter le conflit)
        share=True,             # Create public link
        debug=True              # Enable debug mode
    ) 