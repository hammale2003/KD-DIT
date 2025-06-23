#!/usr/bin/env python3
"""
Techniques de mitigation pour l'apprentissage continu
Implémente Rehearsal et Learning without Forgetting (LwF)
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, List, Tuple
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from datetime import datetime

from models import load_trained_student_model
from config import DEVICE


class RehearsalBuffer:
    """Buffer pour stocker les exemples des tâches précédentes (Experience Replay)"""
    
    def __init__(self, max_size_per_task: int = 50):
        self.max_size_per_task = max_size_per_task
        self.buffer = defaultdict(list)  # task_id -> list of examples
        
    def add_examples(self, task_id: int, examples: List[Dict]):
        """Ajouter des exemples au buffer pour une tâche donnée"""
        if task_id not in self.buffer:
            self.buffer[task_id] = []
        
        # Ajouter les nouveaux exemples
        self.buffer[task_id].extend(examples)
        
        # Limiter la taille du buffer par tâche
        if len(self.buffer[task_id]) > self.max_size_per_task:
            # Garder un échantillon aléatoire
            self.buffer[task_id] = random.sample(
                self.buffer[task_id], 
                self.max_size_per_task
            )
        
        print(f"Buffer tâche {task_id}: {len(self.buffer[task_id])} exemples")
    
    def sample_rehearsal_data(self, num_samples_per_task: int = 10) -> List[Dict]:
        """Échantillonner des données de rehearsal depuis toutes les tâches précédentes"""
        rehearsal_data = []
        
        for task_id, examples in self.buffer.items():
            if examples:
                # Échantillonner depuis cette tâche
                sample_size = min(num_samples_per_task, len(examples))
                sampled = random.sample(examples, sample_size)
                rehearsal_data.extend(sampled)
        
        print(f"Données de rehearsal: {len(rehearsal_data)} exemples au total")
        return rehearsal_data
    
    def get_all_previous_data(self) -> List[Dict]:
        """Obtenir toutes les données des tâches précédentes"""
        all_data = []
        for examples in self.buffer.values():
            all_data.extend(examples)
        return all_data


class LearningWithoutForgetting:
    """Implémentation de Learning without Forgetting (LwF)"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        self.temperature = temperature  # Température pour la distillation
        self.alpha = alpha  # Poids entre connaissance ancienne et nouvelle
        self.old_model = None
        
    def store_old_model(self, model):
        """Stocker l'état du modèle avant d'apprendre une nouvelle tâche"""
        self.old_model = copy.deepcopy(model)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad = False
        print("✅ Ancien modèle stocké pour LwF")
    
    def calculate_lwf_loss(self, student_logits: torch.Tensor, 
                          student_inputs: Dict, 
                          new_task_loss: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Calculer la perte LwF combinée"""
        if self.old_model is None:
            return new_task_loss, 0.0
        
        # Obtenir les prédictions de l'ancien modèle
        with torch.no_grad():
            old_outputs = self.old_model(**student_inputs)
            old_logits = old_outputs.logits
        
        # Distillation de connaissances de l'ancien modèle
        old_probs = F.softmax(old_logits / self.temperature, dim=-1)
        current_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Perte de distillation (KL divergence)
        distillation_loss = F.kl_div(current_log_probs, old_probs, reduction='batchmean')
        distillation_loss *= (self.temperature ** 2)
        
        # Combinaison des pertes
        total_loss = (1 - self.alpha) * new_task_loss + self.alpha * distillation_loss
        
        return total_loss, distillation_loss.item()


class ContinualLearningMitigations:
    """Système d'apprentissage continu avec techniques de mitigation"""
    
    def __init__(self):
        self.class_names = [
            "letter", "form", "email", "handwritten", "advertisement", 
            "scientific report", "scientific publication", "specification", 
            "file folder", "news article", "budget", "invoice", 
            "presentation", "questionnaire", "resume", "memo"
        ]
        
        # Charger le modèle
        print("Chargement du modèle étudiant...")
        self.student_model, self.student_processor = load_trained_student_model("student_model.pth")
        self.initial_state = copy.deepcopy(self.student_model.state_dict())
        
        # Techniques de mitigation
        self.rehearsal_buffer = RehearsalBuffer(max_size_per_task=50)
        self.lwf = LearningWithoutForgetting(temperature=3.0, alpha=0.5)
        
        # Tâches (4 tâches de 4 classes chacune)
        self.tasks = self._create_tasks()
        
        # Résultats
        self.results = {}
    
    def _create_tasks(self) -> List[Dict]:
        """Créer les définitions de tâches"""
        tasks = []
        for task_id in range(4):
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
    
    def simulate_task_data(self, task_id: int, num_samples: int = 30) -> List[Dict]:
        """Simuler des données d'entraînement pour une tâche"""
        task_classes = self.tasks[task_id]['classes']
        simulated_data = []
        
        for i in range(num_samples):
            # Simuler des features OCR aléatoires
            num_words = random.randint(5, 20)
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
    
    def train_with_technique(self, technique: str = "naive") -> Dict:
        """Entraîner avec une technique de mitigation spécifique"""
        print(f"\n{'='*60}")
        print(f"ENTRAÎNEMENT AVEC TECHNIQUE: {technique.upper()}")
        print(f"{'='*60}")
        
        # Réinitialiser
        self.student_model.load_state_dict(self.initial_state)
        self.rehearsal_buffer = RehearsalBuffer(max_size_per_task=50)
        self.lwf = LearningWithoutForgetting(temperature=3.0, alpha=0.5)
        
        task_results = defaultdict(list)
        
        # Entraîner séquentiellement sur chaque tâche
        for task_id in range(len(self.tasks)):
            print(f"\n--- Tâche {task_id}: {self.tasks[task_id]['description']} ---")
            
            # Préparer les données
            current_task_data = self.simulate_task_data(task_id, num_samples=30)
            
            # Technique-specific preparation
            if technique == "rehearsal" and task_id > 0:
                # Ajouter des données de rehearsal
                rehearsal_data = self.rehearsal_buffer.sample_rehearsal_data(10)
                training_data = current_task_data + rehearsal_data
                print(f"Entraînement avec {len(current_task_data)} nouveaux + {len(rehearsal_data)} rehearsal")
            
            elif technique == "lwf" and task_id > 0:
                # Stocker l'ancien modèle pour LwF
                self.lwf.store_old_model(self.student_model)
                training_data = current_task_data
                
            elif technique == "combined" and task_id > 0:
                # Combiner Rehearsal + LwF
                self.lwf.store_old_model(self.student_model)
                rehearsal_data = self.rehearsal_buffer.sample_rehearsal_data(10)
                training_data = current_task_data + rehearsal_data
                print(f"Entraînement combiné: {len(current_task_data)} nouveaux + {len(rehearsal_data)} rehearsal + LwF")
                
            else:
                training_data = current_task_data
            
            # Entraîner sur cette tâche
            self._train_on_task_data(training_data, technique, task_id)
            
            # Ajouter des exemples au buffer pour rehearsal futur
            if technique in ["rehearsal", "combined"]:
                # Prendre un échantillon représentatif de la tâche actuelle
                buffer_examples = random.sample(current_task_data, min(20, len(current_task_data)))
                self.rehearsal_buffer.add_examples(task_id, buffer_examples)
            
            # Évaluer sur toutes les tâches vues
            task_accuracies = self._evaluate_all_seen_tasks(task_id)
            
            # Stocker les résultats
            for eval_task_id, accuracy in task_accuracies.items():
                task_results[eval_task_id].append(accuracy)
        
        return dict(task_results)
    
    def _train_on_task_data(self, training_data: List[Dict], technique: str, task_id: int):
        """Entraîner le modèle sur les données d'une tâche avec technique spécifique"""
        self.student_model.train()
        
        # Optimiseur
        optimizer = optim.AdamW(self.student_model.parameters(), lr=2e-5, weight_decay=0.01)
        
        # Epochs d'entraînement
        num_epochs = 2
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_distillation_loss = 0.0
            num_batches = 0
            
            # Mélanger les données
            random.shuffle(training_data)
            
            for sample in training_data:
                try:
                    # Préparer les entrées
                    inputs = self.student_processor(
                        words=sample['words'],
                        boxes=sample['boxes'],
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=512
                    )
                    
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    label = torch.tensor([sample['label']]).to(DEVICE)
                    
                    # Forward pass
                    outputs = self.student_model(**inputs)
                    logits = outputs.logits
                    
                    # Perte de base
                    base_loss = F.cross_entropy(logits, label)
                    
                    # Appliquer la technique de mitigation
                    if technique == "lwf" and task_id > 0:
                        loss, distill_loss = self.lwf.calculate_lwf_loss(logits, inputs, base_loss)
                        total_distillation_loss += distill_loss
                    elif technique == "combined" and task_id > 0:
                        loss, distill_loss = self.lwf.calculate_lwf_loss(logits, inputs, base_loss)
                        total_distillation_loss += distill_loss
                    else:
                        loss = base_loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Erreur d'entraînement: {e}")
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_distill_loss = total_distillation_loss / num_batches if num_batches > 0 else 0
            
            print(f"Époque {epoch + 1}: Loss = {avg_loss:.4f}, Distill Loss = {avg_distill_loss:.4f}")
    
    def _evaluate_all_seen_tasks(self, max_task_id: int) -> Dict[int, float]:
        """Évaluer sur toutes les tâches vues jusqu'à présent"""
        self.student_model.eval()
        task_accuracies = {}
        
        with torch.no_grad():
            for task_id in range(max_task_id + 1):
                # Générer des données de test pour cette tâche
                test_data = self.simulate_task_data(task_id, num_samples=20)
                
                correct = 0
                total = 0
                
                for sample in test_data:
                    try:
                        inputs = self.student_processor(
                            words=sample['words'],
                            boxes=sample['boxes'],
                            return_tensors="pt",
                            truncation=True,
                            padding="max_length",
                            max_length=512
                        )
                        
                        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                        
                        outputs = self.student_model(**inputs)
                        logits = outputs.logits
                        predicted_class = torch.argmax(logits, dim=-1).item()
                        
                        if predicted_class == sample['label']:
                            correct += 1
                        total += 1
                        
                    except Exception as e:
                        continue
                
                accuracy = correct / total if total > 0 else 0.0
                task_accuracies[task_id] = accuracy
                print(f"  Tâche {task_id}: {accuracy:.3f}")
        
        return task_accuracies
    
    def compare_techniques(self) -> Dict:
        """Comparer toutes les techniques de mitigation"""
        techniques = ["naive", "rehearsal", "lwf", "combined"]
        all_results = {}
        
        for technique in techniques:
            print(f"\n{'#'*70}")
            print(f"TECHNIQUE: {technique.upper()}")
            print(f"{'#'*70}")
            
            results = self.train_with_technique(technique)
            all_results[technique] = results
            
            # Calculer les métriques de synthèse
            final_accuracies = []
            forgetting_scores = []
            
            for task_id, accuracies in results.items():
                if accuracies:
                    final_accuracies.append(accuracies[-1])
                    
                    # Oubli = max accuracy - final accuracy
                    if len(accuracies) > 1:
                        max_acc = max(accuracies[:-1])
                        forgetting = max_acc - accuracies[-1]
                        forgetting_scores.append(forgetting)
            
            avg_final_acc = np.mean(final_accuracies) if final_accuracies else 0
            avg_forgetting = np.mean(forgetting_scores) if forgetting_scores else 0
            
            print(f"RÉSULTATS {technique.upper()}:")
            print(f"  Précision finale moyenne: {avg_final_acc:.3f}")
            print(f"  Oubli catastrophique moyen: {avg_forgetting:.3f}")
        
        return all_results
    
    def plot_comparison(self, all_results: Dict):
        """Créer des graphiques de comparaison"""
        plt.figure(figsize=(16, 12))
        
        techniques = list(all_results.keys())
        colors = ['red', 'blue', 'green', 'orange']
        
        # Graphique 1: Évolution des précisions pour chaque technique
        for i, technique in enumerate(techniques):
            plt.subplot(2, 3, i + 1)
            results = all_results[technique]
            
            for task_id, accuracies in results.items():
                plt.plot(range(len(accuracies)), accuracies, 
                        marker='o', label=f'Tâche {task_id}')
            
            plt.title(f'{technique.upper()} - Évolution par tâche')
            plt.xlabel('Points d\'évaluation')
            plt.ylabel('Précision')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Graphique 5: Comparaison des précisions finales
        plt.subplot(2, 3, 5)
        final_accs = []
        technique_names = []
        
        for technique in techniques:
            results = all_results[technique]
            accuracies = []
            for task_id, task_accs in results.items():
                if task_accs:
                    accuracies.append(task_accs[-1])
            
            avg_acc = np.mean(accuracies) if accuracies else 0
            final_accs.append(avg_acc)
            technique_names.append(technique.upper())
        
        bars = plt.bar(technique_names, final_accs, color=colors, alpha=0.7)
        plt.title('Précision finale moyenne par technique')
        plt.ylabel('Précision')
        plt.xticks(rotation=45)
        
        # Ajouter les valeurs
        for bar, acc in zip(bars, final_accs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Graphique 6: Comparaison de l'oubli catastrophique
        plt.subplot(2, 3, 6)
        forgetting_scores = []
        
        for technique in techniques:
            results = all_results[technique]
            all_forgetting = []
            
            for task_id, task_accs in results.items():
                if len(task_accs) > 1:
                    max_acc = max(task_accs[:-1])
                    forgetting = max_acc - task_accs[-1]
                    all_forgetting.append(forgetting)
            
            avg_forgetting = np.mean(all_forgetting) if all_forgetting else 0
            forgetting_scores.append(avg_forgetting)
        
        bars = plt.bar(technique_names, forgetting_scores, color=colors, alpha=0.7)
        plt.title('Oubli catastrophique moyen')
        plt.ylabel('Oubli (diminution de précision)')
        plt.xticks(rotation=45)
        
        # Ajouter les valeurs
        for bar, forgetting in zip(bars, forgetting_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{forgetting:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('continual_learning_mitigation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Résumé des résultats
        print(f"\n{'='*70}")
        print("RÉSUMÉ DES COMPARAISONS")
        print(f"{'='*70}")
        
        for i, technique in enumerate(techniques):
            print(f"{technique.upper():12} | Précision: {final_accs[i]:.3f} | Oubli: {forgetting_scores[i]:.3f}")


def main():
    """Fonction principale pour tester les techniques de mitigation"""
    print("🔬 Test des Techniques de Mitigation pour l'Apprentissage Continu")
    print("=" * 70)
    
    print("Techniques disponibles:")
    print("  1. NAIVE - Pas de mitigation (baseline)")
    print("  2. REHEARSAL - Rejeu d'exemples des tâches précédentes")
    print("  3. LWF - Learning without Forgetting")
    print("  4. COMBINED - Rehearsal + LwF")
    
    print("\n💡 Concepts clés:")
    print("• REHEARSAL: Stocke des exemples des tâches précédentes et les rejout lors de l'entraînement")
    print("• LwF: Utilise la distillation pour préserver les connaissances de l'ancien modèle")
    print("• COMBINED: Combine les deux techniques pour une mitigation maximale")
    
    print("\n📊 Métriques évaluées:")
    print("• Précision finale moyenne sur toutes les tâches")
    print("• Oubli catastrophique (diminution de performance sur les tâches anciennes)")
    print("• Évolution temporelle des performances par tâche")
    
    print("\n🚀 Pour exécuter les expériences, utilisez:")
    print("  python continual_learning_mitigations.py")


if __name__ == "__main__":
    main() 