#!/usr/bin/env python3
"""
Knowledge Distillation for Document Classification
Teacher: DiT (Document Image Transformer) 
Student: LayoutLMv3-Tiny
Dataset: RVL-CDIP with enhanced features (OCR, bounding boxes)
"""

import os
import argparse
from models import load_teacher_model, load_student_model, get_model_info
from dataset import load_data, get_dataset_info
from train import train_knowledge_distillation, evaluate
from config import (
    CHECKPOINT_PATH, BEST_MODEL_PATH, LEARNING_RATE, 
    BATCH_SIZE, NUM_EPOCHS, ALPHA, TEMPERATURE
)


def main():
    """Main function to run knowledge distillation training"""
    
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')
    parser.add_argument('--mode', choices=['train', 'eval', 'info'], default='train',
                       help='Mode: train, eval, or info')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                       help='Checkpoint path')
    parser.add_argument('--model', type=str, default=BEST_MODEL_PATH,
                       help='Model path for evaluation')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--alpha', type=float, default=ALPHA,
                       help='Distillation weight')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                       help='Distillation temperature')
    
    args = parser.parse_args()
    
    if args.mode == 'info':
        print("=== Informations sur le Dataset ===")
        get_dataset_info()
        return
    
    print("=== Démarrage du processus de distillation de connaissances ===")
    print(f"Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Alpha (distillation weight): {args.alpha}")
    print(f"  Temperature: {args.temperature}")
    
    # Load models
    print("\n=== Chargement des modèles ===")
    teacher_model, teacher_processor = load_teacher_model()
    student_model, student_processor = load_student_model()
    
    # Print model information
    get_model_info(teacher_model, "Teacher (DiT)")
    get_model_info(student_model, "Student (LayoutLMv3-Tiny)")
    
    # Load data
    print("\n=== Chargement du dataset ===")
    train_loader, val_loader = load_data(teacher_processor, student_processor)
    
    if args.mode == 'train':
        print("\n=== Entraînement ===")
        train_knowledge_distillation(
            teacher_model=teacher_model,
            student_model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_path=args.checkpoint,
            best_model_path=args.model,
            learning_rate=args.lr
        )
        
    elif args.mode == 'eval':
        print("\n=== Évaluation ===")
        
        # Load best model if exists
        if os.path.exists(args.model):
            print(f"Chargement du modèle depuis {args.model}...")
            student_model.load_state_dict(torch.load(args.model, map_location='cuda'))
            print("Modèle chargé.")
        else:
            print(f"Aucun modèle trouvé à {args.model}. Utilisation du modèle initialisé.")
        
        # Evaluate on validation set
        val_accuracy = evaluate(student_model, val_loader)
        print(f"\nPrécision sur l'ensemble de validation: {val_accuracy:.4f}")
        
        # Optionally evaluate on train set for comparison
        train_accuracy = evaluate(student_model, train_loader)
        print(f"Précision sur l'ensemble d'entraînement: {train_accuracy:.4f}")


def test_single_batch():
    """Test function to verify everything works with a single batch"""
    print("=== Test avec un seul batch ===")
    
    # Load models
    teacher_model, teacher_processor = load_teacher_model()
    student_model, student_processor = load_student_model()
    
    # Load data
    train_loader, val_loader = load_data(teacher_processor, student_processor)
    
    # Test single batch from train loader
    print("Test du premier batch d'entraînement...")
    try:
        batch = next(iter(train_loader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Teacher inputs keys: {batch['teacher_inputs'].keys()}")
        print(f"Student inputs keys: {batch['student_inputs'].keys()}")
        print(f"Batch size: {batch['label'].size(0)}")
        print(f"Labels: {batch['label']}")
        
        # Test forward pass
        teacher_inputs = {k: v.cuda() for k, v in batch['teacher_inputs'].items()}
        student_inputs = {k: v.cuda() for k, v in batch['student_inputs'].items()}
        
        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs)
            student_outputs = student_model(**student_inputs)
            
        print(f"Teacher logits shape: {teacher_outputs.logits.shape}")
        print(f"Student logits shape: {student_outputs.logits.shape}")
        print("Test réussi !")
        
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import torch
    
    # Uncomment to run test first
    # test_single_batch()
    
    main() 