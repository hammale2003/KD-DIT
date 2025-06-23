#!/usr/bin/env python3
"""
Enhanced Knowledge Distillation Launcher
========================================

Unified script with advanced optimizations:
- Advanced loss functions (label smoothing, focal loss)  
- Learning rate warmup and cosine annealing
- Gradient clipping and weight decay
- Better hyperparameters for >85% accuracy

Usage: 
    python run_distillation.py --mode train --epochs 10 --alpha 0.8
    python run_distillation.py --mode eval
    python run_distillation.py --mode info
"""

import argparse
import os
import sys
import time
import torch

def main():
    parser = argparse.ArgumentParser(description='Enhanced Knowledge Distillation Launcher')
    
    # Mode selection
    parser.add_argument('--mode', choices=['info', 'test', 'train', 'eval'], 
                       default='train', help='What to run')
    
    # ENHANCED Training parameters with better defaults
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (optimized: 2e-4)')
    parser.add_argument('--alpha', type=float, default=0.8, help='Distillation weight (optimized: 0.8)')
    parser.add_argument('--temperature', type=float, default=3.0, help='Temperature (optimized: 3.0)')
    
    # Advanced options
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--focal-loss', action='store_true', help='Use focal loss instead of label smoothing')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    
    # File paths
    parser.add_argument('--checkpoint', type=str, default='latest_checkpoint.pth',
                       help='Checkpoint path')
    parser.add_argument('--model', type=str, default='student_model.pth',
                       help='Best model path')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    if args.mode == 'info':
        print("📊 Getting model and dataset information...")
        try:
            from models import load_teacher_model, load_student_model
            
            # Load models
            print("Loading models...")
            teacher_model, teacher_processor = load_teacher_model()
            student_model, student_processor = load_student_model()
            
            # Count parameters
            teacher_total = sum(p.numel() for p in teacher_model.parameters())
            teacher_trainable = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
            student_total = sum(p.numel() for p in student_model.parameters())
            student_trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
            
            print(f"\n📊 TEACHER MODEL (DiT):")
            print(f"   Total parameters: {teacher_total:,}")
            print(f"   Trainable parameters: {teacher_trainable:,}")
            print(f"   Model size: ~{teacher_total * 4 / 1024**2:.1f} MB")
            
            print(f"\n📊 STUDENT MODEL (LayoutLMv3 Frozen):")
            print(f"   Total parameters: {student_total:,}")
            print(f"   Trainable parameters: {student_trainable:,} ({100*student_trainable/student_total:.1f}%)")
            print(f"   Model size: ~{student_total * 4 / 1024**2:.1f} MB")
            
            print(f"\n📊 COMPRESSION ANALYSIS:")
            print(f"   Teacher/Student size ratio: {teacher_total/student_total:.1f}x")
            print(f"   Teacher/Student trainable ratio: {teacher_total/student_trainable:.1f}x")
            print(f"   Memory savings: {(1 - student_total/teacher_total)*100:.1f}%")
            
            # Dataset info
            from dataset import get_dataset_info
            print(f"\n📊 DATASET INFO:")
            get_dataset_info()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    elif args.mode == 'test':
        print("🧪 Running quick compatibility test...")
        try:
            from models import load_teacher_model, load_student_model
            from dataset import load_data
            
            # Test model loading
            print("Loading models...")
            teacher_model, teacher_processor = load_teacher_model()
            student_model, student_processor = load_student_model()
            print("✅ Models loaded successfully")
            
            # Test data loading
            print("Testing data loading...")
            train_loader, val_loader = load_data(
                teacher_processor, student_processor
            )
            print("✅ Data loaders created successfully")
            
            # Test single batch
            print("Testing single batch processing...")
            batch = next(iter(val_loader))
            
            # Test teacher
            teacher_inputs = {k: v.to('cuda') for k, v in batch['teacher_inputs'].items()}
            with torch.no_grad():
                teacher_outputs = teacher_model(**teacher_inputs)
            print(f"✅ Teacher forward pass: {teacher_outputs.logits.shape}")
            
            # Test student
            student_inputs = {k: v.to('cuda') for k, v in batch['student_inputs'].items()}
            with torch.no_grad():
                student_outputs = student_model(**student_inputs)
            print(f"✅ Student forward pass: {student_outputs.logits.shape}")
            
            print("🎉 All tests passed! Ready for training.")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.mode == 'train':
        print("🚀 Starting ENHANCED knowledge distillation training...")
        print("=" * 55)
        
        # Show enhanced configuration
        print(f"⚙️  ENHANCED CONFIGURATION:")
        print(f"   Epochs: {args.epochs} (increased for better convergence)")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Learning rate: {args.lr} (optimized)")
        print(f"   Alpha (KD weight): {args.alpha} (optimized)")
        print(f"   Temperature: {args.temperature} (optimized)")
        print(f"   Label smoothing: {args.label_smoothing}")
        print(f"   Focal loss: {args.focal_loss}")
        print(f"   Weight decay: {args.weight_decay}")
        print(f"   Resume training: {args.resume}")
        print()
        
        try:
            # Update config with enhanced parameters
            import config
            config.NUM_EPOCHS = args.epochs
            config.BATCH_SIZE = args.batch_size
            config.LEARNING_RATE = args.lr
            config.ALPHA = args.alpha
            config.TEMPERATURE = args.temperature
            config.LABEL_SMOOTHING = args.label_smoothing
            config.WEIGHT_DECAY = args.weight_decay
            
            # Load everything
            from models import load_teacher_model, load_student_model
            from dataset import load_data
            from train import train_knowledge_distillation, evaluate
            from utils import set_random_seed, log_system_info
            
            # Set seed for reproducibility
            set_random_seed(42)
            log_system_info()
            
            # Load models
            print("📦 Loading models...")
            teacher_model, teacher_processor = load_teacher_model()
            student_model, student_processor = load_student_model()
            
            # Show model info
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            student_trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
            print(f"📊 Teacher: {teacher_params:,} parameters")
            print(f"📊 Student: {student_trainable:,} trainable parameters")
            print(f"📊 Compression ratio: {teacher_params/student_trainable:.1f}x")
            
            # Load data
            print("📂 Loading data...")
            train_loader, val_loader = load_data(
                teacher_processor, student_processor
            )
            print(f"✅ Dataset: {len(train_loader)} train batches, {len(val_loader)} val batches")
            
            # Evaluate teacher baseline
            print("\n📏 Teacher baseline:")
            teacher_acc = evaluate(teacher_model, val_loader, model_type="teacher")
            print(f"Teacher accuracy: {teacher_acc:.4f}")
            
            # Evaluate student before training
            print("📏 Student baseline (before training):")
            student_acc = evaluate(student_model, val_loader, model_type="student")
            print(f"Student accuracy (untrained): {student_acc:.4f}")
            
            # Start training with timing
            print(f"\n🎯 Starting enhanced training...")
            start_time = time.time()
            
            # Enhanced training
            train_knowledge_distillation(
                teacher_model=teacher_model,
                student_model=student_model,
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_path=args.checkpoint if args.resume else None,
                best_model_path=args.model,
                learning_rate=args.lr
            )
            
            # Training completed
            training_time = time.time() - start_time
            print(f"\n✅ Training completed in {training_time/3600:.2f} hours")
            
            # Final evaluation
            print("\n🏆 FINAL RESULTS:")
            print("-" * 30)
            student_model.load_state_dict(torch.load(args.model, map_location='cuda'))
            final_acc = evaluate(student_model, val_loader, model_type="student")
            
            print(f"Teacher accuracy: {teacher_acc:.4f}")
            print(f"Student accuracy (final): {final_acc:.4f}")
            print(f"Improvement: +{final_acc - student_acc:.4f}")
            print(f"Knowledge transfer: {final_acc/teacher_acc:.1%}")
            
            # Save results summary
            results = {
                'teacher_acc': teacher_acc,
                'student_initial_acc': student_acc,
                'student_final_acc': final_acc,
                'improvement': final_acc - student_acc,
                'training_hours': training_time / 3600,
                'config': vars(args)
            }
            
            import json
            with open('training_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Results saved to 'training_results.json'")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.mode == 'eval':
        print("📈 Running enhanced evaluation...")
        
        try:
            from models import load_teacher_model, load_student_model
            from dataset import load_data
            from train import evaluate
            
            # Load models
            print("Loading models...")
            teacher_model, teacher_processor = load_teacher_model()
            student_model, student_processor = load_student_model()
            
            # Load best model if exists
            if os.path.exists(args.model):
                print(f"Loading trained model from {args.model}...")
                student_model.load_state_dict(torch.load(args.model, map_location='cuda'))
            else:
                print(f"⚠️  No trained model found at {args.model}, using initialized model")
            
            # Load data
            print("Loading validation data...")
            _, val_loader = load_data(teacher_processor, student_processor)
            
            # Evaluate both models
            print("\n📊 EVALUATION RESULTS:")
            print("-" * 25)
            
            teacher_acc = evaluate(teacher_model, val_loader, model_type="teacher")
            print(f"Teacher accuracy: {teacher_acc:.4f}")
            
            student_acc = evaluate(student_model, val_loader, model_type="student")
            print(f"Student accuracy: {student_acc:.4f}")
            
            print(f"Knowledge transfer: {student_acc/teacher_acc:.1%}")
            print(f"Gap to teacher: {teacher_acc - student_acc:.4f}")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main() 