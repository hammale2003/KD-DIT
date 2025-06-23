import torch
import torch.optim as optim
from tqdm import tqdm
import warnings
from losses import (
    calculate_distillation_loss, 
    calculate_advanced_distillation_loss,
    calculate_accuracy, 
    DistillationLossTracker
)
from config import (
    DEVICE, ALPHA, TEMPERATURE, NUM_EPOCHS, 
    WARMUP_EPOCHS, WEIGHT_DECAY, GRADIENT_CLIP_VALUE, LABEL_SMOOTHING
)


def train_epoch(teacher_model, student_model, train_loader, optimizer, scheduler, alpha, temperature, epoch):
    """
    Train student model for one epoch using knowledge distillation
    """
    student_model.train()
    teacher_model.eval()  # Teacher always in eval mode
    
    loss_tracker = DistillationLossTracker()
    
    progress_bar = tqdm(
        train_loader, 
        desc=f"Époque {epoch+1}/{NUM_EPOCHS} [Train]", 
        leave=True
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move inputs to device
            teacher_inputs = {k: v.to(DEVICE) for k, v in batch['teacher_inputs'].items()}
            student_inputs = {k: v.to(DEVICE) for k, v in batch['student_inputs'].items()}
            labels = batch['label'].to(DEVICE)
            
            # Forward pass - Teacher (no gradients)
            with torch.no_grad():
                teacher_outputs = teacher_model(**teacher_inputs)
                teacher_logits = teacher_outputs.logits
            
            # Forward pass - Student
            student_outputs = student_model(**student_inputs)
            student_logits = student_outputs.logits
            
            # Calculate losses with advanced techniques
            loss, ce_loss, kd_loss = calculate_advanced_distillation_loss(
                student_logits, teacher_logits, labels, alpha, temperature,
                label_smoothing=LABEL_SMOOTHING, use_focal=False
            )
            
            # Calculate accuracy
            accuracy, correct, batch_size = calculate_accuracy(student_logits, labels)
            
            # Backward pass and optimization with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), GRADIENT_CLIP_VALUE)
            
            optimizer.step()
            
            # Update learning rate scheduler (per step)
            scheduler.step()
            
            # Update loss tracker
            loss_tracker.update(
                loss.item(), ce_loss.item(), kd_loss.item(), 
                correct, batch_size
            )
            
            # Update progress bar
            avg_loss, avg_ce_loss, avg_kd_loss, avg_accuracy = loss_tracker.get_averages()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{ce_loss.item():.4f}",
                'kd': f"{kd_loss.item():.4f}",
                'acc': f"{avg_accuracy:.4f}"
            }, refresh=True)
            
        except Exception as e:
            print(f"\n!!!!!! ERREUR DANS train_epoch (Epoch {epoch+1}, Batch {batch_idx}) !!!!!!!!")
            print(f"Exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    # Return epoch averages
    return loss_tracker.get_averages()


def evaluate(model, val_loader, epoch=None, model_type="student"):
    """
    Evaluate model on validation set
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    desc = f"Évaluation {model_type}" if epoch is None else f"Évaluation {model_type} Époque {epoch+1}"
    progress_bar = tqdm(val_loader, desc=desc)
    
    with torch.no_grad():
        for batch in progress_bar:
            try:
                # Use appropriate inputs based on model type
                if model_type == "teacher":
                    model_inputs = {k: v.to(DEVICE) for k, v in batch['teacher_inputs'].items()}
                else:  # student
                    model_inputs = {k: v.to(DEVICE) for k, v in batch['student_inputs'].items()}
                    
                labels = batch['label'].to(DEVICE)
                
                # Forward pass
                outputs = model(**model_inputs)
                logits = outputs.logits
                
                # Calculate accuracy
                _, correct, batch_size = calculate_accuracy(logits, labels)
                total_correct += correct
                total_samples += batch_size
                
                # Update progress bar
                current_acc = total_correct / total_samples
                progress_bar.set_postfix({'acc': f"{current_acc:.4f}"})
                
            except Exception as e:
                print(f"\n!!!!!! ERREUR DANS evaluate ({model_type}) !!!!!!!!")
                print(f"Exception: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise e
    
    accuracy = total_correct / total_samples
    return accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, checkpoint_path):
    """
    Save training checkpoint
    """
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'config': {
            'ALPHA': ALPHA,
            'TEMPERATURE': TEMPERATURE,
            'NUM_EPOCHS': NUM_EPOCHS
        }
    }
    torch.save(checkpoint_data, checkpoint_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load training checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']
    
    return start_epoch, best_val_acc


def train_knowledge_distillation(teacher_model, student_model, train_loader, val_loader, 
                                checkpoint_path, best_model_path, learning_rate=2e-4):
    """
    Enhanced training function for knowledge distillation with advanced optimizations
    """
    # Filter warning
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")
    
    # Setup optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        student_model.parameters(), 
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Enhanced scheduler with warmup
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    
    # Linear warmup + Cosine annealing
    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize training state
    start_epoch = 0
    best_val_acc = 0.0
    
    # Load checkpoint if exists
    try:
        if checkpoint_path and torch.load(checkpoint_path, map_location='cpu'):
            print(f"Chargement du checkpoint depuis '{checkpoint_path}'...")
            start_epoch, best_val_acc = load_checkpoint(
                student_model, optimizer, scheduler, checkpoint_path
            )
            print(f"Checkpoint chargé. Reprise à l'époque {start_epoch}. "
                  f"Meilleure validation précédente: {best_val_acc:.4f}")
    except:
        print("Aucun checkpoint trouvé. Démarrage de l'entraînement depuis le début.")
    
    # Training loop
    print(f"Début de l'entraînement de l'époque {start_epoch} à {NUM_EPOCHS-1}")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n--- Époque {epoch+1}/{NUM_EPOCHS} ---")
        
        # Training
        train_loss, train_ce_loss, train_kd_loss, train_acc = train_epoch(
            teacher_model, student_model, train_loader, optimizer, scheduler,
            ALPHA, TEMPERATURE, epoch
        )
        
        # Evaluation
        val_acc = evaluate(student_model, val_loader, epoch, "student")
        
        # Update learning rate (scheduler is now called per step in train_epoch)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"Fin Époque {epoch+1}/{NUM_EPOCHS}:")
        print(f"  LR Actuel: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f}, CE Loss: {train_ce_loss:.4f}, "
              f"KD Loss: {train_kd_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f} (Meilleure: {best_val_acc:.4f})")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  Nouvelle meilleure précision ! "
                  f"Sauvegarde du meilleur état du modèle dans '{best_model_path}'...")
            torch.save(student_model.state_dict(), best_model_path)
        
        # Save checkpoint
        if checkpoint_path:
            print(f"Sauvegarde du checkpoint à la fin de l'époque {epoch+1} "
                  f"dans '{checkpoint_path}'...")
            save_checkpoint(
                student_model, optimizer, scheduler, epoch, 
                best_val_acc, checkpoint_path
            )
            print("Checkpoint sauvegardé.")
    
    print(f"\nEntraînement terminé après {NUM_EPOCHS} époques.")
    print(f"Meilleure précision sur la validation obtenue: {best_val_acc:.4f}")
    print(f"Le dernier checkpoint a été sauvegardé dans '{checkpoint_path}'")
    print(f"L'état du meilleur modèle a été sauvegardé dans '{best_model_path}'") 
