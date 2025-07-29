import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Any, List, Tuple
import copy


class ViTTrainer:
    """
    Trainer class for Vision Transformer with 2-stage fine-tuning
    """
    
    def __init__(self, config, device):
        """
        Args:
            config: TrainingConfig instance
            device: torch.device for training
        """
        self.config = config
        self.device = device
        
    def train_single_fold(
        self, 
        model, 
        train_loader, 
        val_loader, 
        criterion: nn.Module, 
        fold_num: int = 1
    ) -> Dict[str, Any]:
        """
        Train ViT model with 2-stage fine-tuning
        
        Args:
            model: ViT model to train
            train_loader: Training data loader
            val_loader: Validation data loader  
            criterion: Loss function
            fold_num: Current fold number for logging
            
        Returns:
            dict: Training results and metrics
        """
        print(f"Training Vision Transformer (ViT) - Fold {fold_num}")
        
        best_val_acc = 0.0
        best_model_weights = None
        all_history = {}
        
        # Get stage configurations
        stage_configs = self.config.stage_configs or self._get_default_stage_configs()
        
        # Train each stage
        for stage_name, stage_config in stage_configs.items():
            print(f"\n  {stage_config['name']}")
            
            # Configure model for this stage
            if stage_config['freeze_backbone']:
                model.freeze_backbone()
            else:
                model.unfreeze_all()
            
            # Setup optimizer for this stage
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=stage_config['learning_rate'],
                weight_decay=self.config.weight_decay
            )
            
            # Use cosine annealing for ViT
            if self.config.scheduler_type == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=stage_config['epochs'])
            else:
                scheduler = ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6, verbose=False
                )
            
            print(f"    Trainable parameters: {model.get_trainable_params():,}")
            
            # Train this stage
            stage_history = self._train_stage(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                stage_config['epochs'], stage_config['name']
            )
            
            all_history[stage_name] = stage_history
            
            # Update best model if this stage improved
            if stage_history['best_val_acc'] > best_val_acc:
                best_val_acc = stage_history['best_val_acc']
                best_model_weights = copy.deepcopy(model.state_dict())
        
        # Load best model weights
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
        
        return {
            'best_val_acc': best_val_acc,
            'stage_history': all_history,
            'training_strategy': '2-stage ViT fine-tuning'
        }
    
    def _train_stage(
        self, 
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        epochs: int, 
        stage_name: str
    ) -> Dict[str, Any]:
        """Train a single stage"""
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []
        }
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion)
            history['train_losses'].append(train_loss)
            history['train_accs'].append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            history['val_losses'].append(val_loss)
            history['val_accs'].append(val_acc)
            
            # Learning rate scheduling
            if isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
            else:
                scheduler.step(val_loss)
            
            # Progress logging
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"    {stage_name} Epoch {epoch+1}/{epochs}: "
                      f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Best model tracking
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping for this stage
            if patience_counter >= 8:  # ViT-specific patience
                print(f"    Early stopping for {stage_name} after {epoch+1} epochs")
                break
        
        return {
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'final_epoch': epoch + 1,
            'history': history
        }
    
    def _train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
           
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for ViT stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        return running_loss / len(train_loader.dataset), 100 * correct / total
    
    def _validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return val_loss / len(val_loader.dataset), 100 * correct / total
    
    def _get_default_stage_configs(self):
        """Get default 2-stage configuration if not provided"""
        return {
            'stage1': {
                'name': 'Stage 1: Classifier Head Only',
                'epochs': 30,
                'learning_rate': 0.001,
                'freeze_backbone': True
            },
            'stage2': {
                'name': 'Stage 2: Full Fine-tuning',
                'epochs': 70,
                'learning_rate': 0.0001,
                'freeze_backbone': False
            }
        }