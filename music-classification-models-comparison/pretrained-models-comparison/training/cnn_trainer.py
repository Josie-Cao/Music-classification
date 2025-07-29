import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, List, Tuple
import copy


class CNNTrainer:
    """
    Trainer class for CNN baseline model
    
    Implements standard training strategy for CNN from scratch without transfer learning.
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
        model: nn.Module, 
        train_loader, 
        val_loader, 
        criterion: nn.Module, 
        fold_num: int = 1
    ) -> Dict[str, Any]:
        """
        Train CNN baseline model for a single fold
        
        Args:
            model: CNN model to train
            train_loader: Training data loader
            val_loader: Validation data loader  
            criterion: Loss function
            fold_num: Current fold number for logging
            
        Returns:
            dict: Training results and metrics
        """
        print(f"Training CNN Baseline - Fold {fold_num}")
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience, 
            min_lr=self.config.min_lr, 
            verbose=False
        )
        
        # Training tracking variables
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_model_weights = None
        patience_counter = 0
        
        # History tracking
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'learning_rates': []
        }
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion)
            history['train_losses'].append(train_loss)
            history['train_accs'].append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            history['val_losses'].append(val_loss)
            history['val_accs'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Progress logging
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1}/{self.config.num_epochs}: '
                      f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, '
                      f'LR: {current_lr:.2e}')
            
            # Early stopping and best model tracking
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                print(f'  Early stopping triggered after {epoch+1} epochs')
                break
        
        # Load best model weights
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
        
        # Return training results
        return {
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'final_epoch': epoch + 1,
            'stopped_early': patience_counter >= self.config.patience,
            'history': history
        }
    
    def _train_epoch(
        self, 
        model: nn.Module, 
        train_loader, 
        optimizer: optim.Optimizer, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
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
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(
        self, 
        model: nn.Module, 
        val_loader, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
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
        
        epoch_loss = val_loss / len(val_loader.dataset)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc