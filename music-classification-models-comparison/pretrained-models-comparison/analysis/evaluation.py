import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Dict, Tuple, List
import numpy as np


def evaluate_model(
    model: torch.nn.Module, 
    test_loader, 
    criterion: torch.nn.Module, 
    idx_to_genre: Dict[int, str],
    device: torch.device
) -> Tuple[float, float, Dict, np.ndarray, List[int], List[int]]:
    """
    Evaluate model on test set and return comprehensive metrics
    
    Args:
        model: Trained model to evaluate  
        test_loader: Test data loader
        criterion: Loss function
        idx_to_genre: Mapping from class index to genre name
        device: Device to run evaluation on
        
    Returns:
        tuple: (test_loss, accuracy, classification_report, confusion_matrix, predictions, true_labels)
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Create class names in correct order
    class_names = [idx_to_genre[i] for i in range(len(idx_to_genre))]
    
    # Classification report
    classification_rep = classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return test_loss, accuracy, classification_rep, cm, all_preds, all_labels


def get_model_predictions_with_confidence(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    return_probabilities: bool = True
) -> Tuple[List[int], List[float]]:
    """
    Get model predictions with confidence scores
    
    Args:
        model: Trained model
        test_loader: Test data loader  
        device: Device to run inference on
        return_probabilities: Whether to return softmax probabilities
        
    Returns:
        tuple: (predictions, confidence_scores)
    """
    model.eval()
    all_preds = []
    all_confidences = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            if return_probabilities:
                probs = torch.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probs, 1)
            else:
                confidences, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    return all_preds, all_confidences