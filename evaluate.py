import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.data_loader import get_data_loaders
from models.model_cnn import CustomCNN
from models.model_resnet import ResNet18Transfer

def load_model(model_path, model_type, num_classes, device):
    """Loads a saved model from a .pth file."""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return None
        
    if model_type == 'cnn':
        model = CustomCNN(num_classes=num_classes)
    elif model_type == 'resnet':
        model = ResNet18Transfer(num_classes=num_classes, freeze_backbone=False)
    else:
        raise ValueError(f"Unknown model type {model_type}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate(model, loader, device, class_names):
    """Evaluates the model and returns true labels and predictions."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def calculate_metrics(y_true, y_pred, class_names):
    """Calculates requested metrics: Top-1 Acc and Avg Class Acc."""
    # 1. Top-1 Accuracy
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    top1_acc = correct / total
    
    # 2. Average Accuracy per Class
    class_accs = []
    for i in range(len(class_names)):
        idx = (y_true == i)
        if np.sum(idx) > 0:
            acc = np.sum(y_true[idx] == y_pred[idx]) / np.sum(idx)
            class_accs.append(acc)
        else:
            class_accs.append(0.0)
            
    avg_class_acc = np.mean(class_accs)
    
    return top1_acc, avg_class_acc, class_accs

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='saved_models')
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation script using device: {device}")
    
    # We only need the validation loader for final metric reporting
    print("Loading validation dataset...")
    _, val_loader, class_names = get_data_loaders(args.data_dir, batch_size=32)
    num_classes = len(class_names)
    
    models_to_eval = [
        ('cnn', os.path.join(args.save_dir, 'best_cnn_model.pth')),
        ('resnet', os.path.join(args.save_dir, 'best_resnet_model.pth'))
    ]
    
    results = {}
    
    for model_type, model_path in models_to_eval:
        print(f"\n--- Evaluating {model_type.upper()} ---")
        model = load_model(model_path, model_type, num_classes, device)
        
        if model is None:
            continue
            
        print(f"Model loaded from {model_path}. Running inference...")
        y_true, y_pred = evaluate(model, val_loader, device, class_names)
        
        top1_acc, avg_class_acc, class_accs = calculate_metrics(y_true, y_pred, class_names)
        
        print(f"Top-1 Accuracy: {top1_acc:.4f}")
        print(f"Average Accuracy per Class: {avg_class_acc:.4f}")
        for i, name in enumerate(class_names):
            print(f"  - {name}: {class_accs[i]:.4f}")
            
        # Plot confusion matrix
        cm_path = os.path.join(args.save_dir, f'confusion_matrix_{model_type}.png')
        plot_confusion_matrix(y_true, y_pred, class_names, 
                              f'Confusion Matrix: {model_type.upper()}', cm_path)
        print(f"Saved confusion matrix to {cm_path}")
        
        results[model_type] = {
            'Top-1 Accuracy': top1_acc,
            'Average Class Accuracy': avg_class_acc
        }
        
    print("\n================ FINAL RESULTS ================")
    for m_type, metrics in results.items():
        print(f"Model: {m_type.upper()}")
        print(f"  Top-1 Accuracy                 : {metrics['Top-1 Accuracy']:.4f}")
        print(f"  Average Accuracy per Class     : {metrics['Average Class Accuracy']:.4f}")

if __name__ == "__main__":
    main()
