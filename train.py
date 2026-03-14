import argparse
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.data_loader import get_data_loaders
from models.model_cnn import CustomCNN
from models.model_resnet import ResNet18Transfer

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate Top-1 Accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device, num_classes=10):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For Average Accuracy Per Class
    class_correct = list(0. for _ in range(num_classes))
    class_total = list(0. for _ in range(num_classes))
    
    with torch.no_grad(): # No gradients needed for evaluation
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Overall Top-1 Acc
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Class-specific Acc
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                if i < len(c):
                    label = labels[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # Calculate Average per class
    avg_per_class = sum([class_correct[i]/class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]) / num_classes
    
    return epoch_loss, epoch_acc, avg_per_class, class_correct, class_total

def main():
    parser = argparse.ArgumentParser(description="Image Classification Training Script")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the 'train' folder")
    parser.add_argument('--model', type=str, choices=['cnn', 'resnet'], required=True, help="Model to train")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--save_dir', type=str, default='saved_models', help="Directory to save the best model")
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading data from {args.data_dir}...")
    train_loader, val_loader, class_names = get_data_loaders(
        data_dir=args.data_dir, 
        batch_size=args.batch_size
    )
    
    # 2. Initialize Model
    num_classes = len(class_names)
    if args.model == 'cnn':
        print("Initializing Custom CNN Baseline Model...")
        model = CustomCNN(num_classes=num_classes)
        # Weight decay helps prevent overfitting
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) 
    else:
        print("Initializing ResNet18 Transfer Learning Model...")
        model = ResNet18Transfer(num_classes=num_classes, freeze_backbone=False) # Fine-tune all
        # Use a smaller learning rate for fine-tuning
        optimizer = AdamW(model.parameters(), lr=args.lr / 10, weight_decay=1e-4)
    
    model = model.to(device)
    
    # 3. Training Setup
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, f"best_{args.model}_model.pth")
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    print("\n--- Starting Training ---")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_avg_class_acc, _, _ = validate(model, val_loader, criterion, device, num_classes)
        
        # Log 
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{args.epochs} [{epoch_time:.0f}s] "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AvgClassAcc: {val_avg_class_acc:.4f}")
              
        # Scheduler Step
        scheduler.step(val_loss)
        
        # Early Stopping & Checkpoint Save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f" -> Best Model Saved to {best_model_path} (Loss: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print(f"\n--- Early stopping triggered after {epoch+1} epochs ---")
                break
                
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
