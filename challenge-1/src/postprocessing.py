import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import json
import os


def evaluate_model(model, val_dataset, device, batch_size=32):
    """Evaluate model on validation set and return predictions"""
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_preds, all_labels


def generate_confusion_matrix(all_labels, all_preds, class_names, save_path):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", 
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_classification_metrics(all_labels, all_preds, class_names, save_path):
    """Generate and save classification metrics as JSON"""
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Classification metrics saved to {save_path}")


def generate_test_predictions(model, test_dataset, test_df, device, batch_size=32):
    """Generate predictions for test set"""
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    test_predictions = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_predictions.extend(predicted.cpu().numpy())

    # Create submission DataFrame
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    submission_df = pd.DataFrame({
        "image_id": test_df["image_id"],
        "soil_type": [idx_to_class[pred] for pred in test_predictions],
    })
    
    return submission_df


def plot_training_history(history, save_path):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")


def create_model_architecture_diagram(save_path):
    """Create a simple model architecture description"""
    architecture_info = {
        "model_type": "ResNet50 (pretrained) or Custom CNN",
        "input_shape": [3, 224, 224],
        "num_classes": 4,
        "classes": ["Alluvial soil", "Black Soil", "Clay soil", "Red soil"],
        "preprocessing": {
            "resize": [224, 224],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "augmentation": [
                "RandomHorizontalFlip",
                "RandomRotation(10)",
                "ColorJitter(brightness=0.2, contrast=0.2)"
            ]
        },
        "training_params": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "early_stopping_patience": 5,
            "lr_scheduler": "ReduceLROnPlateau"
        }
    }
    
    # Save architecture info as JSON
    with open(save_path.replace('.png', '.json'), 'w') as f:
        json.dump(architecture_info, f, indent=4)
    
    # Create a simple text-based architecture diagram
    with open(save_path.replace('.png', '.txt'), 'w') as f:
        f.write("SOIL CLASSIFICATION MODEL ARCHITECTURE\n")
        f.write("="*50 + "\n\n")
        f.write("Input: RGB Image (224x224x3)\n")
        f.write("|\n")
        f.write("v\n")
        f.write("Data Augmentation (Training only)\n")
        f.write("|\n")  
        f.write("v\n")
        f.write("Normalization\n")
        f.write("|\n")
        f.write("v\n")
        f.write("Backbone: ResNet50 (pretrained) or Custom CNN\n")
        f.write("|\n")
        f.write("v\n")
        f.write("Feature Extraction\n")
        f.write("|\n")
        f.write("v\n")
        f.write("Classifier Head:\n")
        f.write("  - Linear(features -> 512)\n")
        f.write("  - ReLU\n")
        f.write("  - Dropout(0.3)\n")
        f.write("  - Linear(512 -> 4)\n")
        f.write("|\n")
        f.write("v\n")
        f.write("Output: 4 Classes (Alluvial, Black, Clay, Red soil)\n")
    
    print(f"Model architecture info saved to {save_path.replace('.png', '.json')}")
    print(f"Model architecture diagram saved to {save_path.replace('.png', '.txt')}")


def process_results(model, val_dataset, test_dataset, test_df, history, device, output_dir):
    """Complete post-processing pipeline"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    metrics_path = os.path.join(output_dir, "ml-metrics.json")
    history_plot_path = os.path.join(output_dir, "training_history.png")
    architecture_path = os.path.join(output_dir, "architecture.png")
    submission_path = os.path.join(output_dir, "submission.csv")
    
    # Class names
    class_names = ["Alluvial soil", "Black Soil", "Clay soil", "Red soil"]
    
    # Evaluate model
    all_preds, all_labels = evaluate_model(model, val_dataset, device)
    
    # Generate confusion matrix
    generate_confusion_matrix(all_labels, all_preds, class_names, confusion_matrix_path)
    
    # Save classification metrics
    save_classification_metrics(all_labels, all_preds, class_names, metrics_path)
    
    # Plot training history
    plot_training_history(history, history_plot_path)
    
    # Create model architecture info
    create_model_architecture_diagram(architecture_path)
    
    # Generate test predictions and save submission
    submission_df = generate_test_predictions(model, test_dataset, test_df, device)
    submission_df.to_csv(submission_path, index=False)
    print(f"Predictions saved to {submission_path}")
    
    return submission_df
