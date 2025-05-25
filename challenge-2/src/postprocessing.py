import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=['Non-Soil', 'Soil'], output_dict=True)
    
    metrics = {
        "overall": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1])
        },
        "per_class": {
            "non_soil": {
                "precision": float(report['Non-Soil']['precision']),
                "recall": float(report['Non-Soil']['recall']),
                "f1_score": float(report['Non-Soil']['f1-score']),
                "support": int(report['Non-Soil']['support'])
            },
            "soil": {
                "precision": float(report['Soil']['precision']),
                "recall": float(report['Soil']['recall']),
                "f1_score": float(report['Soil']['f1-score']),
                "support": int(report['Soil']['support'])
            }
        },
        "macro_avg": {
            "precision": float(report['macro avg']['precision']),
            "recall": float(report['macro avg']['recall']),
            "f1_score": float(report['macro avg']['f1-score'])
        },
        "weighted_avg": {
            "precision": float(report['weighted avg']['precision']),
            "recall": float(report['weighted avg']['recall']),
            "f1_score": float(report['weighted avg']['f1-score'])
        }
    }
    
    return metrics


def save_metrics(metrics, filepath):
    """Save metrics to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")


def analyze_predictions(y_true, y_pred, test_preds=None):
    """Analyze model predictions and provide insights"""
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Overall performance
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"Overall F1-Score: {metrics['overall']['f1_score']:.4f}")
    print(f"Overall Precision: {metrics['overall']['precision']:.4f}")
    print(f"Overall Recall: {metrics['overall']['recall']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Negative (Non-Soil correctly classified): {metrics['confusion_matrix']['true_negative']}")
    print(f"False Positive (Non-Soil misclassified as Soil): {metrics['confusion_matrix']['false_positive']}")
    print(f"False Negative (Soil misclassified as Non-Soil): {metrics['confusion_matrix']['false_negative']}")
    print(f"True Positive (Soil correctly classified): {metrics['confusion_matrix']['true_positive']}")
    
    print("\nPer-Class Performance:")
    print(f"Non-Soil - Precision: {metrics['per_class']['non_soil']['precision']:.4f}, "
          f"Recall: {metrics['per_class']['non_soil']['recall']:.4f}, "
          f"F1: {metrics['per_class']['non_soil']['f1_score']:.4f}")
    print(f"Soil - Precision: {metrics['per_class']['soil']['precision']:.4f}, "
          f"Recall: {metrics['per_class']['soil']['recall']:.4f}, "
          f"F1: {metrics['per_class']['soil']['f1_score']:.4f}")
    
    # Test set analysis if provided
    if test_preds is not None:
        print(f"\nTest Set Predictions:")
        test_dist = pd.Series(test_preds).value_counts().sort_index()
        for label, count in test_dist.items():
            label_name = "Non-Soil" if label == 0 else "Soil"
            print(f"{label_name} (Class {label}): {count} images ({count/len(test_preds)*100:.1f}%)")
    
    return metrics


def create_performance_summary(metrics, model_info=None):
    """Create a comprehensive performance summary"""
    
    summary = {
        "model_info": model_info if model_info else {
            "architecture": "ResNet50-based Soil Classifier",
            "training_strategy": "Transfer Learning with Synthetic Negatives",
            "image_size": "224x224",
            "augmentations": "Random crops, flips, rotations, color jitter"
        },
        "performance_metrics": metrics,
        "interpretation": {
            "model_strength": [],
            "areas_for_improvement": [],
            "recommendations": []
        }
    }
    
    # Analyze performance and add interpretations
    overall_f1 = metrics['overall']['f1_score']
    soil_recall = metrics['per_class']['soil']['recall']
    non_soil_recall = metrics['per_class']['non_soil']['recall']
    
    if overall_f1 > 0.95:
        summary["interpretation"]["model_strength"].append("Excellent overall performance")
    elif overall_f1 > 0.85:
        summary["interpretation"]["model_strength"].append("Good overall performance")
    
    if soil_recall > 0.95:
        summary["interpretation"]["model_strength"].append("Very good at detecting soil images")
    
    if non_soil_recall > 0.90:
        summary["interpretation"]["model_strength"].append("Good at rejecting non-soil images")
    elif non_soil_recall < 0.80:
        summary["interpretation"]["areas_for_improvement"].append("Could improve non-soil detection")
        summary["interpretation"]["recommendations"].append("Consider adding more diverse negative samples")
    
    if metrics['confusion_matrix']['false_positive'] > metrics['confusion_matrix']['false_negative']:
        summary["interpretation"]["areas_for_improvement"].append("Tends to over-classify as soil")
        summary["interpretation"]["recommendations"].append("Consider adjusting classification threshold")
    
    return summary