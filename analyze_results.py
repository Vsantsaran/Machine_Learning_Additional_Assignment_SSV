#!/usr/bin/env python3
"""
Analyze and visualize training results for Cotton Leaf Disease Detection
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_training_curves(metrics_file, save_dir):
    """Plot training and validation curves"""
    print("Plotting training curves...")

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    epochs = [m['epoch'] for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    train_acc = [m['train_accuracy'] for m in metrics]
    val_loss = [m['val_loss'] for m in metrics]
    val_acc = [m['val_accuracy'] for m in metrics]
    val_bal_acc = [m['val_balanced_accuracy'] for m in metrics]
    val_f1 = [m['val_macro_f1'] for m in metrics]
    lr = [m['learning_rate'] for m in metrics]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

    # Loss curve
    axes[0, 0].plot(epochs, train_loss, label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[0, 1].plot(epochs, train_acc, label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_acc, label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Balanced accuracy
    axes[0, 2].plot(epochs, val_bal_acc, label='Val Balanced Acc', linewidth=2, color='green')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Balanced Accuracy')
    axes[0, 2].set_title('Validation Balanced Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 0].plot(epochs, val_f1, label='Val Macro F1', linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation Macro F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, lr, label='Learning Rate', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Summary statistics
    best_epoch = np.argmax(val_acc)
    summary_text = f"""
    Best Results at Epoch {epochs[best_epoch]}:

    Training Accuracy: {train_acc[best_epoch]:.4f}
    Validation Accuracy: {val_acc[best_epoch]:.4f}
    Validation Balanced Acc: {val_bal_acc[best_epoch]:.4f}
    Validation Macro F1: {val_f1[best_epoch]:.4f}

    Final Epoch {epochs[-1]}:
    Training Accuracy: {train_acc[-1]:.4f}
    Validation Accuracy: {val_acc[-1]:.4f}
    Validation Balanced Acc: {val_bal_acc[-1]:.4f}
    Validation Macro F1: {val_f1[-1]:.4f}
    """

    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10,
                    verticalalignment='center', family='monospace')
    axes[1, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(conf_matrix_file, class_names, save_dir):
    """Plot confusion matrix"""
    print("Plotting confusion matrix...")

    conf_matrix = np.load(conf_matrix_file)

    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Confusion Matrix', fontsize=16, fontweight='bold')

    # Raw counts
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Normalized
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_per_class_metrics(test_metrics_file, class_names, save_dir):
    """Plot per-class performance metrics"""
    print("Plotting per-class metrics...")

    with open(test_metrics_file, 'r') as f:
        metrics = json.load(f)

    per_class = metrics['per_class_metrics']

    classes = []
    precision = []
    recall = []
    f1_score = []
    support = []

    for class_name in class_names:
        if class_name in per_class:
            classes.append(class_name)
            precision.append(per_class[class_name]['precision'])
            recall.append(per_class[class_name]['recall'])
            f1_score.append(per_class[class_name]['f1_score'])
            support.append(per_class[class_name]['support'])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')

    x = np.arange(len(classes))
    width = 0.25

    # Precision, Recall, F1
    axes[0, 0].bar(x - width, precision, width, label='Precision', alpha=0.8)
    axes[0, 0].bar(x, recall, width, label='Recall', alpha=0.8)
    axes[0, 0].bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Precision, Recall, and F1-Score by Class')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.05])

    # Support (sample counts)
    axes[0, 1].bar(x, support, color='steelblue', alpha=0.8)
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_title('Test Set Support by Class')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(classes, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # F1-Score comparison
    colors = ['green' if f >= 0.9 else 'orange' if f >= 0.8 else 'red' for f in f1_score]
    axes[1, 0].barh(classes, f1_score, color=colors, alpha=0.8)
    axes[1, 0].set_xlabel('F1-Score')
    axes[1, 0].set_ylabel('Class')
    axes[1, 0].set_title('F1-Score by Class (Color-coded)')
    axes[1, 0].set_xlim([0, 1.05])
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='0.9 threshold')
    axes[1, 0].axvline(x=0.8, color='orange', linestyle='--', alpha=0.5, label='0.8 threshold')
    axes[1, 0].legend()

    # Summary statistics table
    summary_data = [
        ['Metric', 'Value'],
        ['Overall Accuracy', f"{metrics['test_accuracy']:.4f}"],
        ['Balanced Accuracy', f"{metrics['test_balanced_accuracy']:.4f}"],
        ['Macro Precision', f"{metrics['test_macro_precision']:.4f}"],
        ['Macro Recall', f"{metrics['test_macro_recall']:.4f}"],
        ['Macro F1', f"{metrics['test_macro_f1']:.4f}"],
        ['Weighted Precision', f"{metrics['test_weighted_precision']:.4f}"],
        ['Weighted Recall', f"{metrics['test_weighted_recall']:.4f}"],
        ['Weighted F1', f"{metrics['test_weighted_f1']:.4f}"],
    ]

    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=summary_data, cellLoc='left', loc='center',
                              colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    axes[1, 1].set_title('Overall Test Metrics', fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'per_class_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class metrics saved to: {save_path}")
    plt.close()


def generate_summary_report(log_dir, save_dir):
    """Generate comprehensive summary report"""
    print("Generating summary report...")

    # Load all metrics
    training_metrics_file = os.path.join(log_dir, 'training_metrics.json')
    test_metrics_file = os.path.join(log_dir, 'test_metrics.json')
    config_file = os.path.join(log_dir, 'config.json')

    with open(training_metrics_file, 'r') as f:
        training_metrics = json.load(f)

    with open(test_metrics_file, 'r') as f:
        test_metrics = json.load(f)

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Find best epoch
    val_accs = [m['val_accuracy'] for m in training_metrics]
    best_epoch = np.argmax(val_accs)
    best_metrics = training_metrics[best_epoch]

    # Generate report
    report = f"""
================================================================================
COTTON LEAF DISEASE DETECTION - TRAINING SUMMARY REPORT
================================================================================

EXPERIMENT CONFIGURATION
--------------------------------------------------------------------------------
Model Architecture:
  - Width Multiplier: {config['width_multiplier']}
  - Expansion Ratio: {config['expansion_ratio']}
  - Transformer Depth: {config['transformer_depth']}
  - Number of Attention Heads: {config['num_heads']}
  - Patch Size: {config['patch_size']}
  - Multi-Scale Patch Sizes: {config['patch_sizes']}
  - Dropout Rate: {config['dropout']}
  - Stochastic Depth: {config['stochastic_depth']}

Training Configuration:
  - Total Epochs: {config['epochs']}
  - Batch Size (per GPU): {config['batch_size']}
  - Accumulation Steps: {config['accumulation_steps']}
  - Effective Batch Size: {config['batch_size'] * 4 * config['accumulation_steps']}
  - Initial Learning Rate: {config['lr']}
  - Minimum Learning Rate: {config['lr_min']}
  - Weight Decay: {config['weight_decay']}
  - Warmup Epochs: {config['warmup_epochs']}

Data Configuration:
  - Image Size: {config['image_size']} x {config['image_size']}
  - Train Ratio: {config['train_ratio']}
  - Validation Ratio: {config['val_ratio']}
  - Test Ratio: {config['test_ratio']}
  - Data Augmentation: MixUp, CutMix, RandAugment
  - Loss Function: {'Focal Loss' if config['use_focal_loss'] else 'Label Smoothing'}

TRAINING RESULTS
--------------------------------------------------------------------------------
Best Validation Performance (Epoch {best_metrics['epoch']}):
  - Training Loss: {best_metrics['train_loss']:.4f}
  - Training Accuracy: {best_metrics['train_accuracy']:.4f}
  - Validation Loss: {best_metrics['val_loss']:.4f}
  - Validation Accuracy: {best_metrics['val_accuracy']:.4f}
  - Validation Balanced Accuracy: {best_metrics['val_balanced_accuracy']:.4f}
  - Validation Macro F1: {best_metrics['val_macro_f1']:.4f}
  - Validation Weighted F1: {best_metrics['val_weighted_f1']:.4f}

Final Epoch Performance (Epoch {training_metrics[-1]['epoch']}):
  - Training Loss: {training_metrics[-1]['train_loss']:.4f}
  - Training Accuracy: {training_metrics[-1]['train_accuracy']:.4f}
  - Validation Loss: {training_metrics[-1]['val_loss']:.4f}
  - Validation Accuracy: {training_metrics[-1]['val_accuracy']:.4f}
  - Validation Balanced Accuracy: {training_metrics[-1]['val_balanced_accuracy']:.4f}
  - Validation Macro F1: {training_metrics[-1]['val_macro_f1']:.4f}
  - Validation Weighted F1: {training_metrics[-1]['val_weighted_f1']:.4f}

TEST SET RESULTS
--------------------------------------------------------------------------------
Overall Metrics:
  - Test Accuracy: {test_metrics['test_accuracy']:.4f}
  - Test Balanced Accuracy: {test_metrics['test_balanced_accuracy']:.4f}
  - Test Macro Precision: {test_metrics['test_macro_precision']:.4f}
  - Test Macro Recall: {test_metrics['test_macro_recall']:.4f}
  - Test Macro F1: {test_metrics['test_macro_f1']:.4f}
  - Test Weighted Precision: {test_metrics['test_weighted_precision']:.4f}
  - Test Weighted Recall: {test_metrics['test_weighted_recall']:.4f}
  - Test Weighted F1: {test_metrics['test_weighted_f1']:.4f}

Per-Class Performance:
"""

    # Add per-class metrics
    class_names = [
        "Bacterial_Blight", "Curl_Virus", "Healthy_Leaf",
        "Herbicide_Growth_Damage", "Leaf_Hopper_Jassids",
        "Leaf_Redding", "Leaf_Variegation"
    ]

    report += f"\n{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n"
    report += "-" * 80 + "\n"

    for class_name in class_names:
        if class_name in test_metrics['per_class_metrics']:
            metrics = test_metrics['per_class_metrics'][class_name]
            report += f"{class_name:<30} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
            report += f"{metrics['f1_score']:<12.4f} {metrics['support']:<10}\n"

    report += "\n" + "="*80 + "\n"
    report += "ANALYSIS COMPLETE\n"
    report += "="*80 + "\n"

    # Save report
    report_path = os.path.join(save_dir, 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    print(report)
    print(f"\nSummary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='Directory containing logs and metrics')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save analysis results (default: same as log-dir)')
    args = parser.parse_args()

    log_dir = args.log_dir
    output_dir = args.output_dir if args.output_dir else log_dir

    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Cotton Leaf Disease Detection - Results Analysis")
    print("="*80)
    print(f"Log Directory: {log_dir}")
    print(f"Output Directory: {output_dir}")
    print("="*80)

    # Class names
    class_names = [
        "Bacterial\nBlight",
        "Curl\nVirus",
        "Healthy\nLeaf",
        "Herbicide\nDamage",
        "Leaf\nHopper",
        "Leaf\nReddening",
        "Leaf\nVariegation"
    ]

    # Check required files
    training_metrics_file = os.path.join(log_dir, 'training_metrics.json')
    test_metrics_file = os.path.join(log_dir, 'test_metrics.json')
    conf_matrix_file = os.path.join(log_dir, 'confusion_matrix.npy')

    if not os.path.exists(training_metrics_file):
        print(f"Warning: Training metrics file not found: {training_metrics_file}")
    else:
        plot_training_curves(training_metrics_file, output_dir)

    if not os.path.exists(conf_matrix_file):
        print(f"Warning: Confusion matrix file not found: {conf_matrix_file}")
    else:
        plot_confusion_matrix(conf_matrix_file, class_names, output_dir)

    if not os.path.exists(test_metrics_file):
        print(f"Warning: Test metrics file not found: {test_metrics_file}")
    else:
        plot_per_class_metrics(test_metrics_file,
                               ["Bacterial_Blight", "Curl_Virus", "Healthy_Leaf",
                                "Herbicide_Growth_Damage", "Leaf_Hopper_Jassids",
                                "Leaf_Redding", "Leaf_Variegation"],
                               output_dir)

    # Generate summary report
    if os.path.exists(training_metrics_file) and os.path.exists(test_metrics_file):
        generate_summary_report(log_dir, output_dir)

    print("\n" + "="*80)
    print("Analysis completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
