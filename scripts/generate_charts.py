"""
Generate visualization charts for RAG prototype evaluation results.

Usage:
    uv run python scripts/generate_charts.py

Outputs charts to data/charts/ directory.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Chart styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']  # P1, P2, P3, P4
PROTOTYPE_NAMES = ['P1_function', 'P2_ast', 'P3_context', 'P4_graph']
PROTOTYPE_LABELS = ['P1 Function', 'P2 AST', 'P3 Context', 'P4 Graph']
CATEGORY_LABELS = ['Cat 1: Simple\nLookup', 'Cat 2: Local\nContext', 'Cat 3: Global\nRelational']


def load_data():
    """Load evaluation data from JSON files."""
    data_dir = Path(__file__).parent.parent / "data"
    
    with open(data_dir / "evaluation_results_raw.json") as f:
        raw_results = json.load(f)
    
    with open(data_dir / "evaluation_final.json") as f:
        final_results = json.load(f)
    
    return raw_results, final_results


def extract_metrics(raw_results, final_results):
    """Extract metrics for each prototype and category."""
    metrics = {
        proto: {
            "correct_by_cat": [0, 0, 0],
            "partial_by_cat": [0, 0, 0],
            "incorrect_by_cat": [0, 0, 0],
            "latency_by_cat": [[], [], []],
            "selfcheck_by_cat": [[], [], []],
        }
        for proto in PROTOTYPE_NAMES
    }
    
    # Process final evaluations
    for eval_item in final_results["evaluations"]:
        q_id = eval_item["question_id"]
        cat_idx = (q_id - 1) // 10  # 0, 1, or 2
        
        for proto in PROTOTYPE_NAMES:
            proto_eval = eval_item["prototype_evaluations"][proto]
            
            if proto_eval["answered_correctly"]:
                if proto_eval.get("partial_answer"):
                    metrics[proto]["partial_by_cat"][cat_idx] += 1
                else:
                    metrics[proto]["correct_by_cat"][cat_idx] += 1
            else:
                if proto_eval.get("partial_answer"):
                    metrics[proto]["partial_by_cat"][cat_idx] += 1
                else:
                    metrics[proto]["incorrect_by_cat"][cat_idx] += 1
    
    # Process raw results for latency and selfcheck
    for result in raw_results["results"]:
        q_id = result["question_id"]
        cat_idx = (q_id - 1) // 10
        
        for proto in PROTOTYPE_NAMES:
            proto_result = result["prototype_results"].get(proto, {})
            
            if proto_result.get("latency_ms"):
                metrics[proto]["latency_by_cat"][cat_idx].append(proto_result["latency_ms"])
            
            if proto_result.get("selfcheck_score") is not None:
                metrics[proto]["selfcheck_by_cat"][cat_idx].append(proto_result["selfcheck_score"])
    
    return metrics


def plot_accuracy_by_category(metrics, output_dir):
    """Bar chart: Correct answers per category for each prototype."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(3)  # Categories
    width = 0.2
    
    for i, (proto, label) in enumerate(zip(PROTOTYPE_NAMES, PROTOTYPE_LABELS)):
        correct = metrics[proto]["correct_by_cat"]
        ax.bar(x + i * width, correct, width, label=label, color=COLORS[i], alpha=0.9)
    
    ax.set_xlabel('Question Category', fontsize=12)
    ax.set_ylabel('Correct Answers (out of 10)', fontsize=12)
    ax.set_title('Prototype Accuracy by Question Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(CATEGORY_LABELS)
    ax.set_ylim(0, 11)
    ax.legend(loc='upper left')
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Perfect Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_category.png", dpi=150)
    plt.close()
    print("Saved: accuracy_by_category.png")


def plot_error_breakdown(metrics, output_dir):
    """Stacked bar chart: Correct, Partial, Incorrect for each prototype."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(4)  # Prototypes
    
    correct = [sum(metrics[p]["correct_by_cat"]) for p in PROTOTYPE_NAMES]
    partial = [sum(metrics[p]["partial_by_cat"]) for p in PROTOTYPE_NAMES]
    incorrect = [sum(metrics[p]["incorrect_by_cat"]) for p in PROTOTYPE_NAMES]
    
    ax.bar(x, correct, label='Correct', color='#2ecc71')
    ax.bar(x, partial, bottom=correct, label='Partial', color='#f39c12')
    ax.bar(x, incorrect, bottom=[c + p for c, p in zip(correct, partial)], 
           label='Incorrect', color='#e74c3c')
    
    ax.set_xlabel('Prototype', fontsize=12)
    ax.set_ylabel('Number of Answers', fontsize=12)
    ax.set_title('Answer Quality Breakdown by Prototype', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(PROTOTYPE_LABELS)
    ax.set_ylim(0, 32)
    ax.legend(loc='upper right')
    
    # Add percentage labels
    for i, (c, p, inc) in enumerate(zip(correct, partial, incorrect)):
        total = c + p + inc
        pct = c / total * 100
        ax.text(i, c / 2, f'{c}', ha='center', va='center', fontweight='bold', color='white')
        if p > 0:
            ax.text(i, c + p / 2, f'{p}', ha='center', va='center', fontweight='bold', color='white')
        if inc > 0:
            ax.text(i, c + p + inc / 2, f'{inc}', ha='center', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(output_dir / "error_breakdown.png", dpi=150)
    plt.close()
    print("Saved: error_breakdown.png")


def plot_latency_by_category(metrics, output_dir):
    """Grouped bar chart: Average latency per category."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(3)  # Categories
    width = 0.2
    
    for i, (proto, label) in enumerate(zip(PROTOTYPE_NAMES, PROTOTYPE_LABELS)):
        avg_latency = [
            np.mean(metrics[proto]["latency_by_cat"][cat]) / 1000  # Convert to seconds
            if metrics[proto]["latency_by_cat"][cat] else 0
            for cat in range(3)
        ]
        ax.bar(x + i * width, avg_latency, width, label=label, color=COLORS[i], alpha=0.9)
    
    ax.set_xlabel('Question Category', fontsize=12)
    ax.set_ylabel('Average Latency (seconds)', fontsize=12)
    ax.set_title('Average Response Time by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(CATEGORY_LABELS)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / "latency_by_category.png", dpi=150)
    plt.close()
    print("Saved: latency_by_category.png")


def plot_selfcheck_consistency(metrics, output_dir):
    """Box plot: SelfCheck score distribution by prototype."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_scores = []
    labels = []
    
    for proto, label in zip(PROTOTYPE_NAMES, PROTOTYPE_LABELS):
        scores = []
        for cat in range(3):
            scores.extend(metrics[proto]["selfcheck_by_cat"][cat])
        all_scores.append(scores)
        labels.append(label)
    
    bp = ax.boxplot(all_scores, patch_artist=True, labels=labels)
    
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Prototype', fontsize=12)
    ax.set_ylabel('SelfCheck Consistency Score', fontsize=12)
    ax.set_title('Response Consistency (SelfCheck) by Prototype', fontsize=14, fontweight='bold')
    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Hallucination Threshold')
    ax.legend(loc='lower left')
    ax.set_ylim(0.3, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / "selfcheck_consistency.png", dpi=150)
    plt.close()
    print("Saved: selfcheck_consistency.png")


def plot_selfcheck_by_category(metrics, output_dir):
    """Grouped bar chart: Average SelfCheck score by category."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(3)  # Categories
    width = 0.2
    
    for i, (proto, label) in enumerate(zip(PROTOTYPE_NAMES, PROTOTYPE_LABELS)):
        avg_score = [
            np.mean(metrics[proto]["selfcheck_by_cat"][cat])
            if metrics[proto]["selfcheck_by_cat"][cat] else 0
            for cat in range(3)
        ]
        ax.bar(x + i * width, avg_score, width, label=label, color=COLORS[i], alpha=0.9)
    
    ax.set_xlabel('Question Category', fontsize=12)
    ax.set_ylabel('Average SelfCheck Score', fontsize=12)
    ax.set_title('Response Consistency by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(CATEGORY_LABELS)
    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / "selfcheck_by_category.png", dpi=150)
    plt.close()
    print("Saved: selfcheck_by_category.png")


def plot_summary_radar(metrics, output_dir):
    """Radar/spider chart: Overall prototype comparison."""
    categories = ['Accuracy', 'Cat 1\nPerformance', 'Cat 2\nPerformance', 
                  'Cat 3\nPerformance', 'Consistency', 'Speed']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    for proto, label, color in zip(PROTOTYPE_NAMES, PROTOTYPE_LABELS, COLORS):
        # Calculate normalized metrics (0-1 scale)
        total_correct = sum(metrics[proto]["correct_by_cat"])
        cat1 = metrics[proto]["correct_by_cat"][0] / 10
        cat2 = metrics[proto]["correct_by_cat"][1] / 10
        cat3 = metrics[proto]["correct_by_cat"][2] / 10
        
        all_scores = []
        for cat in range(3):
            all_scores.extend(metrics[proto]["selfcheck_by_cat"][cat])
        consistency = np.mean(all_scores) if all_scores else 0
        
        all_latencies = []
        for cat in range(3):
            all_latencies.extend(metrics[proto]["latency_by_cat"][cat])
        avg_latency = np.mean(all_latencies) if all_latencies else 5000
        speed = 1 - min(avg_latency / 10000, 1)  # Normalize: lower is better
        
        values = [total_correct / 30, cat1, cat2, cat3, consistency, speed]
        values += values[:1]  # Close the polygon
        
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Prototype Performance Comparison', fontsize=14, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary_radar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: summary_radar.png")


def main():
    """Generate all charts."""
    print("Loading evaluation data...")
    raw_results, final_results = load_data()
    
    print("Extracting metrics...")
    metrics = extract_metrics(raw_results, final_results)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "charts"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating charts to {output_dir}...\n")
    
    plot_accuracy_by_category(metrics, output_dir)
    plot_error_breakdown(metrics, output_dir)
    plot_latency_by_category(metrics, output_dir)
    plot_selfcheck_consistency(metrics, output_dir)
    plot_selfcheck_by_category(metrics, output_dir)
    plot_summary_radar(metrics, output_dir)
    
    print(f"\n✓ All charts saved to: {output_dir}")


if __name__ == "__main__":
    main()
