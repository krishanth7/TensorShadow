import os
import json
import numpy as np
import matplotlib.pyplot as plt

def run_analysis():
    print("\n--- TensorShadow R-Analytics Simulation ---")
    
    if not os.path.exists("data/training_stats.json"):
        print("Error: Training stats not found. Please run training first.")
        return

    with open("data/training_stats.json", "r") as f:
        stats = json.load(f)

    epochs = range(1, len(stats['accuracy']) + 1)
    
    # Generate high-quality visualization
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, stats['accuracy'], marker='o', linestyle='-', color='#276DC3', label='Accuracy')
    plt.plot(epochs, stats['loss'], marker='x', linestyle='--', color='#FF6F00', label='Loss')
    
    plt.title('TensorShadow: Model Intelligence Convergence', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/model_performance.png")
    print("Visualization saved to: docs/model_performance.png")
    
    # Export clean summary
    summary = {
        "final_accuracy": stats['accuracy'][-1],
        "timestamp": "2026-02-21T16:55:00Z",
        "engine": "Fallback-Hybrid"
    }
    with open("data/stats_summary.json", "w") as f:
        json.dump(summary, f)
    
    print("--- Analytics Complete ---\n")

if __name__ == "__main__":
    run_analysis()

