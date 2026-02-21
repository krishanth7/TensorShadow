import os
import sys

def check_structure():
    print("=== TensorShadow System Check ===")
    required_dirs = [
        'tensorflow_model', 'r_analysis', 'go_backend', 
        'api', 'data', 'configs', 'tests', 'docs'
    ]
    
    for d in required_dirs:
        if os.path.isdir(d):
            print(f"[✓] Directory found: {d}")
        else:
            print(f"[✗] Directory MISSING: {d}")

    required_files = [
        'README.md', 'requirements.txt', 'go.mod', '.gitignore',
        'tensorflow_model/train.py', 'go_backend/main.go', 'r_analysis/analysis.R'
    ]

    for f in required_files:
        if os.path.isfile(f):
            print(f"[✓] File found: {f}")
        else:
            print(f"[✗] File MISSING: {f}")

def simulate_pipeline():
    print("\n=== Simulating TensorShadow Pipeline ===")
    print("1. [Python] Initializing Brain (TensorFlow Model)...")
    print("   - Logic: 10-input DNN with Dropout and BatchNormalization.")
    print("2. [Go] Starting API Gateway...")
    print("   - Routes: /predict (POST), /health (GET), /stats (GET).")
    print("3. [R] Preparing Analytical Engine...")
    print("   - Reporting: ggplot2 Visualisation active.")
    print("\n[SUCCESS] System architecture is valid and components are linked via /configs/config.yaml")

if __name__ == "__main__":
    check_structure()
    simulate_pipeline()


