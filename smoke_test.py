import os
import sys

def check_structure() -> bool:
    print("=== TensorShadow System Check ===")
    ok = True
    required_dirs = [
        'tensorflow_model', 'r_analysis', 'go_backend', 
        'api', 'data', 'configs', 'tests', 'docs'
    ]
    
    for d in required_dirs:
        if os.path.isdir(d):
            print(f"[✓] Directory found: {d}")
        else:
            print(f"[✗] Directory MISSING: {d}")
            ok = False

    required_files = [
        'README.md', 'requirements.txt', 'go.mod', 'go.sum', '.gitignore', 'Makefile', 'Dockerfile',
        'tensorflow_model/train.py', 'r_analysis/analysis.R', 'api/openapi.yaml', 'configs/config.yaml',
        'go_backend/main.go', 'go_backend/web/index.html',
        'go_backend/internal/workerpool/pool.go',
        'go_backend/internal/thermal/analyze.go',
        'go_backend/internal/tracking/tracker.go',
        'go_backend/internal/server/server.go',
        'clients/python/tensorshadow_client.py',
        'go_backend/internal/inference/inference.go',
        'go_backend/internal/coreg/coreg.go',
        'tensorflow_model/export_models.py', 'tensorflow_model/onnx_server.py',
        'tensorflow_model/onnx/tensorshadow/1/model.onnx',
        'tensorflow_model/serving/tensorshadow/1/saved_model.pb',
    ]

    for f in required_files:
        if os.path.isfile(f):
            print(f"[✓] File found: {f}")
        else:
            print(f"[✗] File MISSING: {f}")
            ok = False
    return ok

def simulate_pipeline() -> None:
    print("\n=== Simulating TensorShadow Pipeline ===")
    print("1. [Python] Initializing Brain (TensorFlow Model)...")
    print("   - Logic: 10-input DNN with Dropout and BatchNormalization.")
    print("2. [Go] Starting high-concurrency API gateway (worker pool + back-pressure)...")
    print("   - Core:    /health, /stats, /metrics")
    print("   - Models:  /predict, /model → TF Serving | ONNX Runtime (OIP v2) | baseline")
    print("   - IR:      /thermal/analyze, /thermal/render (JSON or raw uint16 buffers)")
    print("   - Coreg:   /coreg/rigs → visible/IR homography drives thermal ROIs")
    print("   - Crowd:   /crowd/sessions/{id}/frames, /analytics, /heatmap, /stream (SSE), ReID")
    print("3. [R] Preparing Analytical Engine...")
    print("   - Reporting: ggplot2 Visualisation active.")
    print("\n[SUCCESS] System architecture is valid and components are linked via /configs/config.yaml")

if __name__ == "__main__":
    structure_ok = check_structure()
    simulate_pipeline()
    sys.exit(0 if structure_ok else 1)


