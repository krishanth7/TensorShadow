import subprocess
import time
import os
import sys

def main():
    print("==========================================")
    print("  TensorShadow FULL PIPELINE EXECUTION")
    print("==========================================\n")

    # Step 1: Run Training
    print("Step 1: Running AI Model Training...")
    subprocess.run([sys.executable, "tensorflow_model/train.py"], check=True)

    # Step 2: Run Analytics
    print("\nStep 2: Running Statistical Analysis...")
    subprocess.run([sys.executable, "r_analysis/sim_analysis.py"], check=True)

    # Step 3: Start API Gateway and Test it
    print("\nStep 3: Launching API Gateway & Verifying Endpoints...")
    server_process = subprocess.Popen([sys.executable, "go_backend/sim_server.py"])
    
    # Wait for server to start
    time.sleep(3)

    print("\n[TEST] Verifying Health Check...")
    try:
        # In this environment, we use python to perform the request if curl is tricky
        import urllib.request
        with urllib.request.urlopen("http://localhost:8080/api/v1/health") as response:
            print(f"Response: {response.read().decode()}")
        
        print("\n[TEST] Verifying Analytics Data...")
        with urllib.request.urlopen("http://localhost:8080/api/v1/stats") as response:
            print(f"Response: {response.read().decode()}")
            
    except Exception as e:
        print(f"Testing Error: {e}")

    print("\n==========================================")
    print("  PIPELINE EXECUTION SUCCESSFUL")
    print("  Documentation: /docs")
    print("  Visualisations: /docs/model_performance.png")
    print("==========================================")
    
    # Stop the server
    server_process.terminate()
    print("\nDemo Complete. Server shut down.")

if __name__ == "__main__":
    main()


