"""
Main entry point for the Anti-Money Laundering (AML) GNN pipeline.

This script orchestrates the execution of the three main stages of the
machine learning pipeline:
1.  Preprocessing (Preprocess/Preprocess.py)
2.  Model Training (Model/Model.py)
3.  Prediction (Prediction/Prediction.py)

It uses subprocesses to run each part, ensuring that file paths (e.g., for
data and saved models) are resolved correctly by setting the current working
directory to the project root (where this script is located).

The script also supports command-line arguments to skip specific stages,
which is useful for development and re-running parts of the pipeline.

Usage:
    python main.py [--skip-preprocess] [--skip-train] [--skip-predict]
"""

import sys
import subprocess
import os
import argparse
import time

# --- Script Path Definitions ---
# Assumes main.py is in the project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths based on your provided structure
PREPROCESS_SCRIPT = os.path.join(BASE_DIR, 'Preprocess', 'Preprocess.py')
MODEL_SCRIPT = os.path.join(BASE_DIR, 'Model', 'Model.py')
PREDICT_SCRIPT = os.path.join(BASE_DIR, 'Prediction', 'Prediction.py')

# Get the path to the current Python executable
PYTHON_EXECUTABLE = sys.executable


def run_script(script_path):
    """
    Executes a Python script as a subprocess and streams its output.

    This function runs the specified script using the current Python executable.
    It sets the subprocess's current working directory to `BASE_DIR` to ensure
    all relative paths (e.g., 'data/acct_transaction.csv') are
    resolved correctly from the project root.

    Args:
        script_path (str): The absolute path to the Python script to execute.

    Returns:
        bool: True if the script executes successfully (exit code 0),
              False otherwise.

    Raises:
        (Prints error messages directly to stdout/stderr on failure)
    """
    script_name = os.path.relpath(script_path, BASE_DIR) # Show relative path
    print(f"\n--- Starting: {script_name} ---")
    start_time = time.time()
    
    try:
        # Use check=True to raise an error if the script fails
        # Do not capture output, allowing script's print() to show in real-time
        
        # --- MODIFICATION HERE ---
        # Add 'cwd=BASE_DIR' to force the script to run
        # from the project's root directory (where main.py is).
        subprocess.run(
            [PYTHON_EXECUTABLE, script_path], 
            check=True, 
            cwd=BASE_DIR  # Set the current working directory
        )
        # --- END MODIFICATION ---
        
        end_time = time.time()
        print(f"--- Finished: {script_name} (Duration: {end_time - start_time:.2f}s) ---")
        return True
        
    except FileNotFoundError:
        print(f"ERROR: Script not found at {script_path}")
        print("Please check your file structure.")
        return False
        
    except subprocess.CalledProcessError as e:
        # Script error messages should already be printed
        print(f"ERROR: {script_name} failed with exit code {e.returncode}.")
        return False
        
    except Exception as e:
        print(f"An unexpected error occurred while running {script_name}: {e}")
        return False

def main(args):
    """
    Runs the main machine learning pipeline stages based on arguments.

    Args:
        args (argparse.Namespace): The parsed command-line arguments,
                                   controlling which steps to skip.

    Returns:
        int: Exit code for the script. 0 for success, 1 for failure.
    """
    print("=========================================================")
    print("               Starting AML GNN Pipeline")
    print("=========================================================")
    
    total_start_time = time.time()
    
    # Step 1: Data Preprocessing
    if not args.skip_preprocess:
        print("\n[Stage 1/3] Running Data Preprocessing...")
        if not run_script(PREPROCESS_SCRIPT):
            print("Data Preprocessing failed. Halting pipeline.")
            return 1  # Return error code
    else:
        print("\n[Stage 1/3] Skipped Data Preprocessing.")
        print("Ensure 'processed_features_with_interactions.pt' already exists.")

    # Step 2: Model Training
    if not args.skip_train:
        print("\n[Stage 2/3] Running Model Training...")
        if not run_script(MODEL_SCRIPT):
            print("Model Training failed. Halting pipeline.")
            return 1
    else:
        print("\n[Stage 2/3] Skipped Model Training.")
        print("Ensure 'best_model.pt' already exists.")

    # Step 3: Run Prediction
    if not args.skip_predict:
        print("\n[Stage 3/3] Running Prediction...")
        if not run_script(PREDICT_SCRIPT):
            print("Prediction failed.")
            return 1
    else:
        print("\n[Stage 3/3] Skipped Prediction.")

    total_end_time = time.time()
    print("\n=========================================================")
    print(f"       Pipeline Finished (Total Time: {(total_end_time - total_start_time) / 60:.2f} mins)")
    if not args.skip_predict:
        print("       Check for 'submission.csv' in the root directory.")
    print("=========================================================")
    return 0  # Return success code

if __name__ == "__main__":
    """Main script entry point."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the AML GNN machine learning pipeline.")
    
    parser.add_argument(
        '--skip-preprocess',
        action='store_true',
        help="Skip the data preprocessing step (assumes 'processed_features_with_interactions.pt' exists)"
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help="Skip the model training step (assumes 'best_model.pt' exists)"
    )
    parser.add_argument(
        '--skip-predict',
        action='store_true',
        help="Skip the prediction step"
    )

    args = parser.parse_args()
    
    # Run the main function and exit with its status code
    sys.exit(main(args))