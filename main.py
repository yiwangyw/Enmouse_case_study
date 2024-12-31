# main.py
import os
import sys
from config import Config
from itertools import product

def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [
        Config.get_data_dir(),
        Config.get_model_path(),
        Config.get_results_path(),
        Config.get_figures_path()
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_pipeline(window_size, user_id):
    """Run the complete pipeline for a specific window size and user ID"""
    # Update configuration
    Config.WINDOW_SIZE = window_size
    Config.USER_ID = user_id
    
    print(f"\nRunning pipeline with Window Size: {window_size}, User ID: {user_id}")
    print(f"Using configuration:")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Base Path: {Config.BASE_PATH}")
    
    try:
        # Run training
        print("\nStarting training phase...")
        import traincode
        traincode.run_training()
        
        # Run GRU training
        print("\nStarting GRU training phase...")
        import gru
        gru.run_gru_training()
        
        # Run testing
        print("\nStarting testing phase...")
        import testcode
        testcode.run_testing()
        
        print(f"\nPipeline completed for Window Size: {window_size}, User ID: {user_id}")
        
    except Exception as e:
        print(f"\nError occurred during execution with Window Size: {window_size}, User ID: {user_id}")
        print(f"Error: {str(e)}")
        raise e

def main():
    # Ensure all necessary directories exist
    ensure_directories()
    
    print("Starting pipeline execution...")
    
    # Define the different window sizes and user IDs to test
    window_sizes =  list(range(10, 301, 10))  
    # window_sizes = [10]
    user_ids = [20]      
    # Run pipeline for each combination
    for window_size, user_id in product(window_sizes, user_ids):
        try:
            run_pipeline(window_size, user_id)
        except Exception as e:
            print(f"\nSkipping remaining combinations due to error in Window Size: {window_size}, User ID: {user_id}")
            raise e
    
    print("\nAll pipeline executions completed successfully!")

if __name__ == "__main__":
    main()