# config.py
import torch
class Config:
    # Global settings
    USER_ID = 9
    WINDOW_SIZE = 100
    BASE_PATH = "C:/Users/Admin/数据处理代码/"
    BATCH_SIZE = 128
    RANDOM_SEED = 3407
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Training settings
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.1
    
    # File paths
    def get_data_dir():
        return f"{Config.BASE_PATH}datacsv/processed_data_user{Config.USER_ID}_len{Config.WINDOW_SIZE}"
    
    def get_model_path():
        return f"{Config.BASE_PATH}model"
    
    def get_results_path():
        return f"{Config.BASE_PATH}results"
    
    def get_figures_path():
        return f"{Config.BASE_PATH}figures"