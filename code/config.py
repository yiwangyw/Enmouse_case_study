# config.py
import torch
class Config:
    # Global settings
    USER_ID = 9
    WINDOW_SIZE = 1
    BASE_PATH = "/data/yanbo.wang/CCS2025/Enmouse_case_study/new/"
    BATCH_SIZE = 256
    RANDOM_SEED = 123  #0 7 42 123 3407  ��������
    Resnet_NUM_EPOCHS = 50
    GRU_NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    
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

    def get_Resnetloss_path():
        return f"{Config.BASE_PATH}Resnetloss"

    def get_Gruloss_path():
        return f"{Config.BASE_PATH}Gruloss"