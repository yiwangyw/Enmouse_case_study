import torch
import os

class AttackConfig:
    # 基础路径设置
    BASE_PATH = os.path.abspath("C:/Users/Admin/数据处理代码")
    
    # 设备配置
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 攻击模型配置
    ATTACK_BATCH_SIZE = 128
    ATTACK_NUM_EPOCHS = 300
    ATTACK_LEARNING_RATE = 0.0001
    
    # WC-DCGAN 配置
    LATENT_DIM = 100
    N_CRITIC = 5
    GP_WEIGHT = 10.0
    BETA1 = 0.5
    BETA2 = 0.9
    
    # 评估配置
    NUM_ATTACK_SAMPLES = 2000
    EVAL_FREQUENCY = 10
    SAVE_FREQUENCY = 10
    
    # 模型参数
    WINDOW_SIZE = 100
    USER_ID = 20
    
    # 目录路径配置
    DIRS = {
        'MODEL': 'attack_models',
        'RESULTS': 'attack_results',
        'FIGURES': 'attack_figures',
        'DATA': 'attack_data',
        'LOGS': 'attack_logs',
        'CHECKPOINTS': 'checkpoints'
    }
    
    # 设置CHECKPOINT_PATH作为类属性
    CHECKPOINT_PATH = os.path.join(BASE_PATH, DIRS['CHECKPOINTS'])
    
    @classmethod
    def get_path(cls, dir_key):
        """获取完整路径"""
        path = os.path.join(cls.BASE_PATH, cls.DIRS[dir_key])
        os.makedirs(path, exist_ok=True)
        return path
    
    @classmethod
    def get_attack_model_path(cls):
        """获取攻击模型路径"""
        return cls.get_path('MODEL')
    
    @classmethod
    def get_attack_results_path(cls):
        """获取结果保存路径"""
        return cls.get_path('RESULTS')
    
    @classmethod
    def get_attack_figures_path(cls):
        """获取图形保存路径"""
        return cls.get_path('FIGURES')
    
    @classmethod
    def get_attack_data_path(cls):
        """获取数据保存路径"""
        return cls.get_path('DATA')
    
    @classmethod
    def get_log_file(cls):
        """获取日志目录路径"""
        return cls.get_path('LOGS')
    
    @classmethod
    def get_checkpoint_path(cls):
        """获取检查点保存路径"""
        return cls.get_path('CHECKPOINTS')
    
    @classmethod
    def get_original_data_dir(cls):
        """获取原始数据目录"""
        return os.path.join(
            cls.BASE_PATH,
            f"datacsv/processed_data_user{cls.USER_ID}_len{cls.WINDOW_SIZE}"
        )
    
    @classmethod
    def get_original_model_path(cls):
        """获取原始模型路径"""
        return os.path.join(cls.BASE_PATH, "model")
    
    @classmethod
    def ensure_all_directories(cls):
        """确保所有必要的目录都存在"""
        try:
            # 确保BASE_PATH存在
            if not os.path.exists(cls.BASE_PATH):
                raise Exception(f"Base path does not exist: {cls.BASE_PATH}")
            
            # 创建所有子目录
            for dir_key in cls.DIRS:
                dir_path = cls.get_path(dir_key)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    print(f"Created directory: {dir_path}")
                else:
                    print(f"Directory already exists: {dir_path}")
            
        except Exception as e:
            print(f"Error in ensure_all_directories: {str(e)}")
            raise
    
    @classmethod
    def print_config(cls):
        """打印当前配置信息"""
        print("\nAttack Configuration:")
        print(f"Base Path: {cls.BASE_PATH}")
        print(f"Device: {cls.DEVICE}")
        print("\nTraining Parameters:")
        print(f"Batch Size: {cls.ATTACK_BATCH_SIZE}")
        print(f"Number of Epochs: {cls.ATTACK_NUM_EPOCHS}")
        print(f"Learning Rate: {cls.ATTACK_LEARNING_RATE}")
        print("\nModel Parameters:")
        print(f"Window Size: {cls.WINDOW_SIZE}")
        print(f"User ID: {cls.USER_ID}")
        print("\nDirectory Structure:")
        for dir_key, dir_name in cls.DIRS.items():
            print(f"{dir_key}: {cls.get_path(dir_key)}")

if __name__ == "__main__":
    # 测试配置
    print("Testing AttackConfig...")
    AttackConfig.print_config()
    print("\nTesting directory creation...")
    AttackConfig.ensure_all_directories()