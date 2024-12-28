import os
import sys
import torch
import logging
from datetime import datetime
from attack_config import AttackConfig
from train_attack_model import WCDCGAN, train_attack_model
from attack_evaluation import evaluate_attack
from data_utils import ensure_cpu_tensor, GetLoader
from itertools import product
import traceback
from tqdm import tqdm

def setup_logging():
    """设置日志"""
    try:
        # 创建日志目录
        log_dir = os.path.join(AttackConfig.get_path('LOGS'))
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'attack_log_{timestamp}.txt')
        
        # 配置根日志记录器
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 清除现有的处理器
        logger.handlers.clear()
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 立即输出一条测试消息
        logging.info("Logging system initialized")
        logging.info(f"Log file created at: {log_file}")
        
        return log_file
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        traceback.print_exc()
        raise

def ensure_attack_directories():
    """确保所有攻击相关目录存在"""
    try:
        AttackConfig.ensure_all_directories()
        logging.info("All required directories have been created successfully:")
        for path in [
            AttackConfig.get_attack_model_path(),
            AttackConfig.get_attack_results_path(),
            AttackConfig.get_attack_figures_path(),
            AttackConfig.get_attack_data_path()
        ]:
            logging.info(f"- {path}")
    except Exception as e:
        logging.error(f"Error creating directories: {str(e)}")
        traceback.print_exc()
        raise

def load_target_model(window_size, user_id):
    """加载目标模型"""
    try:
        logging.info(f"Loading target model for User {user_id} with window size {window_size}")
        
        # 添加model目录到系统路径
        model_dir = os.path.join(AttackConfig.BASE_PATH, 'model')
        if model_dir not in sys.path:
            sys.path.append(model_dir)
            
        from mouse_traj_classification import MouseNeuralNetwork
        
        # 获取输入维度
        test_data, _ = load_test_data()
        sequence_length = test_data.shape[2]
        
        model = MouseNeuralNetwork(sequence_length).to(AttackConfig.DEVICE)
        
        # 修改模型权重文件路径
        model_path = os.path.join(
            AttackConfig.BASE_PATH,
            '数据处理代码',
            f'gru-only-adam-user{user_id}_{window_size}-path.pt'
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=AttackConfig.DEVICE)
        model.load_state_dict(checkpoint['model'])
        logging.info(f"Target model loaded successfully from {model_path}")
        
        return model
        
    except Exception as e:
        logging.error(f"Error loading target model: {str(e)}")
        traceback.print_exc()
        raise

def load_test_data():
    """加载测试数据"""
    try:
        logging.info("Loading test data...")
        import pickle
        
        test_path = os.path.join(
            AttackConfig.get_original_data_dir(),
            'X_test_loader.pkl'
        )
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data file not found: {test_path}")
        
        with open(test_path, 'rb') as f:
            test_dataset = pickle.load(f)
        
        logging.info("Processing test dataset...")
        tensors = []
        for tensor, label in tqdm(test_dataset, desc="Processing test data"):
            tensor = ensure_cpu_tensor(tensor)
            label = ensure_cpu_tensor(label)
            tensors.append((tensor, label))
            
        X = torch.stack([t[0] for t in tensors])
        y = torch.stack([t[1] for t in tensors])
        
        logging.info(f"Test data loaded successfully from {test_path}")
        logging.info(f"Test data shape: X={X.shape}, y={y.shape}")
        
        return X, y
        
    except Exception as e:
        logging.error(f"Error loading test data: {str(e)}")
        traceback.print_exc()
        raise

def run_attack_pipeline(window_size, user_id):
    """运行完整的攻击流程"""
    try:
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting attack pipeline for Window Size: {window_size}, User ID: {user_id}")
        logging.info(f"{'='*50}")
        
        # 更新配置
        AttackConfig.WINDOW_SIZE = window_size
        AttackConfig.USER_ID = user_id
        
        # 加载目标模型和测试数据
        target_model = load_target_model(window_size, user_id)
        test_data, test_labels = load_test_data()
        
        # 初始化WC-DCGAN
        sequence_length = test_data.shape[2]
        wc_dcgan = WCDCGAN(sequence_length, AttackConfig.DEVICE)
        logging.info("WC-DCGAN model initialized successfully")
        
        # 训练攻击模型
        logging.info("\nStarting attack model training...")
        trained_wc_dcgan = train_attack_model(
            wc_dcgan,
            test_data,
            test_labels,
            AttackConfig.ATTACK_NUM_EPOCHS,
            AttackConfig.ATTACK_BATCH_SIZE
        )
        
        # 保存训练后的模型
        checkpoint_path = os.path.join(
            AttackConfig.CHECKPOINT_PATH,
            f'wc_dcgan_final_user{user_id}_window{window_size}.pt'
        )
        torch.save({
            'generator_state_dict': trained_wc_dcgan.generator.state_dict(),
            'discriminator_state_dict': trained_wc_dcgan.discriminator.state_dict(),
        }, checkpoint_path)
        logging.info(f"Trained model saved to {checkpoint_path}")
        
        # 生成攻击样本并评估
        logging.info("\nStarting attack evaluation...")
        attack_results = evaluate_attack(
            target_model,
            trained_wc_dcgan,
            test_data,
            test_labels,
            AttackConfig.NUM_ATTACK_SAMPLES
        )
        
        # 保存结果
        results_path = os.path.join(
            AttackConfig.get_attack_results_path(),
            f'attack_results_user{user_id}_window{window_size}.pt'
        )
        torch.save(attack_results, results_path)
        logging.info(f"Attack results saved to {results_path}")
        
        # 输出结果摘要
        logging.info(f"\nAttack Results Summary for User {user_id}, Window Size {window_size}:")
        for key, value in attack_results.items():
            if isinstance(value, (int, float)):
                logging.info(f"{key}: {value:.4f}")
        
        logging.info(f"{'='*50}\n")
        return True
        
    except Exception as e:
        logging.error(f"Error in attack pipeline: {str(e)}")
        traceback.print_exc()
        return False

def main():
    try:
        # 首先创建必要的目录
        ensure_attack_directories()
        
        # 设置日志
        log_file = setup_logging()
        
        logging.info(f"\nStarting attack main program at {datetime.now()}")
        logging.info(f"Using device: {AttackConfig.DEVICE}")
        
        # 定义要测试的窗口大小和用户ID
        window_sizes = [200]
        user_ids = [20]
        
        total_combinations = len(window_sizes) * len(user_ids)
        logging.info(f"\nTotal combinations to test: {total_combinations}")
        logging.info("Window sizes: " + ", ".join(map(str, window_sizes)))
        logging.info("User IDs: " + ", ".join(map(str, user_ids)))
        
        # 运行攻击流程
        successful_attacks = 0
        failed_attacks = 0
        
        for window_size, user_id in tqdm(list(product(window_sizes, user_ids)), 
                                       desc="Processing combinations"):
            try:
                success = run_attack_pipeline(window_size, user_id)
                if success:
                    successful_attacks += 1
                else:
                    failed_attacks += 1
                    logging.error(
                        f"Attack pipeline failed for Window Size: {window_size}, "
                        f"User ID: {user_id}"
                    )
            except Exception as e:
                failed_attacks += 1
                logging.error(
                    f"Error processing Window Size: {window_size}, "
                    f"User ID: {user_id}"
                )
                logging.error(str(e))
                traceback.print_exc()
                continue
        
        # 输出最终统计
        logging.info("\nAttack Pipeline Completed!")
        logging.info(f"Successful attacks: {successful_attacks}")
        logging.info(f"Failed attacks: {failed_attacks}")
        logging.info(f"Success rate: {successful_attacks/total_combinations:.2%}")
        logging.info(f"Log file location: {log_file}")
        
    except Exception as e:
        logging.error("Critical error in main function")
        logging.error(str(e))
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()