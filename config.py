import argparse

def set_args():
    parser = argparse.ArgumentParser('--CoSENT进行相似性判断')
    # ./data/ATEC/ATEC.train.data
    # ./data/BQ/BQ.train.data
    # ./data/LCQMC/LCQMC.train.data
    # ./data/PAWSX/PAWSX.train.data
    # ./data/STS-B/STS-B.train.data 
    parser.add_argument('--train_data', default='./TASK-3-DATA/train', type=str, help='训练数据集')
    parser.add_argument('--test_data', default='./TASK-3-DATA/test', type=str, help='测试数据集')

    parser.add_argument('--pretrained_model_path', default='output-2-new-bert-base-uncased/task-2-output-1-bert-base-uncased-task-1-bert-base-uncased-2023-03-23_15-02-08-2023-04-02_13-25-51', type=str, help='预训练模型的路径')
    parser.add_argument('--output_dir', default='./output-3-new-(2->3)-last-three-mean', type=str, help='模型输出')
    parser.add_argument('--num_train_epochs', default=3, type=int, help='训练几轮')
    parser.add_argument('--train_batch_size', default=32, type=int, help='训练批次大小')
    parser.add_argument('--val_batch_size', default=32, type=int, help='验证批次大小')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积累几次更新')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='学习率大小')
    parser.add_argument('--seed', default=43, type=int, help='随机种子')
    parser.add_argument('--device', default='cuda:1', type=str, help='device name "cuda:#"')
    return parser.parse_args()