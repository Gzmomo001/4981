# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import json

# Matplotlib
import matplotlib.pyplot as plt

# Optuna
import optuna


from progress.bar import FillingSquaresBar
import time

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

def same_seed(seed):
    """
    Fixes random number generator seeds for reproducibility.

    This function ensures that experiments are reproducible by setting the random seeds of various libraries to the same value.
    It covers both CPU and GPU operations, as well as the cuDNN backend to ensure consistency in deep learning models.

    Parameters:
    - seed (int): The seed value to set for the random number generators.

    Returns:
    None
    """
    # Set the cuDNN backend to deterministic mode for consistent results
    torch.backends.cudnn.deterministic = True
    # Disable cuDNN benchmark mode to fix the convolution algorithm
    torch.backends.cudnn.benchmark = False
    # Set NumPy's random seed for reproducibility of CPU operations
    np.random.seed(seed)
    # Set PyTorch's random seed for reproducibility of CPU operations
    torch.manual_seed(seed)
    # If a GPU is available, set the random seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    '''
    Split provided training data into training set and validation set

    Parameters:
    data_set (Dataset): The original dataset to be split
    valid_ratio (float): The proportion of the validation set in the original dataset
    seed (int): Random seed to ensure reproducibility of the split results

    Returns:
    np.array: Training set and validation set converted to numpy array format
    '''
    # Calculate the size of the validation set
    valid_set_size = int(valid_ratio * len(data_set))
    # Calculate the size of the training set
    train_set_size = len(data_set) - valid_set_size
    # Split the dataset into training and validation sets, using a random seed to ensure reproducibility of the split results
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    # Return the training and validation sets converted to numpy array format
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    # 使用tqdm包装测试数据加载器，以显示加载进度
    for x in tqdm(test_loader):
        # 将输入数据移动到指定设备（如GPU）
        x = x.to(device)
        # 在没有梯度计算的上下文中进行预测，以减少计算资源消耗
        with torch.no_grad():
            # 使用模型进行预测
            pred = model(x)
            # 将预测结果从GPU移动到CPU，并添加到预测结果列表中
            preds.append(pred.detach().cpu())
    # 将所有预测结果合并成一个张量，并转换为numpy数组
    preds = torch.cat(preds, dim=0).numpy()
    # 返回最终的预测结果数组
    return preds


class CovidDataset(Dataset):
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, config['layer'][0]),
            nn.ReLU(),
            nn.Linear(config['layer'][0], config['layer'][1]),
            nn.ReLU(),
            nn.Linear(config['layer'][1], 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x

from sklearn.feature_selection import SelectKBest, f_regression

def select_feat(train_data, valid_data, test_data, no_select_all=True):
    '''Selects useful features to perform regression'''
    global config
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if not no_select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        # Feature selection
        k = config['k']
        selector = SelectKBest(score_func=f_regression, k=k)
        result = selector.fit(train_data[:, :-1], train_data[:,-1])
        idx = np.argsort(result.scores_)[::-1]
        feat_idx = list(np.sort(idx[:k]))

    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid

def trainer(train_loader, valid_loader, model, config, device):
    # Define your loss function, do not modify this.
    criterion = nn.MSELoss(reduction='mean')

    # Define your optimization algorithm with default values for missing keys.
    optim_config = {
        'optim': 'SGD',
        'no_momentum': False,
        'learning_rate': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'n_epochs': 100,
        'early_stop': 10,
        'save_path': './models/best_model.pth',
        'no_tensorboard': False
    }
    optim_config.update(config)

    if optim_config['optim'] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optim_config['learning_rate'],
            momentum=0 if optim_config['no_momentum'] else optim_config['momentum'],
            weight_decay=optim_config['weight_decay']
        )
    elif optim_config['optim'] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_config['learning_rate'],
            weight_decay=optim_config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optim_config['optim']}")

    writer = SummaryWriter() if not optim_config['no_tensorboard'] else None

    try:
        os.makedirs('./models', exist_ok=True)  # Create directory of saving models.
    except Exception as e:
        print(f"Failed to create models directory: {e}")

    n_epochs, step, early_stop_count = (
        optim_config['n_epochs'],  0, 0
    )

    best_loss = math.inf
    valid_scores = []
    all_valid_losses = []  # 新增：保存每个epoch的验证损失
    bar = FillingSquaresBar('Stop count:', max=optim_config['early_stop'])

    for epoch in range(n_epochs):

        model.train()  # Set your model to train mode.
        loss_record = []

        for x, y in train_loader:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

        mean_train_loss = sum(loss_record) / len(loss_record)

        model.eval()  # Set your model to evaluation mode.
        dev_loss_record = []
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                dev_loss_record.append(loss.item())

        mean_valid_loss = sum(dev_loss_record) / len(dev_loss_record)

        if writer is not None and epoch % 10 == 0:
            writer.add_scalar('Loss/train', mean_train_loss, step)
            writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            valid_scores.append(best_loss)
            torch.save(model.state_dict(), optim_config['save_path'])  # Save your best model
            tqdm.write(f"Epoch {epoch+1}, Loss: {best_loss:.4f}")
            early_stop_count = 0
            bar = FillingSquaresBar('Stop count:', max=optim_config['early_stop'])
        else:
            early_stop_count += 1
            bar.next()

        if early_stop_count >= optim_config['early_stop']:
            print(f'Best loss {best_loss:.3f}. Model is not improving, so we halt the training session.')
            break

    loss_record_file = f"best_loss_record.json"
    with open(loss_record_file, 'w') as f:
        json.dump(valid_scores, f)  # 修改：保存所有epoch的验证损失

    return best_loss



def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
    'k': 16,  # Select k features
    'layer': [64, 16],
    'optim': 'Adam',
    'momentum': 0.7,
    'valid_ratio': 0.1,  # validation_size = train_size * valid_ratio
    'n_epochs': 10000,  # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'early_stop': 1000,  # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt',  # Your model will be saved here.
    'no_select_all': True,  # Whether to use all features.
    'no_momentum': False,  # Whether to use momentum
    'no_normal': False,  # Whether to normalize data
    'no_k_cross': False,  # Whether to use K-fold cross validation
    'no_save': False,  # Whether to save model parameters
    'no_tensorboard': False,  # Whether to write tensorboard
}

# 设置 k-fold 中的 k，这里是根据 valid_ratio 设定的
k = int(1 / config['valid_ratio'])

# Set seed for reproducibility
same_seed(config['seed'])

training_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values

num_valid_samples = len(training_data) // k
np.random.shuffle(training_data)



def objective(trial):
    """
    目标函数，用于优化超参数。

    参数:
    trial (optuna.trial.Trial): 一个试验对象，用于选择超参数。

    返回:
    float: 如果是优化过程中的试验，则返回验证分数的平均值；
           如果不是优化过程中的试验，则返回测试数据和测试数据加载器。
    """
    if trial != None:
        print('\nNew trial here')
        # 定义需要调优的超参数空间
        config['learning_rate'] = trial.suggest_float('lr', 0.5e-3, 1.5e-3)
        #config['batch_size'] = trial.suggest_categorical('batch_size', [256, 128, 192])
        # config['k'] = trial.suggest_categorical('k_feats', [16, 32, 48, 56, 64, 80])
        #config['layer'][0] = trial.suggest_categorical('k_feats', [16, 32, 48, 64])
        config['layer'][1] = trial.suggest_categorical('layer2', [8, 16])
        # 选择优化器
        # config['optim'] = trial.suggest_categorical('optim', ['SGD', 'Adam'])
        # 测试不同的方法是否有效果
        # config['no_momentum'] = trial.suggest_categorical('no_momentum', [True, False])
        # config['no_normal'] = trial.suggest_categorical('no_normal', [True, False])
        # config['no_k_cross'] = trial.suggest_categorical('no_k_cross', [True, False])
    # 打印所需的超参数
    print(f'''hyper-parameter: 
        optimizer: {config['optim']},
        lr: {config['learning_rate']}, 
        batch_size: {config['batch_size']}, 
        k: {config['k']}, 
        layer: {config['layer']},
        no_momentum: {config['no_momentum']}, 
        no_normal: {config['no_normal']}, 
        no_k_cross: {config['no_k_cross']}
        ''')

    global valid_scores
    # 每次 trial 初始化 valid_scores，可以不初始化，通过 trial * k + fold 来访问当前 trial 的 valid_score，
    # 这样可以让 trainer() 保存 trials 中最好的模型参数，但这并不意味着该参数对应的 k-fold validation loss 最低。
    valid_scores = []

    for fold in range(k):
        # Data split
        valid_data = training_data[num_valid_samples * fold:
                                   num_valid_samples * (fold + 1)]
        train_data = np.concatenate((
            training_data[:num_valid_samples * fold],
            training_data[num_valid_samples * (fold + 1):]))

        # Normalization
        if not config['no_normal']:
            train_mean = np.mean(train_data[:, 35:-1], axis=0)  # 前 35 列为 one-hot vector
            train_std = np.std(train_data[:, 35:-1], axis=0)
            train_data[:, 35:-1] -= train_mean
            train_data[:, 35:-1] /= train_std
            valid_data[:, 35:-1] -= train_mean
            valid_data[:, 35:-1] /= train_std
            test_data[:, 35:] -= train_mean
            test_data[:, 35:] /= train_std

        x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data,
                                                                 config['no_select_all'])

        train_dataset, valid_dataset, test_dataset = CovidDataset(x_train, y_train), \
            CovidDataset(x_valid, y_valid), \
            CovidDataset(x_test)

        # Pytorch data loader loads pytorch dataset into batches.
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

        model = My_Model(input_dim=x_train.shape[1]).to(
            device)  # put your model and data on the same computation device.
        valid_score = trainer(train_loader, valid_loader, model, config, device)
        valid_scores.append(valid_score)

        # 重置模型参数
        model = My_Model(input_dim=x_train.shape[1]).to(device)

        if not config['no_k_cross']:
            break

        if valid_score > 1:
            print(f'在第{fold + 1}折上欠拟合')  # 提前终止，减少计算资源
            break

    print(f'valid_scores: {np.average(valid_scores)}')

    if trial != None:
        return np.average(valid_scores)
    else:
        return x_test, test_loader



AUTO_TUNE_PARAM = False  # Whether to tune parameters automatically

if AUTO_TUNE_PARAM:
    # 使用Optuna库进行超参数搜索
    n_trials = 10  # 设置试验数量
    print(f'AUTO_TUNE_PARAM: {AUTO_TUNE_PARAM}\nn_trials: {n_trials}')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs=1)  # 启用并行 trial

    # 输出最优的超参数组合和性能指标
    print('Best hyperparameters: {}'.format(study.best_params))
    print('Best performance: {:.4f}'.format(study.best_value))
else:
    # 注意，只有非自动调参时才进行了predict，节省一下计算资源
    print(f'You could set AUTO_TUNE_PARAM True to tune parameters automatically.\nAUTO_TUNE_PARAM: {AUTO_TUNE_PARAM}')
    x_test, test_loader = (
        objective(None))
    model = My_Model(input_dim=x_test.shape[1]).to(device)
    model.load_state_dict(torch.load(config['save_path']))
    preds = predict(test_loader, model, device)
    save_pred(preds, 'submission.csv')