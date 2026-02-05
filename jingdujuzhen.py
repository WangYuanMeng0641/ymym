import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re
from sklearn.metrics import confusion_matrix
from collections import Counter
import itertools
from tqdm import tqdm
import pickle
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader


# 从main.py中复制必要的类和函数
class SequenceDataset(Dataset):
    def __init__(self, sequences, cksnap_features, kmer_features, adj_matrices, labels):
        self.sequences = sequences
        self.cksnap_features = cksnap_features
        self.kmer_features = kmer_features
        self.adj_matrices = adj_matrices
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_id = self.sequences[idx]
        cksnap = torch.tensor(self.cksnap_features[seq_id], dtype=torch.float32)
        kmer = torch.tensor(self.kmer_features[seq_id], dtype=torch.float32)
        adj_matrix = torch.tensor(self.adj_matrices[seq_id], dtype=torch.float32)
        label = torch.tensor(self.labels[seq_id], dtype=torch.float32)
        return cksnap, kmer, adj_matrix, label


def read_nucleotide_sequences(file):
    print(f"正在读取FASTA文件: {file}")
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    fasta_sequences = []
    for fasta in tqdm(records, desc="解析FASTA记录"):
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACGTU-]', '-', ''.join(array[1:]).upper())
        sequence = re.sub('U', 'T', sequence)
        fasta_sequences.append([header, sequence])
    print(f"成功读取 {len(fasta_sequences)} 条序列")
    return fasta_sequences


def get_min_sequence_length(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen


def CKSNAP(fastas, gap=5, **kw):
    print(f"计算CKSNAP特征 (gap={gap})...")
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if get_min_sequence_length(fastas) < gap + 2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    AA = kw['order'] if kw['order'] != None else 'ACGT'

    encodings = {}
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    for i in tqdm(fastas, desc="计算CKSNAP"):
        seq_key = i[0]
        sequence = i[1]
        code = []
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum if sum > 0 else 0)
        encodings[seq_key] = code
    print(f"CKSNAP特征计算完成，共处理 {len(encodings)} 条序列")
    return encodings


def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer


def Kmer(fastas, k=6, type="DNA", upto=True, normalize=True, **kw):
    print(f"计算Kmer特征 (k={k}, upto={upto}, normalize={normalize})...")
    encoding = {}
    NA = 'ACGT'
    if type in ("DNA", 'RNA'):
        NA = 'ACGT'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'

    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0

    # 生成头部
    header = []
    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                header.append(''.join(kmer))
    else:
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))

    for i in tqdm(fastas, desc="计算Kmer"):
        seq_key = i[0]
        sequence = re.sub('-', '', i[1])
        count = Counter()

        if upto == True:
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
        else:
            kmers = kmerArray(sequence, k)
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)

        code = []
        for mer in header:
            code.append(count.get(mer, 0))
        encoding[seq_key] = code
    print(f"Kmer特征计算完成，共处理 {len(encoding)} 条序列")
    return encoding


def build_de_bruijn_adjacency(sequence, k=3):
    """构建有向德布鲁因图邻接矩阵（返回原始二值矩阵）"""
    nucleotides = ['A', 'C', 'G', 'T']
    kmers = [''.join(combo) for combo in itertools.product(nucleotides, repeat=k)]
    kmer_to_idx = {kmer: idx for idx, kmer in enumerate(kmers)}
    n_kmers = len(kmers)

    # 初始化邻接矩阵
    adj_matrix = np.zeros((n_kmers, n_kmers), dtype=np.float32)

    # 提取序列中的k-mer
    seq_kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if all(n in nucleotides for n in kmer):
            seq_kmers.append(kmer)

    # 构建有向邻接矩阵
    for i in range(len(seq_kmers) - 1):
        kmer1 = seq_kmers[i]
        kmer2 = seq_kmers[i + 1]

        if kmer1 in kmer_to_idx and kmer2 in kmer_to_idx:
            idx1 = kmer_to_idx[kmer1]
            idx2 = kmer_to_idx[kmer2]
            adj_matrix[idx1, idx2] = 1.0

    return adj_matrix


def cache_to_file(filename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if os.path.exists(filename):
                print(f"从缓存加载数据: {filename}")
                with open(filename, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"计算数据并缓存到: {filename}")
                result = func(*args, **kwargs)
                with open(filename, 'wb') as f:
                    pickle.dump(result, f)
                print(f"数据已缓存到: {filename}")
                return result

        return wrapper

    return decorator


# 现在导入模型相关的模块
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import SequenceModel


def generate_precision_confusion_matrix(model, dataloader, device, label_columns, output_path):
    """生成每个亚细胞的精度混淆矩阵，排列成3x3格式"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for c, k, a, y in dataloader:
            c, k, a = [i.to(device) for i in (c, k, a)]
            outputs = model(c, k, a)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # 为每个亚细胞位置生成二分类混淆矩阵
    n_classes = len(label_columns)

    # 创建更大的图形
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.ravel()

    for i in range(n_classes):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]

        # 计算二分类混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 计算精度矩阵（按列归一化）
        precision_matrix = cm.astype('float') / (cm.sum(axis=0) + 1e-8)

        # 绘制热力图 - 增大数字字体
        sns.heatmap(precision_matrix,
                    annot=True,
                    fmt='.3f',
                    cmap='Blues',
                    cbar=True,
                    square=True,
                    ax=axes[i],
                    vmin=0, vmax=1,
                    annot_kws={"size": 14},  # 增大数字字体
                    cbar_kws={'shrink': 0.8})

        # 设置轴标签 - 明确标注真实标签和预测标签
        axes[i].set_xlabel('Predicted label', fontsize=12)
        axes[i].set_ylabel('True label', fontsize=12)

        # 设置刻度标签
        axes[i].set_xticklabels(['N', 'P'], fontsize=11)
        axes[i].set_yticklabels(['N', 'P'], fontsize=11)

        # 在子图上方添加亚细胞名称
        axes[i].set_title(label_columns[i],
                          fontsize=14,
                          fontweight='bold',
                          pad=15)

    # 隐藏多余的子图
    for i in range(n_classes, 9):
        axes[i].set_visible(False)

    # 调整子图间距
    plt.subplots_adjust(
        left=0.05,
        bottom=0.08,
        right=0.95,
        top=0.95,
        wspace=0.3,
        hspace=0.3
    )

    # 保存图像
    plt.savefig(os.path.join(output_path, 'precision_confusion_matrix_3x3.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'precision_confusion_matrix_3x3.pdf'),
                bbox_inches='tight')
    plt.show()

    return all_preds, all_targets
def generate_combined_confusion_matrix(model, dataloader, device, label_columns, output_path):
    """生成组合的精度混淆矩阵（类似文献图7）"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for c, k, a, y in dataloader:
            c, k, a = [i.to(device) for i in (c, k, a)]
            outputs = model(c, k, a)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # 创建大的混淆矩阵
    n_classes = len(label_columns)
    combined_cm = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            # 计算当预测为j时，真实为i的比例（精度视角）
            pred_j = all_preds[:, j] == 1
            if np.sum(pred_j) > 0:
                combined_cm[i, j] = np.sum((all_targets[pred_j, i] == 1)) / np.sum(pred_j)

    # 绘制组合混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(combined_cm,
                annot=True,
                fmt='.3f',
                cmap='Blues',
                cbar=True,
                square=True,
                xticklabels=label_columns,
                yticklabels=label_columns,
                vmin=0, vmax=1)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'combined_precision_confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'combined_precision_confusion_matrix.pdf'),
                bbox_inches='tight')
    plt.show()
    return combined_cm, all_preds, all_targets


def run_independent_test_with_matrices():
    """使用第二折模型在独立测试集上运行并生成精度矩阵"""
    # 设置参数
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32

    os.makedirs(output_path, exist_ok=True)

    # 读取独立标签和序列
    ind_csv_path = os.path.join(base_dir, "dataset", "independent.csv")
    ind_fasta_path = os.path.join(base_dir, "dataset", "independent_seqs")

    # 读取独立标签
    ind_label_df = pd.read_csv(ind_csv_path)
    label_columns = ind_label_df.columns[1:]
    ind_labels = {row[0]: row[1:].values.astype(np.float32)
                  for _, row in ind_label_df.iterrows()}

    # 读取独立序列
    ind_fasta = read_nucleotide_sequences(ind_fasta_path)

    # 特征提取（使用缓存）
    @cache_to_file(os.path.join(output_path, 'ind_cksnap.pkl'))
    def get_ind_cksnap():
        return CKSNAP(ind_fasta, gap=5, order='ACGT')

    @cache_to_file(os.path.join(output_path, 'ind_kmer.pkl'))
    def get_ind_kmer():
        return Kmer(ind_fasta, k=6, type='RNA', upto=True, normalize=True, order='ACGT')

    @cache_to_file(os.path.join(output_path, 'ind_adj.pkl'))
    def get_ind_adj():
        return {i[0]: build_de_bruijn_adjacency(i[1], k=3) for i in tqdm(ind_fasta, desc='构建邻接矩阵')}

    ind_ck = get_ind_cksnap()
    ind_km = get_ind_kmer()
    ind_adj = get_ind_adj()

    # 对齐序列
    ind_seqs = [i[0] for i in ind_fasta
                if i[0] in ind_labels and i[0] in ind_ck and i[0] in ind_km and i[0] in ind_adj]

    print(f"独立测试集序列数量: {len(ind_seqs)}")
    print(f"亚细胞位置标签: {label_columns.tolist()}")

    # 创建数据集
    ind_set = SequenceDataset(ind_seqs, ind_ck, ind_km, ind_adj, ind_labels)
    ind_loader = DataLoader(ind_set, batch_size=batch_size, shuffle=False)

    # 加载第二折模型
    model_path = os.path.join(output_path, 'best_model_fold_2.pt')
    model = SequenceModel(n_classes=len(label_columns)).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载第二折模型: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        # 列出所有可用的模型文件
        available_models = [f for f in os.listdir(output_path) if f.endswith('.pt')]
        print(f"可用的模型文件: {available_models}")
        if available_models:
            model_path = os.path.join(output_path, available_models[0])
            print(f"使用第一个可用模型: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("没有找到任何模型文件，请先运行训练。")
            return

    # 生成3x3格式的精度矩阵
    print("生成3x3格式的精度混淆矩阵...")
    preds, targets = generate_precision_confusion_matrix(
        model, ind_loader, device, label_columns, output_path
    )

    # 生成组合精度矩阵（类似文献图7）
    print("生成组合精度混淆矩阵...")
    combined_cm, _, _ = generate_combined_confusion_matrix(
        model, ind_loader, device, label_columns, output_path
    )

    # 保存详细结果
    results = {
        'predictions': preds,
        'targets': targets,
        'combined_confusion_matrix': combined_cm,
        'label_columns': label_columns.tolist()
    }

    with open(os.path.join(output_path, 'precision_matrix_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print("精度矩阵生成完成！")
    print(f"结果保存在: {output_path}")


if __name__ == "__main__":
    run_independent_test_with_matrices()