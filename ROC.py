import re, os, sys
import torch
import argparse
import random
import numpy as np
from Bio import SeqIO
import itertools
from collections import Counter
import pandas as pd
import pickle
from sklearn.metrics import jaccard_score, label_ranking_average_precision_score, coverage_error, hamming_loss, \
    label_ranking_loss, average_precision_score, roc_curve, auc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 设置基础目录路径
base_dir = os.path.dirname(os.path.abspath(__file__))


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)


# 清理内存函数
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== AUC计算和绘图功能 ====================
def calculate_auc_per_class(model, dataloader, device, label_columns):
    """
    计算每个类别的AUC值
    """
    model.eval()
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for cksnap, kmer, adj, labels in tqdm(dataloader, desc="计算AUC"):
            cksnap = cksnap.to(device)
            kmer = kmer.to(device)
            adj = adj.to(device)

            outputs = model(cksnap, kmer, adj)
            probabilities = torch.sigmoid(outputs)

            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 合并所有batch的结果
    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probabilities)

    # 计算每个类别的AUC
    auc_scores = {}
    fpr_dict = {}
    tpr_dict = {}

    for i, class_name in enumerate(label_columns):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_name] = roc_auc
        fpr_dict[class_name] = fpr
        tpr_dict[class_name] = tpr

    return auc_scores, fpr_dict, tpr_dict, y_true, y_prob


def plot_auc_curves(auc_scores, fpr_dict, tpr_dict, label_columns, output_path, figsize=(14, 10)):
    """
    绘制所有亚细胞定位的AUC曲线图 - 使用更鲜明的颜色，保持原有线条宽度
    """
    # 设置样式
    plt.style.use('default')

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 使用鲜明、高对比度的颜色方案
    colors = [
        '#E41A1C',  # 红色
        '#377EB8',  # 蓝色
        '#4DAF4A',  # 绿色
        '#984EA3',  # 紫色
        '#FF7F00',  # 橙色
        '#FFFF33',  # 黄色
        '#A65628',  # 棕色
        '#F781BF',  # 粉色
        '#66C2A5',  # 蓝绿色
        '#FC8D62',  # 橙红色
        '#8DA0CB',  # 淡紫色
        '#E78AC3',  # 浅粉色
    ]

    # 为每个类别绘制ROC曲线
    for i, class_name in enumerate(label_columns):
        if class_name in fpr_dict and class_name in tpr_dict:
            fpr = fpr_dict[class_name]
            tpr = tpr_dict[class_name]
            roc_auc = auc_scores[class_name]

            color = colors[i % len(colors)]

            # 绘制ROC曲线 - 保持原有线条宽度，使用鲜明颜色
            ax.plot(fpr, tpr,
                    color=color,
                    lw=2.5,  # 保持原有线条宽度
                    alpha=0.9,
                    label=f'{class_name} (AUC = {roc_auc:.4f})')

    # 绘制对角线（随机分类器）
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, lw=2, label='Random Classifier (AUC = 0.5000)')

    # 设置图形属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves for Subcellular Localization Prediction',
                 fontsize=16, fontweight='bold', pad=20)

    # 设置图例 - 放在图内右下角，稍微调整位置
    ax.legend(bbox_to_anchor=(0.98, 0.02),
              loc='lower right',
              fontsize=10,
              frameon=True,
              fancybox=True,
              shadow=True,
              framealpha=0.9)
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

    # 调整布局，为图例留出空间
    plt.tight_layout()

    # 保存图片
    plt.savefig(os.path.join(output_path, 'auc_curves.png'),
                dpi=600,
                bbox_inches='tight',
                facecolor='white')
    plt.savefig(os.path.join(output_path, 'auc_curves.pdf'),
                bbox_inches='tight',
                facecolor='white')
    plt.close()

    print(f"AUC曲线图已保存到: {os.path.join(output_path, 'auc_curves.png')}")


def print_auc_statistics(auc_scores):
    """
    打印AUC统计信息
    """
    print("\n" + "=" * 80)
    print("各亚细胞定位AUC统计信息")
    print("=" * 80)

    # 按AUC值排序
    sorted_auc = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)

    for i, (class_name, auc_value) in enumerate(sorted_auc, 1):
        performance_level = "优秀" if auc_value >= 0.9 else "很好" if auc_value >= 0.8 else "一般" if auc_value >= 0.7 else "较差"
        print(f"{i:2d}. {class_name:<20} AUC = {auc_value:.4f} ({performance_level})")

    # 计算统计量
    auc_values = list(auc_scores.values())
    mean_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)
    max_auc = max(auc_values)
    min_auc = min(auc_values)

    print("\n统计摘要:")
    print(f"  平均AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  最高AUC: {max_auc:.4f} ({sorted_auc[0][0]})")
    print(f"  最低AUC: {min_auc:.4f} ({sorted_auc[-1][0]})")
    print(f"  AUC > 0.9的类别: {sum(1 for x in auc_values if x >= 0.9)}个")
    print(f"  AUC > 0.8的类别: {sum(1 for x in auc_values if x >= 0.8)}个")

    return sorted_auc


def compute_multilabel_metrics_literature(y_true, y_logits):
    """
    计算多标签评估指标
    """
    # 将logits转换为概率
    y_prob = torch.sigmoid(y_logits).numpy()
    y_true_np = y_true.numpy()

    # 使用阈值0.5进行二值化预测
    y_pred = (y_prob > 0.5).astype(int)

    aiming = np.mean([np.sum(y_true_np[i] * y_pred[i]) / np.sum(y_pred[i])
                      if np.sum(y_pred[i]) > 0 else 0 for i in range(len(y_true_np))])

    coverage = np.mean([np.sum(y_true_np[i] * y_pred[i]) / np.sum(y_true_np[i])
                            if np.sum(y_true_np[i]) > 0 else 0 for i in range(len(y_true_np))])

    accuracy = jaccard_score(y_true_np, y_pred, average='samples')

    absolute_true = np.mean([np.array_equal(y_true_np[i], y_pred[i]) for i in range(len(y_true_np))])

    absolute_false = hamming_loss(y_true_np, y_pred)

    return {
        'aiming': aiming,
        'coverage': coverage,
        'accuracy': accuracy,
        'absolute_true': absolute_true,
        'absolute_false': absolute_false
    }


# ==================== 模型定义（保持不变）====================
def normalize_adjacency(adj):
    B, N, _ = adj.shape
    I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
    A_hat = adj + I
    D = A_hat.sum(dim=2)
    D_safe = torch.where(D == 0, torch.ones_like(D), D)
    D_inv = torch.diag_embed(1.0 / D_safe)
    adj_norm = D_inv @ A_hat
    return adj_norm


def normalize_node_features(x):
    return x / (x.sum(dim=2, keepdim=True) + 1e-8)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):
        adj_norm = normalize_adjacency(adj)
        x_norm = normalize_node_features(x)
        x = torch.bmm(adj_norm, x_norm)
        x = self.linear(x)
        batch_size, num_nodes, out_features = x.shape
        x = x.view(-1, out_features)
        x = self.bn(x)
        x = x.view(batch_size, num_nodes, out_features)
        return F.relu(x)


class GraphProcessingModule(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim=12):
        super(GraphProcessingModule, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gcn1 = GCNLayer(input_dim, 12)
        self.gcn2 = GCNLayer(12, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.dropout(x)
        x = self.gcn2(x, adj)
        x = self.norm(x)
        out_degree = adj.sum(dim=2)
        in_degree = adj.sum(dim=1)
        node_exists = (out_degree > 0) | (in_degree > 0)
        x_masked = x * node_exists.unsqueeze(-1).float()
        batch_size = x.size(0)
        x_flatten = x_masked.view(batch_size, -1)
        return x_flatten


class SequenceModel(nn.Module):
    def __init__(self, cksnap_dim=96, kmer_dim=5460, graph_feat_dim=768, n_classes=9, dropout=0.15):
        super().__init__()
        self.cksnap_dim = cksnap_dim
        self.mlp_kmer = nn.Sequential(
            nn.Linear(kmer_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.graph_processor = GraphProcessingModule(64, 12, 12)
        self.classifier = nn.Sequential(
            nn.Linear(cksnap_dim + 256 + graph_feat_dim, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, n_classes)
        )

    def forward(self, cksnap, kmer, adj_matrix):
        x1 = cksnap
        x2 = self.mlp_kmer(kmer)
        batch_size = adj_matrix.size(0)
        node_features = self.create_fixed_node_features().unsqueeze(0).repeat(batch_size, 1, 1).to(adj_matrix.device)
        x3 = self.graph_processor(node_features, adj_matrix)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.classifier(x)
        return x

    def create_fixed_node_features(self):
        nucleotides = ['A', 'C', 'G', 'T']
        kmers = [''.join(combo) for combo in itertools.product(nucleotides, repeat=3)]
        ncp_code = {
            'A': [1, 1, 1],
            'G': [1, 0, 0],
            'C': [0, 1, 0],
            'T': [0, 0, 1]
        }
        eiip_value = {
            'A': 0.1260,
            'G': 0.0806,
            'C': 0.1340,
            'T': 0.1335
        }
        features = []
        for kmer in kmers:
            ncp_features = []
            eiip_features = []
            for base in kmer:
                ncp_features.extend(ncp_code[base])
                eiip_features.append(eiip_value[base])
            feature = ncp_features + eiip_features
            features.append(feature)
        return torch.tensor(features, dtype=torch.float32)


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


# ==================== 特征提取函数（保持不变）====================
def get_min_sequence_length(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen


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
    nucleotides = ['A', 'C', 'G', 'T']
    kmers = [''.join(combo) for combo in itertools.product(nucleotides, repeat=k)]
    kmer_to_idx = {kmer: idx for idx, kmer in enumerate(kmers)}
    n_kmers = len(kmers)
    adj_matrix = np.zeros((n_kmers, n_kmers), dtype=np.float32)
    seq_kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if all(n in nucleotides for n in kmer):
            seq_kmers.append(kmer)
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


# ==================== 主函数 - 直接使用第二折模型 ====================
def main():
    # 1. 参数与路径
    csv_path = os.path.join(base_dir, "dataset", "training_validation.csv")
    fasta_path = os.path.join(base_dir, "dataset", "training_validation_seqs")
    ind_csv_path = os.path.join(base_dir, "dataset", "independent.csv")
    ind_fasta_path = os.path.join(base_dir, "dataset", "independent_seqs")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fasta', default=fasta_path, help='training/validation fasta')
    parser.add_argument('--label_csv', default=csv_path, help='training/validation label csv')
    parser.add_argument('--ind_csv', default=ind_csv_path, help='independent label csv')
    parser.add_argument('--ind_fasta', default=ind_fasta_path, help='independent fasta')
    parser.add_argument('--output_path', default="results", help='output path')
    parser.add_argument('--model_path', default=None, help='第二折模型路径，如果为None则自动查找')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--batch_size', type=int, default=32)
    opt = parser.parse_args()

    os.makedirs(opt.output_path, exist_ok=True)
    set_seed(42)

    # 2. 读取标签信息
    label_df = pd.read_csv(opt.label_csv)
    label_columns = label_df.columns[1:]
    print(f"标签类别: {label_columns.tolist()}")

    # 3. 确定第二折模型路径
    if opt.model_path is None:
        # 自动查找第二折模型
        model_path = os.path.join(opt.output_path, 'best_model_fold_2.pt')
        if not os.path.exists(model_path):
            # 尝试其他可能的命名
            model_path = os.path.join(opt.output_path, 'fold_2_model.pt')
            if not os.path.exists(model_path):
                print("错误: 找不到第二折模型文件!")
                print("请使用 --model_path 参数指定模型路径")
                return
    else:
        model_path = opt.model_path

    print(f"使用模型: {model_path}")

    # 4. 读取独立测试集
    print(f"\n{'=' * 60}")
    print("加载独立测试集")
    print(f"{'=' * 60}")

    # 读取独立标签
    ind_label_df = pd.read_csv(opt.ind_csv)
    ind_labels = {row[0]: row[1:].values.astype(np.float32)
                  for _, row in ind_label_df.iterrows()}

    # 读取独立序列
    ind_fasta = read_nucleotide_sequences(opt.ind_fasta)

    # 5. 提取独立测试集特征
    print(f"\n{'=' * 60}")
    print("提取独立测试集特征")
    print(f"{'=' * 60}")

    @cache_to_file(os.path.join(opt.output_path, 'ind_cksnap.pkl'))
    def get_ind_cksnap():
        return CKSNAP(ind_fasta, gap=5, order='ACGT')

    @cache_to_file(os.path.join(opt.output_path, 'ind_kmer.pkl'))
    def get_ind_kmer():
        return Kmer(ind_fasta, k=6, type='RNA', upto=True, normalize=True, order='ACGT')

    @cache_to_file(os.path.join(opt.output_path, 'ind_adj.pkl'))
    def get_ind_adj():
        return {i[0]: build_de_bruijn_adjacency(i[1], k=3) for i in tqdm(ind_fasta, desc='构建独立测试集邻接矩阵')}

    ind_ck = get_ind_cksnap()
    ind_km = get_ind_kmer()
    ind_adj = get_ind_adj()

    # 6. 创建独立测试集数据加载器
    ind_seqs = [i[0] for i in ind_fasta
                if i[0] in ind_labels and i[0] in ind_ck and i[0] in ind_km and i[0] in ind_adj]

    print(f"独立测试集有效序列数: {len(ind_seqs)}")

    if len(ind_seqs) == 0:
        print("错误: 没有找到有效的独立测试序列!")
        return

    ind_set = SequenceDataset(ind_seqs, ind_ck, ind_km, ind_adj, ind_labels)
    ind_loader = DataLoader(ind_set, batch_size=opt.batch_size, shuffle=False)

    # 7. 加载第二折模型
    print(f"\n{'=' * 60}")
    print("加载第二折模型")
    print(f"{'=' * 60}")

    model = SequenceModel(n_classes=len(label_columns)).to(opt.device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=opt.device))
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    model.eval()

    # 8. 计算多标签评估指标
    print(f"\n{'=' * 60}")
    print("计算多标签评估指标")
    print(f"{'=' * 60}")

    with torch.no_grad():
        logits, ys = [], []
        for c, k, a, y in ind_loader:
            c, k, a = [i.to(opt.device) for i in (c, k, a)]
            logits.append(model(c, k, a).cpu())
            ys.append(y)
        logits = torch.cat(logits)
        ys = torch.cat(ys)
        ind_metrics = compute_multilabel_metrics_literature(ys, logits)

    # 9. 计算并绘制AUC曲线
    print(f"\n{'=' * 60}")
    print("计算各亚细胞定位AUC值并绘制曲线")
    print(f"{'=' * 60}")

    # 计算每个类别的AUC
    auc_scores, fpr_dict, tpr_dict, y_true, y_prob = calculate_auc_per_class(
        model, ind_loader, opt.device, label_columns
    )

    # 绘制AUC曲线
    plot_auc_curves(auc_scores, fpr_dict, tpr_dict, label_columns, opt.output_path)

    # 打印AUC统计信息
    sorted_auc = print_auc_statistics(auc_scores)

    # 保存AUC结果
    auc_results = {
        'auc_scores': auc_scores,
        'fpr_dict': fpr_dict,
        'tpr_dict': tpr_dict,
        'y_true': y_true,
        'y_prob': y_prob,
        'sorted_auc': sorted_auc
    }

    with open(os.path.join(opt.output_path, 'fold2_independent_auc_results.pkl'), 'wb') as f:
        pickle.dump(auc_results, f)

    # 10. 打印独立测试结果
    print("\n独立测试集多标签评估结果:")
    print("-" * 80)
    for k, v in ind_metrics.items():
        print(f"{k}: {v:.4f}")

    # 11. 保存所有结果
    res_file = os.path.join(opt.output_path, 'fold2_independent_test_results')

    with open(res_file + '.txt', 'w') as f:
        f.write('Fold 2 Model - Independent Test Results\n')
        f.write('=' * 50 + '\n')

        f.write('\nMulti-label Metrics:\n')
        f.write('-' * 30 + '\n')
        for k, v in ind_metrics.items():
            f.write(f'{k}: {v:.4f}\n')

        f.write('\nAUC Results:\n')
        f.write('-' * 30 + '\n')
        for class_name, auc_value in sorted_auc:
            f.write(f'{class_name}: {auc_value:.4f}\n')
        f.write(f'\nAverage AUC: {np.mean(list(auc_scores.values())):.4f}\n')

        f.write(f'\nModel Used: {model_path}\n')
        f.write(f'Independent Test Samples: {len(ind_seqs)}\n')

    with open(res_file + '.pkl', 'wb') as f:
        pickle.dump({
            'multi_label_metrics': ind_metrics,
            'auc_results': auc_results,
            'model_path': model_path,
            'test_samples': len(ind_seqs)
        }, f)

    print(f"\n所有结果已保存到: {res_file}.txt")
    print(f"AUC曲线图已保存到: {os.path.join(opt.output_path, 'auc_curves.png')}")


if __name__ == "__main__":
    main()