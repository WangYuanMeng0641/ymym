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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import jaccard_score, label_ranking_average_precision_score, coverage_error, hamming_loss, \
    label_ranking_loss, average_precision_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc
from tqdm import tqdm
import time
import torch.nn.functional as F

# 添加SHAP相关导入
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

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


def normalize_adjacency(adj):
    """
    有向图的邻接矩阵归一化：D^(-1) * (A + I)
    adj: [B, N, N]
    """
    B, N, _ = adj.shape
    I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
    A_hat = adj + I  # 加自环

    # 有向图使用出度归一化
    D = A_hat.sum(dim=2)  # 出度 [B, N]
    # 避免除零错误
    D_safe = torch.where(D == 0, torch.ones_like(D), D)
    D_inv = torch.diag_embed(1.0 / D_safe)  # D^(-1)

    adj_norm = D_inv @ A_hat
    return adj_norm


def normalize_node_features(x):
    """
    行归一化节点特征
    x: [B, N, F]
    """
    return x / (x.sum(dim=2, keepdim=True) + 1e-8)


# GCN层 (图卷积网络)
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):
        """
        x: [B, N, F]
        adj: [B, N, N]
        """
        # 归一化邻接矩阵（有向图版本）
        adj_norm = normalize_adjacency(adj)

        # 归一化节点特征
        x_norm = normalize_node_features(x)

        # 有向图卷积：A * X
        x = torch.bmm(adj_norm, x_norm)  # [B, N, F]

        # 线性变换
        x = self.linear(x)  # [B, N, out_features]

        # BN 处理
        batch_size, num_nodes, out_features = x.shape
        x = x.view(-1, out_features)
        x = self.bn(x)
        x = x.view(batch_size, num_nodes, out_features)

        return F.relu(x)


# 图处理模块 - 修改为两层GCN：12→12→12
class GraphProcessingModule(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim=12):  # 输入输出都是12维
        super(GraphProcessingModule, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 两层GCN - 12 → 12 → 12
        self.gcn1 = GCNLayer(input_dim, 12)  # 第一层GCN，输出12维
        self.gcn2 = GCNLayer(12, output_dim)  # 第二层GCN，输出12维

        # 归一化层
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.2)  # 添加dropout

    def forward(self, x, adj):
        # x: [batch_size, num_nodes, input_dim]
        # adj: [batch_size, num_nodes, num_nodes] - 有向邻接矩阵

        # 两层GCN处理
        x = self.gcn1(x, adj)  # [batch_size, num_nodes, 12]
        x = self.dropout(x)  # 添加dropout
        x = self.gcn2(x, adj)  # [batch_size, num_nodes, 12]

        # 归一化
        x = self.norm(x)

        # 识别实际存在的节点（有出边或入边的节点）
        # 计算每个节点的出度和入度
        out_degree = adj.sum(dim=2)  # [batch_size, num_nodes] - 出度
        in_degree = adj.sum(dim=1)  # [batch_size, num_nodes] - 入度

        # 节点存在条件：有出边或入边（度大于0）
        node_exists = (out_degree > 0) | (in_degree > 0)  # [batch_size, num_nodes]

        # 用0掩码不存在的节点
        x_masked = x * node_exists.unsqueeze(-1).float()  # [batch_size, num_nodes, 12]

        # 将64个节点的12维特征拼接成768维 [batch_size, 64*12=768]
        batch_size = x.size(0)
        x_flatten = x_masked.view(batch_size, -1)  # [batch_size, 768]

        return x_flatten


# 定义序列模型 - 修改MLP分类器为1120→384→64→9
class SequenceModel(nn.Module):
    def __init__(self, cksnap_dim=96, kmer_dim=5460, graph_feat_dim=768, n_classes=9, dropout=0.15, kmer_names=None):
        super().__init__()

        # CKSNAP: 不经过MLP，直接使用原始特征
        self.cksnap_dim = cksnap_dim

        # 保存Kmer特征名称
        if kmer_names is None:
            # 生成默认的k-mer名称
            NA = 'ACGT'
            kmer_names = []
            for k in range(1, 7):  # 1到6-mer
                for kmer in itertools.product(NA, repeat=k):
                    kmer_names.append(''.join(kmer))
        self.kmer_names = kmer_names[:kmer_dim]  # 确保长度匹配

        # Kmer: 2层普通MLP设计 - 5460 → 1024 → 256
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

        # 图处理模块 - 输出768维拼接特征
        self.graph_processor = GraphProcessingModule(64, 12, 12)  # 输出768维 (64*12)

        # 分类器: 输入维度为 96(CKSNAP) + 256(Kmer) + 768(Graph) = 1120
        # 3层MLP: 1120 → 384 → 64 → 9
        self.classifier = nn.Sequential(
            # 1120 → 384
            nn.Linear(cksnap_dim + 256 + graph_feat_dim, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(dropout),

            # 384 → 64
            nn.Linear(384, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),

            # 64 → 9
            nn.Linear(64, n_classes)
        )

    def forward(self, cksnap, kmer, adj_matrix):
        # 处理CKSNAP特征 - 直接使用，不经过MLP
        x1 = cksnap  # [B, 96]

        # 处理Kmer特征
        x2 = self.mlp_kmer(kmer)  # [B, 5460] → [B, 256]

        # 处理图特征
        batch_size = adj_matrix.size(0)
        node_features = self.create_fixed_node_features().unsqueeze(0).repeat(batch_size, 1, 1).to(adj_matrix.device)
        x3 = self.graph_processor(node_features, adj_matrix)  # [B, 768]

        # 拼接三个特征: 96 + 256 + 768 = 1120维
        x = torch.cat((x1, x2, x3), dim=1)  # [B, 1120]
        x = self.classifier(x)  # [B, 1120] → [B, 9]
        return x

    def forward_for_shap(self, x_combined):
        """
        专门为SHAP设计的forward函数，接受合并的特征输入
        x_combined: [B, 1120] 其中前96维是CKSNAP，接着256维是Kmer，最后768维是Graph
        """
        # 分割特征
        x1 = x_combined[:, :self.cksnap_dim]  # CKSNAP: [B, 96]
        x2 = x_combined[:, self.cksnap_dim:self.cksnap_dim + 256]  # Kmer after MLP: [B, 256]
        x3 = x_combined[:, self.cksnap_dim + 256:]  # Graph: [B, 768]

        # 直接通过分类器
        x = self.classifier(x_combined)  # [B, 9]
        return x

    def create_fixed_node_features(self):
        """创建固定的节点特征（64个3-mer的NCP+EIIP特征）"""
        nucleotides = ['A', 'C', 'G', 'T']
        kmers = [''.join(combo) for combo in itertools.product(nucleotides, repeat=3)]

        # 自定义NCP编码表
        ncp_code = {
            'A': [1, 1, 1],
            'G': [1, 0, 0],
            'C': [0, 1, 0],
            'T': [0, 0, 1]
        }

        # EIIP值
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
            # 拼接：9维NCP + 3维EIIP = 12维
            feature = ncp_features + eiip_features
            features.append(feature)

        return torch.tensor(features, dtype=torch.float32)


# 自定义数据集类（保持不变）
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


# 计算多标签评估指标（保持不变）
def compute_multilabel_metrics_literature(y_true, y_logits):
    """
    计算十个多标签评估指标
    y_true: 真实标签 [n_samples, n_labels]
    y_logits: 模型输出的logits [n_samples, n_labels]
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


# 特征提取函数（保持不变）
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

    # 生成头部和特征名称
    header = []
    kmer_names = []
    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                kmer_name = ''.join(kmer)
                header.append(kmer_name)
                kmer_names.append(kmer_name)
    else:
        for kmer in itertools.product(NA, repeat=k):
            kmer_name = ''.join(kmer)
            header.append(kmer_name)
            kmer_names.append(kmer_name)

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
    print(f"Kmer特征计算完成，共处理 {len(encoding)} 条序列，特征维度：{len(kmer_names)}")
    return encoding, kmer_names


# 构建有向德布鲁因图邻接矩阵（返回原始二值矩阵）
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

    # 移除归一化步骤，直接返回二值矩阵
    return adj_matrix  # 不再进行归一化


# 缓存装饰器（保持不变）
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


# 改进的SHAP分析函数 - 使用1000个样本
def perform_shap_analysis(model, dataloader, label_columns, device, output_path):
    """
    使用PermutationExplainer进行SHAP分析，调整参数以适应高维特征
    只分析CKSNAP和Graph特征
    """
    print("\n" + "=" * 60)
    print("开始SHAP特征重要性分析（使用1000个样本）")
    print("只分析CKSNAP和Graph特征，跳过Kmer特征")
    print("=" * 60)

    # 准备数据
    model.eval()
    all_features = []
    all_labels = []

    # 收集所有数据
    with torch.no_grad():
        for cksnap, kmer, adj, labels in tqdm(dataloader, desc="收集数据"):
            # 将三个特征转换为模型输入格式
            cksnap = cksnap.to(device)
            kmer = kmer.to(device)
            adj = adj.to(device)

            # 获取模型中间特征
            kmer_processed = model.mlp_kmer(kmer)  # [B, 256]

            # 处理图特征
            batch_size = adj.size(0)
            node_features = model.create_fixed_node_features().unsqueeze(0).repeat(batch_size, 1, 1).to(device)
            graph_processed = model.graph_processor(node_features, adj)  # [B, 768]

            # 拼接三个特征: 96(CKSNAP) + 256(Kmer) + 768(Graph) = 1120
            combined_features = torch.cat((cksnap, kmer_processed, graph_processed), dim=1)

            all_features.append(combined_features.cpu().numpy())
            all_labels.append(labels.numpy())

    # 合并所有数据
    X = np.vstack(all_features)  # [n_samples, 1120]
    y = np.vstack(all_labels)  # [n_samples, 9]

    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"总样本数: {X.shape[0]}，使用1000个样本进行SHAP分析")

    # 定义特征名称
    # CKSNAP特征名称 (96维)
    cksnap_names = []
    aaPairs = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
               'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    for g in range(6):  # gap=0-5
        for pair in aaPairs:
            cksnap_names.append(f"CKSNAP_gap{g}_{pair}")

    # Graph特征名称 (768维)
    graph_names = []
    nucleotides = ['A', 'C', 'G', 'T']
    kmers_3 = [''.join(combo) for combo in itertools.product(nucleotides, repeat=3)]
    feature_types = ['NCP1', 'NCP2', 'NCP3', 'NCP4', 'NCP5', 'NCP6',
                     'NCP7', 'NCP8', 'NCP9', 'EIIP1', 'EIIP2', 'EIIP3']
    for i, kmer in enumerate(kmers_3):
        for j, feat_type in enumerate(feature_types):
            graph_names.append(f"Graph_{kmer}_{feat_type}")

    # 所有特征名称（只包含CKSNAP和Graph）
    all_feature_names = cksnap_names + graph_names

    # 为每个亚细胞分别计算SHAP值
    cell_types = label_columns

    for cell_idx, cell_type in enumerate(cell_types):
        print(f"\n分析亚细胞: {cell_type}")

        # 使用所有样本，不筛选正负样本
        X_all = X  # 所有样本
        y_cell = y[:, cell_idx]  # 该细胞类型的标签

        # 统计正负样本数量
        pos_count = np.sum(y_cell == 1)
        neg_count = len(y_cell) - pos_count

        print(f"  总样本数: {len(y_cell)}, 正样本: {pos_count}, 负样本: {neg_count}")
        print(f"  正样本比例: {pos_count / len(y_cell) * 100:.1f}%")

        # 使用1000个样本加快计算
        sample_size = min(1000, X_all.shape[0])
        indices = np.random.choice(X_all.shape[0], sample_size, replace=False)
        X_sample = X_all[indices]

        print(f"  使用{sample_size}个样本进行SHAP分析以加快计算")

        # 方法1：使用PermutationExplainer
        try:
            print("  使用PermutationExplainer计算SHAP值...")

            # 创建背景数据（使用较少的样本作为背景）
            background_size = min(100, X_sample.shape[0])
            background_indices = np.random.choice(X_sample.shape[0], background_size, replace=False)
            background = X_sample[background_indices]

            # 定义模型预测函数
            def model_predict_single(x):
                """包装模型预测函数，只返回当前类别的概率"""
                x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                with torch.no_grad():
                    output = model.forward_for_shap(x_tensor)
                probs = torch.sigmoid(output).cpu().numpy()
                return probs[:, cell_idx]  # 只返回当前类别的概率

            # 测试预测函数
            test_preds = model_predict_single(X_sample[:5])
            print(f"  前5个样本的预测概率: {test_preds}")
            print(f"  预测概率范围: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
            print(f"  预测概率标准差: {test_preds.std():.6f}")

            # 如果预测变化很小，SHAP值可能接近0
            if test_preds.std() < 0.01:
                print(f"  警告: 预测概率变化很小（标准差={test_preds.std():.6f}），SHAP值可能接近0")

            # 创建PermutationExplainer - 增加max_evals到至少2241
            min_evals = 2 * X_sample.shape[1] + 1  # 2 * num_features + 1
            max_evals_value = max(5000, min_evals)  # 至少5000

            print(f"  最小需要的评估次数: {min_evals}")
            print(f"  实际使用的评估次数: {max_evals_value}")

            explainer = shap.PermutationExplainer(
                model_predict_single,
                background,
                max_evals=max_evals_value
            )

            # 计算SHAP值
            shap_values = explainer(X_sample)

            # 分析每个特征组（只分析CKSNAP和Graph）
            analyze_feature_groups(shap_values, X_sample, all_feature_names,
                                   cell_type, output_path)

            # 新增：创建全局SHAP摘要图（两个特征一起的前10个特征）
            create_global_shap_summary(shap_values, X_sample, all_feature_names,
                                       cell_type, output_path)

        except Exception as e:
            print(f"  PermutationExplainer失败: {e}")
            print("  尝试使用基于模型权重的特征重要性分析...")

            # 如果PermutationExplainer失败，使用基于权重的特征重要性
            analyze_feature_importance_by_weights(model, X_sample, all_feature_names,
                                                  cell_type, output_path, device)

    print("\nSHAP分析完成！")


def create_global_shap_summary(shap_values, X_sample, all_feature_names, cell_type, output_path, top_n=10):
    """
    创建CKSNAP和Graph特征组合的全局SHAP摘要图（前10个特征）
    """
    print(f"  创建全局SHAP摘要图（前{top_n}个特征）...")

    # ===== 1. 处理SHAP值输入 =====
    if isinstance(shap_values, (list, tuple)):
        if len(shap_values) == 2:
            shap_values_array = shap_values[0]
        else:
            shap_values_array = shap_values
    else:
        shap_values_array = shap_values.values if hasattr(shap_values, 'values') else shap_values

    # ===== 2. 转换为numpy数组 =====
    if hasattr(shap_values_array, 'cpu'):
        shap_values_array = shap_values_array.cpu().numpy()
    elif hasattr(shap_values_array, 'numpy'):
        shap_values_array = shap_values_array.numpy()

    # ===== 3. 计算所有特征的平均绝对SHAP值 =====
    mean_abs_shap = np.abs(shap_values_array).mean(axis=0)

    # ===== 4. 选择前top_n个最重要的特征，确保不重复 =====
    all_indices = np.arange(len(all_feature_names))

    # 按重要性排序
    sorted_indices_by_importance = all_indices[np.argsort(mean_abs_shap)[::-1]]

    # 选择前top_n个特征，并确保特征名称不重复
    selected_indices = []
    selected_names = set()

    for idx in sorted_indices_by_importance:
        if len(selected_indices) >= top_n:
            break

        # 获取特征名称
        feature_name = all_feature_names[idx]

        # 检查是否重复
        if feature_name not in selected_names:
            selected_indices.append(idx)
            selected_names.add(feature_name)

    # ===== 5. 保存全局特征重要性结果 =====
    global_importance_data = []
    for idx in selected_indices:
        feature_name = all_feature_names[idx]
        # 确定特征组
        if idx < 96:
            feature_group = "CKSNAP"
        else:
            feature_group = "Graph"

        global_importance_data.append({
            'Feature_Index': idx,
            'Feature_Name': feature_name,
            'Feature_Group': feature_group,
            'Mean_Abs_SHAP': mean_abs_shap[idx],
            'SHAP_Min': shap_values_array[:, idx].min(),
            'SHAP_Max': shap_values_array[:, idx].max(),
            'SHAP_Std': shap_values_array[:, idx].std()
        })

    global_importance_df = pd.DataFrame(global_importance_data)
    global_importance_file = os.path.join(output_path, f'shap_global_top{top_n}_{cell_type}.csv')
    global_importance_df.to_csv(global_importance_file, index=False)
    print(f"    全局特征重要性前{top_n}已保存到: {global_importance_file}")

    # ===== 6. 创建全局SHAP摘要图 =====
    if len(selected_indices) > 0:
        # 提取要显示的数据
        display_shap = shap_values_array[:, selected_indices]
        display_features = X_sample[:, selected_indices]
        display_names = [all_feature_names[idx] for idx in selected_indices]

        # 简短的名称用于显示
        short_names = []
        for name in display_names:
            # 截断过长的名称
            if len(name) > 30:
                short_names.append(name[:27] + "...")
            else:
                short_names.append(name)

        # 创建自定义颜色映射（红蓝）
        colors = []
        for l in np.linspace(1, 0, 100):
            colors.append((30. / 255, 136. / 255, 229. / 255, l))
        for l in np.linspace(0, 1, 100):
            colors.append((255. / 255, 13. / 255, 87. / 255, l))
        red_blue_cmap = LinearSegmentedColormap.from_list("red_blue", colors, N=200)

        # 创建图形
        plt.figure(figsize=(12, max(8, len(selected_indices) * 0.4)))

        # 绘制SHAP摘要图
        shap.summary_plot(
            display_shap,
            display_features,
            feature_names=short_names,
            plot_type="dot",
            color=red_blue_cmap,
            show=False,
            max_display=len(selected_indices)
        )

        plt.title(f"{cell_type} - Global Top {len(selected_indices)} Features (CKSNAP+Graph)",
                  fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 保存图像
        plot_file = os.path.join(output_path, f'shap_top{top_n}_{cell_type}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    全局SHAP摘要图已保存到: {plot_file}")


def analyze_feature_groups(shap_values, X_sample, all_feature_names, cell_type, output_path):
    """分析特征组的SHAP值，只分析CKSNAP和Graph"""
    # 特征分组（只包含CKSNAP和Graph）
    feature_groups = {
        'CKSNAP': (0, 96),
        'Graph': (96, 96 + 768)  # 跳过Kmer特征
    }

    # ===== 1. 处理不同类型的SHAP值输入 =====
    if isinstance(shap_values, (list, tuple)):
        # shap_values可能是列表或元组
        if len(shap_values) == 2:  # (shap_values_array, base_values)
            shap_values_array = shap_values[0]
        else:
            shap_values_array = shap_values
    else:
        # 检查是否有.values属性（shap.Explanation对象）
        shap_values_array = shap_values.values if hasattr(shap_values, 'values') else shap_values

    # ===== 2. 转换为numpy数组 =====
    if hasattr(shap_values_array, 'cpu'):
        # 如果是torch.Tensor，转换为numpy
        shap_values_array = shap_values_array.cpu().numpy()
    elif hasattr(shap_values_array, 'numpy'):
        # 如果是tensorflow Tensor，转换为numpy
        shap_values_array = shap_values_array.numpy()

    # ===== 3. 验证SHAP值 =====
    print(f"  SHAP值形状: {shap_values_array.shape}")
    print(f"  SHAP值统计 - 最小值: {shap_values_array.min():.6f}, "
          f"最大值: {shap_values_array.max():.6f}, "
          f"平均值: {shap_values_array.mean():.6f}, "
          f"标准差: {shap_values_array.std():.6f}")

    # 检查SHAP值是否全为0
    if np.all(np.abs(shap_values_array) < 1e-8):
        print("  警告: SHAP值全为0，可能原因：")
        print("    1. 模型对该细胞类型的预测不敏感")
        print("    2. SHAP计算失败")
        print("    3. 所有特征对预测都没有贡献")
        return

    # ===== 4. 分析每个特征组 =====
    for group_name, (start_idx, end_idx) in feature_groups.items():
        print(f"  分析{group_name}特征...")

        # 提取该组的SHAP值
        group_shap_values = shap_values_array[:, start_idx:end_idx]

        # 计算每个特征的平均绝对SHAP值
        mean_abs_shap = np.abs(group_shap_values).mean(axis=0)

        # 找出非零特征（重要性大于阈值的特征）
        threshold = 1e-6  # 设置一个较小的阈值
        nonzero_mask = mean_abs_shap > threshold
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_features = [all_feature_names[start_idx + i] for i in nonzero_indices]

        print(f"    {group_name}特征总数: {mean_abs_shap.shape[0]}")
        print(f"    非零特征数: {len(nonzero_indices)}")

        if len(nonzero_indices) == 0:
            print(f"    警告: {group_name}所有特征的SHAP值都接近0，跳过")
            continue

        # ===== 5. 选择最重要的特征（前10个），确保不重复 =====
        # 按重要性排序
        sorted_by_importance = nonzero_indices[np.argsort(mean_abs_shap[nonzero_indices])[::-1]]

        # 选择前10个特征，确保特征名称不重复
        selected_indices = []
        selected_names = set()

        for idx in sorted_by_importance:
            if len(selected_indices) >= 10:
                break

            feature_name = all_feature_names[start_idx + idx]
            if feature_name not in selected_names:
                selected_indices.append(idx)
                selected_names.add(feature_name)

        # ===== 6. 保存特征重要性结果 =====
        importance_data = []
        for i, idx in enumerate(selected_indices):
            feature_name = all_feature_names[start_idx + idx]
            importance_data.append({
                'Feature_Index': idx,
                'Feature_Name': feature_name,
                'Mean_Abs_SHAP': mean_abs_shap[idx],
                'SHAP_Min': group_shap_values[:, idx].min(),
                'SHAP_Max': group_shap_values[:, idx].max(),
                'SHAP_Std': group_shap_values[:, idx].std()
            })

        importance_df = pd.DataFrame(importance_data)
        importance_file = os.path.join(output_path, f'shap_top10_{cell_type}_{group_name}.csv')
        importance_df.to_csv(importance_file, index=False)
        print(f"    特征重要性前{len(selected_indices)}已保存到: {importance_file}")

        # ===== 7. 创建SHAP摘要图 =====
        if len(selected_indices) > 0:
            # 提取要显示的特征数据
            display_shap = group_shap_values[:, selected_indices]
            display_features = X_sample[:, start_idx + selected_indices]
            display_names = [all_feature_names[start_idx + i] for i in selected_indices]

            # 简短的名称用于显示
            short_names = []
            for name in display_names:
                # 截断过长的名称
                if len(name) > 30:
                    short_names.append(name[:27] + "...")
                else:
                    short_names.append(name)

            # 创建自定义颜色映射（红蓝）
            colors = []
            for l in np.linspace(1, 0, 100):
                colors.append((30. / 255, 136. / 255, 229. / 255, l))
            for l in np.linspace(0, 1, 100):
                colors.append((255. / 255, 13. / 255, 87. / 255, l))
            red_blue_cmap = LinearSegmentedColormap.from_list("red_blue", colors, N=200)

            # 创建图形
            plt.figure(figsize=(12, max(8, len(selected_indices) * 0.4)))

            # 绘制SHAP摘要图
            shap.summary_plot(
                display_shap,
                display_features,
                feature_names=short_names,
                plot_type="dot",
                color=red_blue_cmap,
                show=False,
                max_display=len(selected_indices)
            )

            plt.title(f"{cell_type} - {group_name} Top {len(selected_indices)} Features",
                      fontsize=14, fontweight='bold')
            plt.tight_layout()

            # 保存图像
            plot_file = os.path.join(output_path, f'shap_top10_{cell_type}_{group_name}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    SHAP前{len(selected_indices)}特征摘要图已保存到: {plot_file}")


def analyze_feature_importance_by_weights(model, X_sample, all_feature_names, cell_type, output_path, device):
    """基于模型权重的特征重要性分析"""
    print("  使用基于模型权重的特征重要性分析...")

    # 获取分类器的权重
    classifier_weights = []
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            classifier_weights.append(layer.weight.detach().cpu().numpy())

    # 最后一层权重 (64 -> 9)
    if len(classifier_weights) >= 3:
        final_weights = classifier_weights[-1]  # [9, 64]
        second_last_weights = classifier_weights[-2]  # [64, 384]
        first_weights = classifier_weights[0]  # [384, 1120]

        # 假设我们要分析的类别索引
        cell_idx = get_cell_type_index(cell_type)
        if cell_idx is not None and cell_idx < final_weights.shape[0]:
            # 计算特征重要性
            cell_weight = final_weights[cell_idx]  # [64]
            importance_64 = np.abs(cell_weight)  # [64]

            # 传播到384维
            importance_384 = np.abs(second_last_weights.T @ importance_64)  # [384]

            # 传播到1120维
            importance_1120 = np.abs(first_weights.T @ importance_384)  # [1120]

            # 特征分组（只包含CKSNAP和Graph）
            feature_groups = {
                'CKSNAP': (0, 96),
                'Graph': (96 + 256, 96 + 256 + 768)  # 跳过Kmer特征
            }

            # 为每个特征组分析
            for group_name, (start_idx, end_idx) in feature_groups.items():
                print(f"  分析{group_name}特征...")

                # 提取该组的重要性
                group_importance = importance_1120[start_idx:end_idx]
                group_feature_names = all_feature_names[start_idx:end_idx] if start_idx < len(all_feature_names) else []

                # 归一化重要性
                if np.sum(group_importance) > 0:
                    group_importance = group_importance / np.sum(group_importance)

                # 选择前10个最重要的特征，确保不重复
                sorted_by_importance = np.argsort(group_importance)[::-1]
                selected_indices = []
                selected_names = set()

                for idx in sorted_by_importance:
                    if len(selected_indices) >= 10:
                        break

                    if idx < len(group_feature_names):
                        feature_name = group_feature_names[idx]
                        if feature_name not in selected_names:
                            selected_indices.append(idx)
                            selected_names.add(feature_name)

                # 保存特征重要性结果
                if len(selected_indices) > 0:
                    importance_df = pd.DataFrame({
                        'Feature': [group_feature_names[i] for i in selected_indices],
                        'Importance': group_importance[selected_indices]
                    })

                    importance_file = os.path.join(output_path, f'weight_top10_{cell_type}_{group_name}.csv')
                    importance_df.to_csv(importance_file, index=False)
                    print(f"    特征重要性前{len(selected_indices)}已保存到: {importance_file}")

                    # 绘制条形图
                    create_feature_importance_plot(group_feature_names, group_importance,
                                                   selected_indices, cell_type, group_name, output_path)

            # 新增：全局特征重要性分析（基于权重）
            print("  创建基于权重的全局特征重要性分析...")

            # 选择前10个最重要的全局特征，确保不重复
            sorted_global_importance = np.argsort(importance_1120)[::-1]
            selected_global_indices = []
            selected_global_names = set()

            for idx in sorted_global_importance:
                if len(selected_global_indices) >= 10:
                    break

                if idx < len(all_feature_names):
                    feature_name = all_feature_names[idx]
                    if feature_name not in selected_global_names:
                        selected_global_indices.append(idx)
                        selected_global_names.add(feature_name)

            # 保存全局特征重要性
            if len(selected_global_indices) > 0:
                global_importance_df = pd.DataFrame({
                    'Feature_Index': selected_global_indices,
                    'Feature_Name': [all_feature_names[i] for i in selected_global_indices],
                    'Feature_Group': ['CKSNAP' if i < 96 else 'Graph' for i in selected_global_indices],
                    'Importance': [importance_1120[i] for i in selected_global_indices]
                })

                global_importance_file = os.path.join(output_path, f'weight_global_top10_{cell_type}.csv')
                global_importance_df.to_csv(global_importance_file, index=False)
                print(f"    基于权重的全局特征重要性前10已保存到: {global_importance_file}")


def get_cell_type_index(cell_type):
    """获取细胞类型索引（简化版本）"""
    # 这里需要根据实际情况调整
    cell_type_mapping = {
        'Exosome': 0, 'Microvesicle': 1, 'Apoptotic_body': 2,
        # ... 添加其他细胞类型映射
    }
    return cell_type_mapping.get(cell_type)


def create_feature_importance_plot(feature_names, importance, top_indices,
                                   cell_type, group_name, output_path):
    """创建特征重要性条形图"""
    if len(top_indices) > 0:
        plt.figure(figsize=(12, 8))

        # 获取特征的信息
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance[top_indices]

        # 创建条形图
        colors = plt.cm.RdYlBu(np.linspace(0, 1, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_importance, color=colors)

        plt.xlabel('特征重要性', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.title(f"{cell_type} - {group_name} Top {len(top_features)} 特征重要性", fontsize=14, fontweight='bold')
        plt.yticks(range(len(top_features)), top_features)

        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, top_importance)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{value:.4f}', va='center', fontsize=9)

        plt.tight_layout()

        # 保存图像
        plot_file = os.path.join(output_path, f'feature_top{len(top_features)}_{cell_type}_{group_name}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    特征重要性前{len(top_features)}图已保存到: {plot_file}")


# 独立测试函数（直接加载第二折模型）
def independent_test_with_shap(opt):
    """
    直接使用第二折模型进行独立测试和SHAP分析
    """
    print(f"\n{'=' * 60}")
    print(f"使用第2折的最佳模型进行独立测试和SHAP分析")
    print(f"{'=' * 60}")

    # 设置输出路径
    CACHE_DIR = opt.output_path
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 读取独立标签
    ind_label_df = pd.read_csv(opt.ind_csv)
    label_columns = ind_label_df.columns[1:]
    ind_labels = {row[0]: row[1:].values.astype(np.float32)
                  for _, row in ind_label_df.iterrows()}

    # 读取独立序列
    ind_fasta = read_nucleotide_sequences(opt.ind_fasta)

    # 独立特征（缓存）
    @cache_to_file(os.path.join(CACHE_DIR, 'ind_cksnap.pkl'))
    def get_ind_cksnap():
        return CKSNAP(ind_fasta, gap=5, order='ACGT')

    @cache_to_file(os.path.join(CACHE_DIR, 'ind_kmer.pkl'))
    def get_ind_kmer():
        # 修改：返回编码和k-mer名称
        return Kmer(ind_fasta, k=6, type='RNA', upto=True, normalize=True, order='ACGT')

    @cache_to_file(os.path.join(CACHE_DIR, 'ind_adj.pkl'))
    def get_ind_adj():
        return {i[0]: build_de_bruijn_adjacency(i[1], k=3) for i in tqdm(ind_fasta, desc='adj ind')}

    ind_ck = get_ind_cksnap()

    # 修改：解包k-mer结果
    ind_km_result = get_ind_kmer()
    if isinstance(ind_km_result, tuple) and len(ind_km_result) == 2:
        ind_km, kmer_names = ind_km_result
        print(f"获取到 {len(kmer_names)} 个Kmer特征名称")
    else:
        ind_km = ind_km_result
        # 生成默认的k-mer名称
        NA = 'ACGT'
        kmer_names = []
        for k in range(1, 7):
            for kmer in itertools.product(NA, repeat=k):
                kmer_names.append(''.join(kmer))
        print(f"使用默认的 {len(kmer_names)} 个Kmer特征名称")

    ind_adj = get_ind_adj()

    # 对齐
    ind_seqs = [i[0] for i in ind_fasta
                if i[0] in ind_labels and i[0] in ind_ck and i[0] in ind_km and i[0] in ind_adj]

    print(f"独立测试序列数: {len(ind_seqs)}")

    # 创建数据集
    ind_set = SequenceDataset(ind_seqs, ind_ck, ind_km, ind_adj, ind_labels)
    ind_loader = DataLoader(ind_set, batch_size=opt.batch_size, shuffle=False)

    # 加载第二折的最佳模型
    best_model_path = os.path.join(CACHE_DIR, 'best_model_fold_2.pt')
    if not os.path.exists(best_model_path):
        print(f"错误: 找不到模型文件 {best_model_path}")
        print("请确保已经完成了5折交叉验证训练")
        return

    print(f"加载模型: {best_model_path}")
    # 修改：创建模型时传递kmer_names
    model = SequenceModel(n_classes=len(label_columns), kmer_names=kmer_names).to(opt.device)
    model.load_state_dict(torch.load(best_model_path, map_location=opt.device))
    model.eval()

    # 进行独立测试
    print("\n进行独立测试...")
    with torch.no_grad():
        logits, ys = [], []
        for c, k, a, y in ind_loader:
            c, k, a = [i.to(opt.device) for i in (c, k, a)]
            logits.append(model(c, k, a).cpu())
            ys.append(y)
        logits = torch.cat(logits)
        ys = torch.cat(ys)
        ind_metrics = compute_multilabel_metrics_literature(ys, logits)

    # 打印独立测试结果
    print("\n独立测试集结果:")
    print("-" * 80)
    for k, v in ind_metrics.items():
        print(f"{k}: {v:.4f}")

    # 进行SHAP分析
    shap_output_path = os.path.join(CACHE_DIR, 'shap_analysis')
    os.makedirs(shap_output_path, exist_ok=True)

    print(f"\nSHAP分析结果将保存到: {shap_output_path}")

    # 进行SHAP分析（只分析CKSNAP和Graph）
    print("\n" + "=" * 60)
    print("开始CKSNAP和Graph特征SHAP分析（使用1000个样本）")
    print("跳过Kmer特征分析")
    print("=" * 60)
    perform_shap_analysis(model, ind_loader, label_columns, opt.device, shap_output_path)

    # 保存所有结果
    res_file = os.path.join(CACHE_DIR, 'independent_test_results')
    with open(res_file + '.txt', 'w') as f:
        f.write('Independent Test Results (Fold 2 Model)\n')
        f.write('=' * 50 + '\n')
        for k, v in ind_metrics.items():
            f.write(f'{k}: {v:.4f}\n')

    with open(res_file + '.pkl', 'wb') as f:
        pickle.dump({
            'independent_test': ind_metrics,
            'shap_output_path': shap_output_path
        }, f)

    print(f"\n所有结果已保存到: {CACHE_DIR}")


# 主函数
def main():
    # 参数设置
    csv_path = os.path.join(base_dir, "dataset", "training_validation.csv")
    fasta_path = os.path.join(base_dir, "dataset", "training_validation_seqs")
    ind_csv_path = os.path.join(base_dir, "dataset", "independent.csv")
    ind_fasta_path = os.path.join(base_dir, "dataset", "independent_seqs")

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='independent',
                        choices=['independent', 'train'],
                        help='运行模式: independent(直接独立测试) 或 train(完整训练)')
    parser.add_argument('--input_fasta', default=fasta_path, help='training/validation fasta')
    parser.add_argument('--label_csv', default=csv_path, help='training/validation label csv')
    parser.add_argument('--ind_csv', default=ind_csv_path, help='independent label csv')
    parser.add_argument('--ind_fasta', default=ind_fasta_path, help='independent fasta')
    parser.add_argument('--output_path', default="results", help='output path')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_folds', type=int, default=5, help='number of folds for cross-validation')
    opt = parser.parse_args()

    # 根据模式选择运行
    if opt.mode == 'independent':
        # 直接进行独立测试和SHAP分析
        independent_test_with_shap(opt)
    else:
        # 运行完整的训练流程
        run_training_pipeline(opt)


# 原有的训练流程函数
def run_training_pipeline(opt):
    """完整的训练流程"""
    os.makedirs(opt.output_path, exist_ok=True)
    CACHE_DIR = opt.output_path
    set_seed(42)

    # 读取训练+验证标签
    label_df = pd.read_csv(opt.label_csv)
    label_columns = label_df.columns[1:]
    labels_dict = {row[0]: row[1:].values.astype(np.float32)
                   for _, row in label_df.iterrows()}

    # 读取训练+验证序列
    fasta_records = read_nucleotide_sequences(opt.input_fasta)

    # 特征提取（缓存）- 修改以支持k-mer名称
    @cache_to_file(os.path.join(CACHE_DIR, 'train_cksnap.pkl'))
    def get_train_cksnap():
        return CKSNAP(fasta_records, gap=5, order='ACGT')

    @cache_to_file(os.path.join(CACHE_DIR, 'train_kmer.pkl'))
    def get_train_kmer():
        # 修改：返回编码和k-mer名称
        return Kmer(fasta_records, k=6, type='RNA', upto=True, normalize=True, order='ACGT')

    @cache_to_file(os.path.join(CACHE_DIR, 'train_adj.pkl'))
    def get_train_adj():
        return {i[0]: build_de_bruijn_adjacency(i[1], k=3) for i in tqdm(fasta_records, desc='adj train')}

    cksnap_feat = get_train_cksnap()

    # 修改：解包k-mer结果
    kmer_result = get_train_kmer()
    if isinstance(kmer_result, tuple) and len(kmer_result) == 2:
        kmer_feat, kmer_names = kmer_result
        print(f"获取到 {len(kmer_names)} 个Kmer特征名称")
    else:
        kmer_feat = kmer_result
        # 生成默认的k-mer名称
        NA = 'ACGT'
        kmer_names = []
        for k in range(1, 7):
            for kmer in itertools.product(NA, repeat=k):
                kmer_names.append(''.join(kmer))
        print(f"使用默认的 {len(kmer_names)} 个Kmer特征名称")

    adj_feat = get_train_adj()

    # 保存k-mer名称以备后用
    with open(os.path.join(CACHE_DIR, 'kmer_names.pkl'), 'wb') as f:
        pickle.dump(kmer_names, f)

    # 准备5折交叉验证
    all_seqs = [i[0] for i in fasta_records
                if i[0] in labels_dict and i[0] in cksnap_feat and i[0] in kmer_feat and i[0] in adj_feat]

    print(f"总序列数: {len(all_seqs)}")

    kf = KFold(n_splits=opt.n_folds, shuffle=True, random_state=42)

    # 训练单个fold的函数
    def train_fold(fold, train_seqs, valid_seqs, cksnap_feat, kmer_feat, adj_feat, labels_dict, label_columns, opt,
                   CACHE_DIR):
        print(f"\n{'=' * 50}")
        print(f"正在训练第 {fold + 1} 折")
        print(f"{'=' * 50}")

        # 数据加载器
        train_set = SequenceDataset(train_seqs, cksnap_feat, kmer_feat, adj_feat, labels_dict)
        valid_set = SequenceDataset(valid_seqs, cksnap_feat, kmer_feat, adj_feat, labels_dict)
        train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=opt.batch_size, shuffle=False)

        # 模型 - 修改：传递kmer_names
        model = SequenceModel(n_classes=len(label_columns), kmer_names=kmer_names).to(opt.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        best_acc = 0.0
        patience_c = 0
        fold_model_path = os.path.join(CACHE_DIR, f'best_model_fold_{fold + 1}.pt')
        best_metrics = None

        # 训练
        for epoch in range(1, opt.epochs + 1):
            # ---- train ----
            model.train()
            train_loss = 0.0
            for c, k, a, y in tqdm(train_loader, desc=f'Fold {fold + 1} - Epoch {epoch}/{opt.epochs} - train'):
                c, k, a, y = [i.to(opt.device) for i in (c, k, a, y)]
                optimizer.zero_grad()
                out = model(c, k, a)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # ---- valid ----
            model.eval()
            with torch.no_grad():
                logits, ys = [], []
                for c, k, a, y in valid_loader:
                    c, k, a = [i.to(opt.device) for i in (c, k, a)]
                    logits.append(model(c, k, a).cpu())
                    ys.append(y)
                logits = torch.cat(logits)
                ys = torch.cat(ys)
                metrics = compute_multilabel_metrics_literature(ys, logits)

            scheduler.step(metrics['accuracy'])

            print(f'Fold {fold + 1} - Epoch {epoch:03d}  Train Loss: {train_loss / len(train_loader):.4f}  '
                  f'Valid-Accuracy: {metrics["accuracy"]:.4f}  Best: {best_acc:.4f}  Patience: {patience_c}/{opt.patience}')

            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                best_metrics = metrics.copy()
                patience_c = 0
                torch.save(model.state_dict(), fold_model_path)
                print(f'Fold {fold + 1} - 新的最佳模型已保存!')
            else:
                patience_c += 1
                if patience_c >= opt.patience:
                    print(f'Fold {fold + 1} - Early stopping!')
                    break

        # 清理内存
        del model, train_loader, valid_loader
        clear_memory()

        return best_acc, best_metrics, fold_model_path

    # 存储每折的结果
    fold_results = []
    fold_models = []

    # 5折交叉验证训练
    for fold, (train_idx, valid_idx) in enumerate(kf.split(all_seqs)):
        train_seqs = [all_seqs[i] for i in train_idx]
        valid_seqs = [all_seqs[i] for i in valid_idx]

        print(f"\nFold {fold + 1}: 训练集 {len(train_seqs)} 条, 验证集 {len(valid_seqs)} 条")

        best_acc, best_metrics, model_path = train_fold(
            fold, train_seqs, valid_seqs, cksnap_feat, kmer_feat, adj_feat,
            labels_dict, label_columns, opt, CACHE_DIR
        )

        fold_results.append({
            'fold': fold + 1,
            'best_accuracy': best_acc,
            'metrics': best_metrics
        })
        fold_models.append((fold + 1, best_acc, model_path))

        print(f"\nFold {fold + 1} 完成 - 最佳准确率: {best_acc:.4f}")

    # 选择最佳模型
    best_fold_info = max(fold_models, key=lambda x: x[1])
    best_fold, best_fold_acc, best_model_path = best_fold_info

    print(f"\n{'=' * 60}")
    print(f"5折交叉验证完成!")
    print(f"最佳模型来自第 {best_fold} 折，验证集准确率: {best_fold_acc:.4f}")
    print(f"{'=' * 60}")

    # 保存结果
    with open(os.path.join(CACHE_DIR, 'cv_results.pkl'), 'wb') as f:
        pickle.dump({
            'fold_results': fold_results,
            'best_fold': best_fold,
            'kmer_names': kmer_names  # 保存k-mer名称
        }, f)

    print(f"\n训练完成！模型已保存到: {CACHE_DIR}")


if __name__ == "__main__":
    main()