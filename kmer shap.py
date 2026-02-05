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

# 根据SHAP版本调整导入
try:
    from shap.explainers import Permutation as PermutationExplainer
    print("使用 shap.explainers.Permutation (SHAP 0.41.0+)")
except ImportError:
    try:
        from shap.explainers import PermutationExplainer
        print("使用 shap.explainers.PermutationExplainer (SHAP <= 0.40.0)")
    except ImportError:
        print("警告: 无法导入PermutationExplainer，将使用基于权重的分析")
        PermutationExplainer = None
        SHAP_AVAILABLE = False
    else:
        SHAP_AVAILABLE = True
else:
    SHAP_AVAILABLE = True

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


# 自定义数据集类
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


# 计算多标签评估指标
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


# 特征提取函数
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


# 缓存装饰器
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


def get_feature_type(idx):
    """根据特征索引判断特征类型"""
    if idx < 96:
        return "CKSNAP"
    elif idx < 96 + 256:
        return "Kmer"
    else:
        return "Graph"


def generate_unique_kmer_features(model):
    """
    生成唯一的Kmer特征名称，确保每个原始k-mer名称只出现一次
    """
    print("正在分析Kmer MLP层权重以获取唯一的原始k-mer名称...")

    # 获取Kmer MLP的权重
    kmer_mlp_weights = []
    for layer in model.mlp_kmer:
        if isinstance(layer, nn.Linear):
            kmer_mlp_weights.append(layer.weight.detach().cpu().numpy())

    kmer_feature_mapping = []
    kmer_usage_count = {}  # 记录每个k-mer的使用次数
    feature_names = []

    if len(kmer_mlp_weights) >= 2:
        # 第一层权重: 5460 -> 1024
        w1 = kmer_mlp_weights[0]  # [1024, 5460]
        # 第二层权重: 1024 -> 256
        w2 = kmer_mlp_weights[1]  # [256, 1024]

        # 计算每个256维输出对应的原始k-mer重要性
        combined_weights = np.abs(w2 @ w1)  # [256, 5460]

        # 第一步：为每个256维特征找到最重要的原始k-mer
        for i in range(256):
            # 找到贡献最大的原始k-mer
            top_idx = np.argmax(combined_weights[i])
            if top_idx < len(model.kmer_names):
                top_kmer = model.kmer_names[top_idx]

                # 记录k-mer使用情况
                if top_kmer not in kmer_usage_count:
                    kmer_usage_count[top_kmer] = 1
                else:
                    kmer_usage_count[top_kmer] += 1

                kmer_feature_mapping.append({
                    'index': 96 + i,
                    'original_name': top_kmer,
                    'kmer_name': f"kmer_{top_kmer}"
                })
            else:
                kmer_feature_mapping.append({
                    'index': 96 + i,
                    'original_name': f"Kmer_{i}",
                    'kmer_name': f"Kmer_feat_{i:03d}"
                })

        # 第二步：处理重复的k-mer，为重复的添加后缀
        kmer_name_counts = {}
        unique_kmer_feature_mapping = []

        for mapping in kmer_feature_mapping:
            original_name = mapping['original_name']
            base_name = mapping['kmer_name']

            if original_name.startswith('Kmer_'):  # 非真实k-mer，直接使用
                unique_kmer_feature_mapping.append(mapping)
                feature_names.append(mapping['kmer_name'])
            else:
                # 检查是否重复
                if base_name in kmer_name_counts:
                    kmer_name_counts[base_name] += 1
                    new_name = f"{base_name}_{kmer_name_counts[base_name]}"
                    unique_kmer_feature_mapping.append({
                        'index': mapping['index'],
                        'original_name': mapping['original_name'],
                        'kmer_name': new_name
                    })
                    feature_names.append(new_name)
                else:
                    kmer_name_counts[base_name] = 1
                    unique_kmer_feature_mapping.append(mapping)
                    feature_names.append(base_name)

        kmer_feature_mapping = unique_kmer_feature_mapping

        # 统计唯一k-mer数量
        unique_original_kmers = set([m['original_name'] for m in kmer_feature_mapping])
        print(f"生成的Kmer特征中，唯一原始k-mer数量: {len(unique_original_kmers)}")

        return kmer_feature_mapping, feature_names

    else:
        # 无法获取权重，使用通用名称
        for i in range(256):
            kmer_feature_mapping.append({
                'index': 96 + i,
                'original_name': f"Kmer_{i}",
                'kmer_name': f"Kmer_feat_{i:03d}"
            })
            feature_names.append(f"Kmer_feat_{i:03d}")

        return kmer_feature_mapping, feature_names


def perform_kmer_shap_analysis(model, dataloader, label_columns, device, output_path):
    """
    只进行Kmer特征的SHAP分析，生成kmer_top10_shap_{cell_type}_500samples.png
    """
    if not SHAP_AVAILABLE or PermutationExplainer is None:
        print("SHAP库不可用，跳过SHAP分析")
        return

    print("\n" + "=" * 80)
    print("开始Kmer特征SHAP分析（只分析Kmer特征）")
    print("使用500个样本，生成kmer_top10_shap_{cell_type}.png")
    print("=" * 80)

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

    # 生成Kmer特征名称 - 使用唯一的k-mer名称
    kmer_feature_mapping, kmer_feature_names = generate_unique_kmer_features(model)

    # 为每个亚细胞定位分别计算SHAP值
    cell_types = label_columns

    for cell_idx, cell_type in enumerate(cell_types):
        print(f"\n分析亚细胞: {cell_type}")

        # 使用所有可用样本
        X_all = X
        y_cell = y[:, cell_idx]

        # 统计正负样本数量
        pos_count = np.sum(y_cell == 1)
        neg_count = len(y_cell) - pos_count
        print(f"  总样本数: {len(y_cell)}, 正样本: {pos_count}, 负样本: {neg_count}")

        # 检查预测概率
        def get_pred_probs(x_data):
            x_tensor = torch.tensor(x_data, dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model.forward_for_shap(x_tensor)
            return torch.sigmoid(output).cpu().numpy()[:, cell_idx]

        # 选择500个样本进行SHAP分析
        sample_size = 500
        if X_all.shape[0] < sample_size:
            sample_size = X_all.shape[0]

        # 确保正负样本都被采样到
        if pos_count > 0 and neg_count > 0:
            pos_indices = np.where(y_cell == 1)[0]
            neg_indices = np.where(y_cell == 0)[0]
            pos_sample_size = min(int(sample_size * pos_count / len(y_cell)), len(pos_indices))
            neg_sample_size = min(sample_size - pos_sample_size, len(neg_indices))

            if neg_sample_size < sample_size - pos_sample_size:
                pos_sample_size = sample_size - neg_sample_size

            pos_sample_indices = np.random.choice(pos_indices, pos_sample_size, replace=False)
            neg_sample_indices = np.random.choice(neg_indices, neg_sample_size, replace=False)
            indices = np.concatenate([pos_sample_indices, neg_sample_indices])
        else:
            indices = np.random.choice(X_all.shape[0], sample_size, replace=False)

        X_sample = X_all[indices]
        y_sample = y_cell[indices]

        # 检查样本集的预测概率
        sample_probs = get_pred_probs(X_sample)
        print(f"  使用 {len(indices)} 个样本进行SHAP分析")

        # 使用PermutationExplainer
        try:
            print("  使用PermutationExplainer计算SHAP值...")

            # 创建背景数据
            background_size = min(50, X_sample.shape[0])
            background_indices = np.random.choice(X_sample.shape[0], background_size, replace=False)
            background = X_sample[background_indices]

            # 定义模型预测函数
            def model_predict_single(x):
                x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                with torch.no_grad():
                    output = model.forward_for_shap(x_tensor)
                probs = torch.sigmoid(output).cpu().numpy()
                return probs[:, cell_idx]

            # 创建PermutationExplainer
            min_evals = 2 * X_sample.shape[1] + 1
            max_evals_value = max(3000, min_evals)

            explainer = PermutationExplainer(
                model_predict_single,
                background,
                max_evals=max_evals_value
            )

            # 计算SHAP值
            print("  计算SHAP值，这可能需要一些时间...")
            shap_values = explainer(X_sample)

            # 获取SHAP值数组
            if isinstance(shap_values, (list, tuple)):
                shap_values_array = shap_values[0]
            else:
                shap_values_array = shap_values.values if hasattr(shap_values, 'values') else shap_values

            # 转换为numpy数组
            if hasattr(shap_values_array, 'cpu'):
                shap_values_array = shap_values_array.cpu().numpy()
            elif hasattr(shap_values_array, 'numpy'):
                shap_values_array = shap_values_array.numpy()

            print(f"  SHAP值形状: {shap_values_array.shape}")

            # 计算每个特征的平均绝对SHAP值
            mean_abs_shap = np.abs(shap_values_array).mean(axis=0)

            # 绘制Kmer特征前10个最重要的SHAP摘要图（确保不重复）
            print("  绘制Kmer特征前10个最重要的SHAP摘要图（确保不重复）...")

            # 提取Kmer特征的SHAP值 (索引96-351)
            kmer_start = 96
            kmer_end = 352
            kmer_shap = shap_values_array[:, kmer_start:kmer_end]
            kmer_mean_abs = np.abs(kmer_shap).mean(axis=0)

            # 找出Kmer特征中前10个最重要的（基于SHAP值）
            kmer_top_indices_by_shap = np.argsort(kmer_mean_abs)[-20:][::-1]  # 取前20个，然后去重

            # 确保不重复：基于原始k-mer名称去重
            unique_kmer_indices = []
            seen_original_kmers = set()

            for idx in kmer_top_indices_by_shap:
                if idx < len(kmer_feature_mapping):
                    mapping = kmer_feature_mapping[idx]
                    original_name = mapping['original_name']

                    # 跳过非真实k-mer
                    if original_name.startswith('Kmer_'):
                        continue

                    if original_name not in seen_original_kmers:
                        seen_original_kmers.add(original_name)
                        unique_kmer_indices.append(idx)
                        if len(unique_kmer_indices) >= 10:
                            break

            # 如果不够10个，添加其他不重复的k-mer
            if len(unique_kmer_indices) < 10:
                for idx in range(len(kmer_mean_abs)):
                    if idx not in unique_kmer_indices and idx < len(kmer_feature_mapping):
                        mapping = kmer_feature_mapping[idx]
                        original_name = mapping['original_name']

                        if original_name.startswith('Kmer_'):
                            continue

                        if original_name not in seen_original_kmers:
                            seen_original_kmers.add(original_name)
                            unique_kmer_indices.append(idx)
                            if len(unique_kmer_indices) >= 10:
                                break

            kmer_top_shap = kmer_shap[:, unique_kmer_indices]
            kmer_top_features = X_sample[:, kmer_start:kmer_end][:, unique_kmer_indices]
            kmer_top_names = [kmer_feature_names[i] for i in unique_kmer_indices]

            # 创建固定的颜色映射
            colors_fixed = []
            for l in np.linspace(1, 0, 100):
                colors_fixed.append((30. / 255, 136. / 255, 229. / 255, l))
            for l in np.linspace(0, 1, 100):
                colors_fixed.append((255. / 255, 13. / 255, 87. / 255, l))
            red_blue_cmap_fixed = LinearSegmentedColormap.from_list("red_blue_fixed", colors_fixed, N=200)

            # 绘制Kmer SHAP摘要图
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                kmer_top_shap,
                kmer_top_features,
                feature_names=kmer_top_names,
                plot_type="dot",
                color=red_blue_cmap_fixed,
                show=False,
                max_display=10
            )

            plt.title(f'{cell_type} - Top 10 Kmer Features', fontsize=16, fontweight='bold')
            plt.tight_layout()

            # 保存图像 - 按照原来的格式生成文件名
            kmer_file = os.path.join(output_path, f'{cell_type}_unique.png')
            plt.savefig(kmer_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Kmer特征前10个SHAP图已保存到: {kmer_file}")

            # 保存Kmer前10个特征到CSV
            kmer_df = pd.DataFrame({
                'Feature_Index': [kmer_start + i for i in unique_kmer_indices],
                'Original_Kmer': [
                    kmer_feature_mapping[i]['original_name'] if i < len(kmer_feature_mapping) else f"Kmer_{i}"
                    for i in unique_kmer_indices],
                'Feature_Name': kmer_top_names,
                'Mean_Abs_SHAP': kmer_mean_abs[unique_kmer_indices],
                'Feature_Type': 'Kmer'
            })

            kmer_csv = os.path.join(output_path, f'kmer_top10_features_{cell_type}_unique.csv')
            kmer_df.to_csv(kmer_csv, index=False)
            print(f"  Kmer前10个特征（不重复）已保存到: {kmer_csv}")

            # 打印Kmer前10个特征
            print("\n  Kmer前10个最重要特征（不重复）:")
            for i, idx in enumerate(unique_kmer_indices):
                mapping = kmer_feature_mapping[idx] if idx < len(kmer_feature_mapping) else None
                if mapping:
                    print(f"    {i + 1}. {mapping['original_name']} (SHAP={kmer_mean_abs[idx]:.6f})")

        except Exception as e:
            print(f"  SHAP分析失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nKmer特征SHAP分析完成！")


# 独立测试函数（直接加载第二折模型）
def independent_test_with_kmer_shap(opt):
    """
    直接使用第二折模型进行独立测试和Kmer SHAP分析
    只生成{cell_type}_unique_500samples.png
    """
    print(f"\n{'=' * 80}")
    print(f"使用第2折的最佳模型进行独立测试和Kmer SHAP分析")
    print(f"只生成<cell_type>_unique.png")  # 修复：使用占位符而不是变量
    print(f"{'=' * 80}")

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

    # 进行Kmer特征SHAP分析
    shap_output_path = os.path.join(CACHE_DIR, 'kmer_shap_analysis')
    os.makedirs(shap_output_path, exist_ok=True)

    print(f"\nKmer特征SHAP分析结果将保存到: {shap_output_path}")

    # 进行Kmer特征SHAP分析
    print("\n" + "=" * 80)
    print("开始Kmer特征SHAP分析（只分析Kmer特征）使用500个样本")
    print("=" * 80)
    perform_kmer_shap_analysis(model, ind_loader, label_columns, opt.device, shap_output_path)

    # 保存所有结果
    res_file = os.path.join(CACHE_DIR, 'independent_test_results_kmer_shap_500samples')
    with open(res_file + '.txt', 'w') as f:
        f.write('Independent Test Results (Fold 2 Model) - Kmer SHAP Analysis\n')
        f.write('=' * 80 + '\n')
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
        # 直接进行独立测试和Kmer SHAP分析
        independent_test_with_kmer_shap(opt)
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