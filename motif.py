# ===== 在代码最开头添加 =====
import os

# 解决OpenMP库重复初始化问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# 导入其他库
import torch
import re, sys
import argparse
import random
import numpy as np
from Bio import SeqIO
import itertools
from collections import Counter, defaultdict
import pandas as pd
import pickle
import gc
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import heapq

# 新增导入
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import logomaker
import json
import time
from datetime import datetime

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


# ===== 辅助函数：6-mer编码 =====
def encode_6mer(seq):
    """将6-mer序列编码为24维向量（6×4 one-hot）"""
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoding = np.zeros(24, dtype=np.float32)

    for i, base in enumerate(seq):
        if base in base_map:
            encoding[i * 4 + base_map[base]] = 1

    return encoding


def encode_6mers(seq_list):
    """批量编码6-mer序列"""
    return np.array([encode_6mer(seq) for seq in seq_list], dtype=np.float32)


# ===== 数据集类 =====
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


# ===== 完整的SequenceModel（基于原始代码结构）=====
class GraphProcessingModule(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim=12):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 简化的GCN层
        self.gcn1 = nn.Linear(input_dim, 12)
        self.gcn2 = nn.Linear(12, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj):
        # 简化处理
        x = F.relu(self.gcn1(x))
        x = self.dropout(x)
        x = F.relu(self.gcn2(x))

        # 拼接特征
        batch_size = x.size(0)
        x_flatten = x.view(batch_size, -1)

        return x_flatten


class SequenceModel(nn.Module):
    """基于原始代码的完整模型"""

    def __init__(self, cksnap_dim=96, kmer_dim=5460, graph_feat_dim=768, n_classes=9, dropout=0.15):
        super().__init__()

        self.cksnap_dim = cksnap_dim

        # Kmer处理部分
        self.mlp_kmer = nn.Sequential(
            nn.Linear(kmer_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 图处理模块
        self.graph_processor = GraphProcessingModule(64, 12, 12)

        # 分类器
        classifier_input = cksnap_dim + 256 + graph_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, n_classes)
        )

    def forward(self, cksnap, kmer, adj_matrix):
        # 处理CKSNAP特征
        x1 = cksnap

        # 处理Kmer特征
        x2 = self.mlp_kmer(kmer)

        # 处理图特征
        batch_size = adj_matrix.size(0)
        node_features = self.create_fixed_node_features().unsqueeze(0).repeat(batch_size, 1, 1).to(adj_matrix.device)
        x3 = self.graph_processor(node_features, adj_matrix)

        # 拼接特征
        x = torch.cat((x1, x2, x3), dim=1)
        output = self.classifier(x)

        return output

    def create_fixed_node_features(self):
        """创建固定的节点特征"""
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


# ===== 6-mer分析函数 =====
def generate_6mer_names():
    """生成所有可能的6-mer名称"""
    nucleotides = ['A', 'C', 'G', 'T']
    kmer_names = []

    # 生成所有6-mer
    for kmer in itertools.product(nucleotides, repeat=6):
        kmer_names.append(''.join(kmer))

    print(f"共生成 {len(kmer_names)} 个6-mer")
    return kmer_names


def get_6mer_indices_in_kmer_features():
    """
    计算6-mer在1-6mer特征向量中的位置
    6-mer: 4096个 (位置 1364-5459)
    """
    sixmer_start = 1364
    sixmer_end = 1364 + 4096
    return sixmer_start, sixmer_end


class WrapperModelFor6merIG(nn.Module):
    """包装模型，专门用于6-mer的IG分析"""

    def __init__(self, original_model, sixmer_start, sixmer_end):
        super().__init__()
        self.original_model = original_model
        self.sixmer_start = sixmer_start
        self.sixmer_end = sixmer_end

        # 创建全零的占位符用于其他k-mer部分
        self.kmer_placeholder = None

    def create_kmer_placeholder(self, batch_size, device):
        """创建k-mer占位符"""
        if self.kmer_placeholder is None:
            placeholder = torch.zeros(5460, device=device)
            self.kmer_placeholder = placeholder

        # 扩展到batch size
        return self.kmer_placeholder.unsqueeze(0).repeat(batch_size, 1)

    def forward(self, kmer_6mer_only, cksnap, adj):
        """
        前向传播：将6-mer部分插入到完整k-mer中
        kmer_6mer_only: [B, 4096] 只包含6-mer
        cksnap: [B, 96]
        adj: [B, 64, 64]
        """
        batch_size = kmer_6mer_only.shape[0]
        device = kmer_6mer_only.device

        # 创建完整的k-mer特征
        full_kmer = self.create_kmer_placeholder(batch_size, device)

        # 将6-mer部分插入到正确位置
        full_kmer[:, self.sixmer_start:self.sixmer_end] = kmer_6mer_only

        # 使用原始模型
        return self.original_model(cksnap, full_kmer, adj)


# ===== 聚类函数（分批处理，解决内存问题）=====
# ===== 修复聚类函数中的索引问题 =====
def cluster_6mers_by_sequence_batch(sequences, scores, method='hierarchical',
                                    similarity_threshold=0.8, min_cluster_size=3,
                                    batch_size=50000, max_sequences=50000):
    """
    分批处理大量6-mer序列的聚类

    参数:
    - sequences: 6-mer序列列表
    - scores: 对应的IG分数列表
    - batch_size: 每批处理的序列数量
    - max_sequences: 最大处理序列数

    返回:
    - clusters: 聚类结果列表
    """
    if len(sequences) == 0:
        return []

    print(f"开始聚类 {len(sequences)} 个6-mer...")
    print(f"使用分批处理: 每批 {batch_size} 个序列，最多处理 {max_sequences} 个")

    # 1. 首先按分数排序，选择最重要的序列
    combined = list(zip(sequences, scores))
    combined.sort(key=lambda x: x[1], reverse=True)

    # 限制处理的数量
    n_to_process = min(len(combined), max_sequences)
    combined = combined[:n_to_process]

    sequences_limited = [item[0] for item in combined]
    scores_limited = [item[1] for item in combined]

    print(f"选择前 {n_to_process} 个最高分的6-mer进行聚类")

    # 2. 分批编码
    encoded_seqs = []
    for i in range(0, len(sequences_limited), batch_size):
        batch_seqs = sequences_limited[i:i + batch_size]
        batch_encoded = encode_6mers(batch_seqs)
        encoded_seqs.append(batch_encoded)

        if (i // batch_size) % 10 == 0:
            print(f"  已编码 {i + len(batch_seqs)} / {len(sequences_limited)} 个6-mer")

    # 合并编码结果
    encoded_seqs = np.vstack(encoded_seqs)
    print(f"编码完成，总维度: {encoded_seqs.shape}")

    # 3. 使用MiniBatchKMeans进行预聚类（适合大数据集）
    # 估计聚类数量：根据数据量动态调整
    n_clusters = min(100, len(sequences_limited) // 100)
    if n_clusters < 10:
        n_clusters = 10

    print(f"使用MiniBatchKMeans预聚类，聚类数: {n_clusters}")

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, random_state=42)
    kmeans_labels = kmeans.fit_predict(encoded_seqs)

    # 4. 在每个KMeans聚类内部进行更精细的层次聚类
    final_clusters = []

    for cluster_id in range(n_clusters):
        # 获取当前聚类的索引
        indices = np.where(kmeans_labels == cluster_id)[0]

        if len(indices) < min_cluster_size:
            continue

        print(f"  处理KMeans聚类 {cluster_id + 1}/{n_clusters}: {len(indices)} 个6-mer")

        # 如果聚类太大，进一步细分
        if len(indices) > 1000:
            # 只取前500个最高分的进行细聚类
            cluster_data = [(sequences_limited[i], scores_limited[i]) for i in indices]
            cluster_data.sort(key=lambda x: x[1], reverse=True)
            cluster_data = cluster_data[:500]

            sub_sequences = [item[0] for item in cluster_data]
            sub_scores = [item[1] for item in cluster_data]
            sub_encoded = encode_6mers(sub_sequences)
        else:
            sub_sequences = [sequences_limited[i] for i in indices]
            sub_scores = [scores_limited[i] for i in indices]
            sub_encoded = encoded_seqs[indices]

        # 计算相似性矩阵（小规模）
        if len(sub_sequences) <= 1000:
            # 小规模聚类可以直接计算相似性
            sub_similarity = cosine_similarity(sub_encoded)
            sub_distance = 1 - sub_similarity

            # 层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - similarity_threshold,
                metric='precomputed',
                linkage='average'
            )

            try:
                sub_labels = clustering.fit_predict(sub_distance)
            except:
                # 如果层次聚类失败，将整个子集作为一个聚类
                sub_labels = np.zeros(len(sub_sequences))
        else:
            # 大规模子集，使用MiniBatchKMeans
            sub_n_clusters = max(3, len(sub_sequences) // 100)
            sub_kmeans = MiniBatchKMeans(n_clusters=sub_n_clusters, batch_size=1000)
            sub_labels = sub_kmeans.fit_predict(sub_encoded)

        # 整理子聚类
        unique_sub_labels = np.unique(sub_labels)
        for sub_label in unique_sub_labels:
            if sub_label == -1:
                continue

            sub_indices = np.where(sub_labels == sub_label)[0]
            if len(sub_indices) < min_cluster_size:
                continue

            # 提取当前子聚类的序列和分数
            cluster_seqs = [sub_sequences[i] for i in sub_indices]
            cluster_scores = [sub_scores[i] for i in sub_indices]

            if not cluster_seqs:
                continue

            # 计算平均分数
            avg_score = np.mean(cluster_scores)

            # 找到代表性序列（分数最高的）
            # 修复：使用子聚类内部的索引
            rep_idx = np.argmax(cluster_scores)
            rep_seq = cluster_seqs[rep_idx]
            rep_score = cluster_scores[rep_idx]

            # 计算聚类内相似性
            if len(sub_indices) > 1:
                # 获取子聚类的编码
                cluster_encoded = sub_encoded[sub_indices]
                cluster_distances = 1 - cosine_similarity(cluster_encoded)
                intra_similarity = 1 - np.mean(cluster_distances)
            else:
                intra_similarity = 1.0

            final_clusters.append({
                'cluster_id': len(final_clusters),
                'sequences': cluster_seqs,
                'scores': cluster_scores,
                'avg_score': float(avg_score),
                'representative_seq': rep_seq,
                'representative_score': float(rep_score),
                'size': len(sub_indices),
                'intra_similarity': float(intra_similarity),
                'indices': sub_indices.tolist()
            })

    # 按平均分数排序
    final_clusters.sort(key=lambda x: x['avg_score'], reverse=True)

    print(f"聚类完成：共得到 {len(final_clusters)} 个聚类")
    return final_clusters


# ===== 修改 extract_and_cluster_6mers_by_compartment_optimized 函数，添加批次保存功能 =====
def extract_and_cluster_6mers_by_compartment_optimized_with_checkpoint(
        model, dataloader, label_columns, device,
        sixmer_names, output_dir,
        max_6mers_per_compartment=50000,
        min_score_threshold=0.0001,
        max_batches=None,
        checkpoint_interval=10,  # 每处理多少批次保存一次检查点
        resume_from_checkpoint=True  # 是否从检查点恢复
):
    """
    优化版本：减少内存使用，只保留高分6-mer，支持检查点保存和恢复

    参数:
    - checkpoint_interval: 每处理多少个批次保存一次检查点
    - resume_from_checkpoint: 是否从检查点恢复
    """
    print(f"\n{'=' * 60}")
    print(f"优化版本（支持检查点）：按{len(label_columns)}个细胞区室提取和聚类6-mer")
    print(f"最大6-mer数量: {max_6mers_per_compartment} 每个区室")
    print(f"最小IG阈值: {min_score_threshold}")
    print(f"检查点间隔: 每{checkpoint_interval}个批次")
    print(f"从检查点恢复: {resume_from_checkpoint}")
    print(f"{'=' * 60}")

    model.eval()

    # 获取6-mer在k-mer特征中的位置
    sixmer_start, sixmer_end = get_6mer_indices_in_kmer_features()

    # 创建包装模型
    wrapper_model = WrapperModelFor6merIG(model, sixmer_start, sixmer_end).to(device)
    wrapper_model.eval()

    # 初始化IG解释器
    ig = IntegratedGradients(wrapper_model)

    # 创建检查点目录
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 检查点文件路径
    checkpoint_file = os.path.join(checkpoint_dir, 'batch_checkpoint.pkl')
    checkpoint_meta_file = os.path.join(checkpoint_dir, 'checkpoint_metadata.json')

    # 存储每个区室的前N个最高分6-mer
    compartment_top_6mers = defaultdict(list)  # {comp_name: [(score, sixmer, seq_id), ...]}

    # 批次统计
    batch_count = 0
    total_6mers_processed = 0
    total_positive_samples = 0
    processed_batches = set()  # 记录已经处理过的批次索引

    # 如果从检查点恢复，加载之前的状态
    if resume_from_checkpoint and os.path.exists(checkpoint_file):
        print(f"\n从检查点恢复...")
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)

            compartment_top_6mers = checkpoint_data['compartment_top_6mers']
            processed_batches = set(checkpoint_data['processed_batches'])
            batch_count = checkpoint_data['batch_count']
            total_6mers_processed = checkpoint_data['total_6mers_processed']
            total_positive_samples = checkpoint_data['total_positive_samples']

            print(f"  已恢复检查点: 已处理 {len(processed_batches)} 个批次")
            print(f"  已提取 {total_6mers_processed} 个6-mer")

            # 加载元数据
            if os.path.exists(checkpoint_meta_file):
                with open(checkpoint_meta_file, 'r') as f:
                    meta = json.load(f)
                print(f"  检查点创建时间: {meta.get('created_at', '未知')}")

        except Exception as e:
            print(f"  警告: 加载检查点失败: {str(e)}")
            print("  将重新开始处理")
            compartment_top_6mers = defaultdict(list)
            processed_batches = set()
    else:
        print(f"\n开始新的处理...")

    print("开始提取最高分6-mer...")

    # 处理所有批次
    for batch_idx, (cksnap, kmer, adj, labels) in enumerate(dataloader):
        # 如果已经处理过这个批次，跳过
        if batch_idx in processed_batches:
            if batch_idx % 50 == 0:
                print(f"跳过已处理批次 {batch_idx}...")
            continue

        # 检查是否达到最大批次限制
        if max_batches is not None and batch_count >= max_batches:
            print(f"已达到最大批次限制 ({max_batches})，停止处理")
            break

        batch_count += 1
        processed_batches.add(batch_idx)

        if batch_count % 10 == 0:
            print(f"处理批次 {batch_idx} ({batch_count}批次已处理)...")
            print(f"  已处理阳性样本: {total_positive_samples}")
            print(f"  已提取6-mer: {total_6mers_processed}")

        # 移动到设备
        cksnap, kmer, adj, labels = cksnap.to(device), kmer.to(device), adj.to(device), labels.to(device)
        batch_size = kmer.size(0)

        # 提取6-mer部分
        kmer_6mer_only = kmer[:, sixmer_start:sixmer_end]
        kmer_6mer_only.requires_grad = True

        # 创建baseline（零向量）
        baseline_6mer = torch.zeros_like(kmer_6mer_only)

        # 对每个区室处理
        for comp_idx, comp_name in enumerate(label_columns):
            # 获取该区室为阳性的样本
            pos_mask = labels[:, comp_idx] == 1
            pos_count = pos_mask.sum().item()

            if pos_count == 0:
                continue

            total_positive_samples += pos_count

            pos_6mer = kmer_6mer_only[pos_mask]
            pos_adj = adj[pos_mask]
            pos_cksnap = cksnap[pos_mask]

            try:
                # 计算IG attribution
                attributions, delta = ig.attribute(
                    inputs=pos_6mer,
                    baselines=baseline_6mer[pos_mask],
                    target=comp_idx,
                    additional_forward_args=(pos_cksnap, pos_adj),
                    return_convergence_delta=True,
                    n_steps=20,
                    method='riemann_trapezoid',
                    internal_batch_size=min(4, pos_6mer.shape[0])
                )

                # 对于每个阳性样本，只记录最高分的几个6-mer
                for sample_idx in range(attributions.shape[0]):
                    sample_attributions = attributions[sample_idx].abs().detach().cpu().numpy()

                    # 找到分数最高的前10个6-mer
                    top_indices = np.argsort(sample_attributions)[-10:][::-1]

                    for sixmer_idx in top_indices:
                        score = sample_attributions[sixmer_idx]

                        # 跳过分数太低的
                        if score < min_score_threshold:
                            continue

                        sixmer = sixmer_names[sixmer_idx]
                        seq_id = f"batch{batch_idx}_sample{sample_idx}"

                        # 使用最小堆维护前N个最高分
                        current_list = compartment_top_6mers[comp_name]

                        if len(current_list) < max_6mers_per_compartment:
                            heapq.heappush(current_list, (score, sixmer, seq_id))
                        else:
                            # 如果堆已满，且新分数高于最小分数，则替换
                            if score > current_list[0][0]:
                                heapq.heapreplace(current_list, (score, sixmer, seq_id))

                        total_6mers_processed += 1

            except Exception as e:
                if batch_count <= 5:
                    print(f"    计算{comp_name}的IG时出错: {str(e)[:100]}")
                continue

        # 每处理checkpoint_interval个批次保存一次检查点
        if batch_count % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_file=checkpoint_file,
                checkpoint_meta_file=checkpoint_meta_file,
                compartment_top_6mers=compartment_top_6mers,
                processed_batches=processed_batches,
                batch_count=batch_count,
                total_6mers_processed=total_6mers_processed,
                total_positive_samples=total_positive_samples
            )

    # 处理完成后保存最终检查点
    save_checkpoint(
        checkpoint_file=checkpoint_file,
        checkpoint_meta_file=checkpoint_meta_file,
        compartment_top_6mers=compartment_top_6mers,
        processed_batches=processed_batches,
        batch_count=batch_count,
        total_6mers_processed=total_6mers_processed,
        total_positive_samples=total_positive_samples,
        is_final=True
    )

    print(f"\n完成{batch_count}个批次的分析")
    print(f"总共处理了 {total_positive_samples} 个阳性样本")
    print(f"总共收集了约 {total_6mers_processed} 个6-mer")

    # 将堆转换为排序列表
    for comp_name in compartment_top_6mers:
        heap = compartment_top_6mers[comp_name]
        # 堆是最小堆，转换为降序列表
        sorted_list = sorted(heap, key=lambda x: x[0], reverse=True)
        compartment_top_6mers[comp_name] = sorted_list

        if sorted_list:
            print(f"{comp_name}: 保留了 {len(sorted_list)} 个最高分6-mer")
            print(f"  最高分: {sorted_list[0][0]:.6f}, 最低分: {sorted_list[-1][0]:.6f}")
        else:
            print(f"{comp_name}: 没有提取到6-mer")

    # 对每个区室的6-mer进行聚类
    print(f"\n开始对每个区室的6-mer进行聚类...")
    clustered_results = {}

    for comp_name in label_columns:
        if comp_name not in compartment_top_6mers or len(compartment_top_6mers[comp_name]) == 0:
            print(f"{comp_name}: 没有提取到6-mer，跳过聚类")
            clustered_results[comp_name] = []
            continue

        sixmer_list = compartment_top_6mers[comp_name]
        print(f"\n{comp_name}: 共保留 {len(sixmer_list)} 个最高分6-mer")

        # 提取序列和分数
        sequences = [item[1] for item in sixmer_list]  # sixmer
        scores = [item[0] for item in sixmer_list]  # score

        # 使用优化的分批聚类
        clusters = cluster_6mers_by_sequence_batch(
            sequences, scores,
            method='hierarchical',
            similarity_threshold=0.8,
            min_cluster_size=3,
            batch_size=50000,
            max_sequences=50000  # 最多处理50000个
        )

        clustered_results[comp_name] = clusters

        # 输出聚类统计
        if clusters:
            print(f"  聚类结果: {len(clusters)} 个聚类")
            for i, cluster in enumerate(clusters[:3]):  # 显示前3个聚类
                print(f"    聚类 {i + 1}: {cluster['size']} 个6-mer，平均分数: {cluster['avg_score']:.4f}")
                print(
                    f"      代表性序列: {cluster['representative_seq']} (分数: {cluster['representative_score']:.4f})")
        else:
            print(f"  未得到有效的聚类")

    return clustered_results, compartment_top_6mers


def save_checkpoint(checkpoint_file, checkpoint_meta_file, compartment_top_6mers,
                    processed_batches, batch_count, total_6mers_processed,
                    total_positive_samples, is_final=False):
    """
    保存检查点
    """
    try:
        # 准备检查点数据
        checkpoint_data = {
            'compartment_top_6mers': compartment_top_6mers,
            'processed_batches': list(processed_batches),
            'batch_count': batch_count,
            'total_6mers_processed': total_6mers_processed,
            'total_positive_samples': total_positive_samples,
            'timestamp': time.time()
        }

        # 保存检查点数据
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        # 保存元数据
        meta = {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'batch_count': batch_count,
            'processed_batches': len(processed_batches),
            'total_6mers_processed': total_6mers_processed,
            'total_positive_samples': total_positive_samples,
            'is_final': is_final
        }

        with open(checkpoint_meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

        if is_final:
            print(f"✓ 已保存最终检查点: {checkpoint_file}")
            print(f"  共处理 {batch_count} 个批次")
            print(f"  共提取 {total_6mers_processed} 个6-mer")
        elif batch_count % 50 == 0:  # 每50批次打印一次保存信息
            print(f"✓ 已保存检查点: {checkpoint_file}")
            print(f"  当前批次: {batch_count}, 已处理批次: {len(processed_batches)}")

    except Exception as e:
        print(f"警告: 保存检查点失败: {str(e)}")


# ===== 添加批次结果导出函数 =====
def export_batch_results(compartment_top_6mers, output_dir):
    """
    导出批次处理结果
    """
    print(f"\n{'=' * 60}")
    print("导出批次处理结果")
    print(f"{'=' * 60}")

    batch_results_dir = os.path.join(output_dir, 'batch_results')
    os.makedirs(batch_results_dir, exist_ok=True)

    # 导出每个区室的6-mer统计
    summary_data = []

    for comp_name, sixmer_list in compartment_top_6mers.items():
        if not sixmer_list:
            continue

        # 导出详细列表
        detail_file = os.path.join(batch_results_dir, f'{comp_name}_detailed_batches.csv')

        # 从6-mer ID中提取批次信息
        batch_data = defaultdict(list)
        for score, sixmer, seq_id in sixmer_list:
            # 解析批次信息
            batch_match = re.search(r'batch(\d+)_sample(\d+)', seq_id)
            if batch_match:
                batch_num = int(batch_match.group(1))
                sample_num = int(batch_match.group(2))

                batch_data[batch_num].append({
                    '6mer': sixmer,
                    'score': score,
                    'sample_id': sample_num,
                    'seq_id': seq_id
                })

        # 创建批次统计DataFrame
        batch_stats = []
        for batch_num in sorted(batch_data.keys()):
            sixmers_in_batch = batch_data[batch_num]
            avg_score = np.mean([item['score'] for item in sixmers_in_batch])
            max_score = np.max([item['score'] for item in sixmers_in_batch])
            min_score = np.min([item['score'] for item in sixmers_in_batch])

            batch_stats.append({
                'compartment': comp_name,
                'batch_number': batch_num,
                'sixmer_count': len(sixmers_in_batch),
                'avg_score': avg_score,
                'max_score': max_score,
                'min_score': min_score,
                'unique_6mers': len(set([item['6mer'] for item in sixmers_in_batch]))
            })

        # 保存批次统计
        if batch_stats:
            batch_stats_df = pd.DataFrame(batch_stats)
            batch_stats_file = os.path.join(batch_results_dir, f'{comp_name}_batch_stats.csv')
            batch_stats_df.to_csv(batch_stats_file, index=False, encoding='utf-8')

            # 添加到汇总
            for stat in batch_stats:
                summary_data.append(stat)

    # 保存汇总统计
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(batch_results_dir, 'all_compartments_batch_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')

        # 按区室分组汇总
        compartment_summary = summary_df.groupby('compartment').agg({
            'batch_number': 'count',
            'sixmer_count': 'sum',
            'avg_score': 'mean',
            'unique_6mers': 'sum'
        }).reset_index()

        compartment_summary.columns = ['Compartment', 'Batches_Processed', 'Total_6mers',
                                       'Average_Score', 'Unique_6mers']

        comp_summary_file = os.path.join(batch_results_dir, 'compartment_summary.csv')
        compartment_summary.to_csv(comp_summary_file, index=False, encoding='utf-8')

        print(f"\n批次结果已导出到: {batch_results_dir}")
        print(f"\n批次处理统计:")
        print(compartment_summary.to_string(index=False))

    return batch_results_dir


# ===== 保存原始6-mer数据 =====
def save_original_6mers(compartment_top_6mers, output_dir):
    """
    保存每个区室的原始6-mer数据
    """
    print(f"\n{'=' * 60}")
    print("保存原始6-mer数据")
    print(f"{'=' * 60}")

    original_dir = os.path.join(output_dir, 'original_6mers')
    os.makedirs(original_dir, exist_ok=True)

    for comp_name, sixmer_list in compartment_top_6mers.items():
        if not sixmer_list:
            continue

        # 保存为CSV
        csv_file = os.path.join(original_dir, f'{comp_name}_original_6mers.csv')
        df = pd.DataFrame(sixmer_list, columns=['score', '6mer', 'seq_id'])
        df = df.sort_values('score', ascending=False)
        df.to_csv(csv_file, index=False, encoding='utf-8')

        # 保存为FASTA
        fasta_file = os.path.join(original_dir, f'{comp_name}_original_6mers.fasta')
        with open(fasta_file, 'w', encoding='utf-8') as f:
            for i, (score, sixmer, seq_id) in enumerate(sixmer_list):
                f.write(f">seq{i + 1}_score{score:.4f}_{seq_id}\n")
                f.write(f"{sixmer}\n")

        print(f"{comp_name}: 保存了 {len(sixmer_list)} 个6-mer")

    print(f"原始6-mer数据已保存到: {original_dir}")


# ===== 从聚类构建PWM =====
# ===== 修改PWM生成函数，不使用伪计数 =====
def create_pwm_from_cluster(cluster_sequences, cluster_scores=None):
    """
    从聚类中的序列创建PWM矩阵（不使用伪计数）

    参数:
    - cluster_sequences: 聚类中的6-mer序列列表
    - cluster_scores: 对应的IG分数（可选，用于加权）

    返回:
    - pwm: 位置权重矩阵 [6, 4]
    - consensus: 共识序列
    """
    if not cluster_sequences:
        return None, ""

    # 初始化PWM矩阵
    pwm = np.zeros((6, 4))
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    total_weight = 0

    for seq_idx, seq in enumerate(cluster_sequences):
        # 如果有分数，使用分数作为权重
        weight = cluster_scores[seq_idx] if cluster_scores else 1.0
        total_weight += weight

        for pos in range(6):
            base = seq[pos]
            if base in base_map:
                pwm[pos, base_map[base]] += weight

    # 直接归一化（不使用伪计数）
    # 检查每行的总和，如果为0则设置为均匀分布（避免除0错误）
    for pos in range(6):
        row_sum = pwm[pos].sum()
        if row_sum > 0:
            pwm[pos] = pwm[pos] / row_sum
        else:
            # 如果该位置没有观察到任何碱基，则使用均匀分布
            pwm[pos] = np.array([0.25, 0.25, 0.25, 0.25])

    # 生成共识序列
    consensus = ""
    for pos in range(6):
        max_idx = np.argmax(pwm[pos])
        if max_idx == 0:
            consensus += 'A'
        elif max_idx == 1:
            consensus += 'C'
        elif max_idx == 2:
            consensus += 'G'
        else:
            consensus += 'T'

    return pwm, consensus


# ===== 修改信息含量计算函数，处理零概率 =====
def calculate_information_content(pwm, background=None):
    """
    计算PWM的信息含量（以bits为单位）
    处理零概率情况，避免log2(0)
    """
    if background is None:
        background = [0.25, 0.25, 0.25, 0.25]  # 假设均匀背景

    ic = 0
    for pos in range(pwm.shape[0]):
        pos_ic = 0
        for base_idx in range(4):
            if pwm[pos, base_idx] > 0:
                # 只有当概率大于0时才计算，避免log2(0)
                pos_ic += pwm[pos, base_idx] * np.log2(pwm[pos, base_idx] / background[base_idx])
        ic += pos_ic

    return ic


# ===== 修改build_pwm_from_clusters函数，添加不使用伪计数的选项 =====
def build_pwm_from_clusters(clustered_results, output_dir, top_clusters_per_compartment=3, use_pseudocount=False):
    """
    从聚类结果构建PWM矩阵
    每个区室选择top_clusters_per_compartment个聚类构建PWM

    参数:
    - use_pseudocount: 是否使用伪计数（默认为False，不使用）

    返回:
    - compartment_pwms: 每个区室的PWM列表
    - pwm_dir: PWM保存目录
    """
    print(f"\n{'=' * 60}")
    print(f"从聚类构建PWM矩阵（每个区室选择前{top_clusters_per_compartment}个聚类）")
    print(f"使用伪计数: {use_pseudocount}")
    print(f"{'=' * 60}")

    pwm_dir = os.path.join(output_dir, 'clustered_pwm_matrices')
    os.makedirs(pwm_dir, exist_ok=True)

    # 添加设置文件
    settings_file = os.path.join(pwm_dir, 'pwm_settings.txt')
    with open(settings_file, 'w', encoding='utf-8') as f:
        f.write("PWM生成设置\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"使用伪计数: {use_pseudocount}\n")
        f.write(f"每个区室选择聚类数: {top_clusters_per_compartment}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    compartment_pwms = {}
    compartment_consensus = {}
    compartment_ic = {}

    for comp_name, clusters in clustered_results.items():
        if not clusters:
            print(f"\n{comp_name}: 没有聚类数据，跳过")
            continue

        print(f"\n为{comp_name}构建PWM矩阵...")
        print(f"  聚类数量: {len(clusters)}")

        # 选择前top_clusters_per_compartment个聚类
        selected_clusters = clusters[:min(top_clusters_per_compartment, len(clusters))]

        pwms = []
        consensus_seqs = []
        ic_values = []

        for cluster_idx, cluster in enumerate(selected_clusters):
            print(f"  处理聚类 {cluster_idx + 1}: {cluster['size']} 个6-mer，平均分数: {cluster['avg_score']:.4f}")

            # 从聚类构建PWM（使用新的函数）
            if use_pseudocount:
                # 如果使用伪计数，调用原始函数
                pwm, consensus = create_pwm_from_cluster_with_pseudocount(cluster['sequences'], cluster['scores'])
            else:
                # 如果不使用伪计数，调用新函数
                pwm, consensus = create_pwm_from_cluster(cluster['sequences'], cluster['scores'])

            if pwm is not None:
                # 计算信息含量
                ic = calculate_information_content(pwm)

                pwms.append(pwm)
                consensus_seqs.append(consensus)
                ic_values.append(ic)

                print(f"    共识序列: {consensus}")
                print(f"    信息含量: {ic:.3f} bits")

                # 保存单个PWM矩阵
                suffix = "_with_pseudo" if use_pseudocount else "_no_pseudo"
                pwm_file = os.path.join(pwm_dir, f'{comp_name}_cluster{cluster_idx + 1}_pwm{suffix}.csv')
                pwm_df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])
                pwm_df.index.name = 'Position'
                pwm_df.to_csv(pwm_file, encoding='utf-8')

                # 保存聚类信息
                cluster_info_file = os.path.join(pwm_dir, f'{comp_name}_cluster{cluster_idx + 1}_info{suffix}.txt')
                with open(cluster_info_file, 'w', encoding='utf-8') as f:
                    f.write(f"聚类 {cluster_idx + 1} - {comp_name}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"聚类大小: {cluster['size']}\n")
                    f.write(f"平均IG分数: {cluster['avg_score']:.4f}\n")
                    f.write(f"代表性序列: {cluster['representative_seq']}\n")
                    f.write(f"共识序列: {consensus}\n")
                    f.write(f"信息含量: {ic:.3f} bits\n")
                    f.write(f"使用伪计数: {use_pseudocount}\n")
                    f.write(f"聚类内相似性: {cluster['intra_similarity']:.3f}\n\n")
                    f.write("聚类中的序列:\n")
                    for i, seq in enumerate(cluster['sequences'][:20]):  # 只显示前20个
                        f.write(f"  {i + 1}. {seq}: {cluster['scores'][i]:.4f}\n")

        if pwms:
            compartment_pwms[comp_name] = pwms
            compartment_consensus[comp_name] = consensus_seqs
            compartment_ic[comp_name] = ic_values

    return compartment_pwms, compartment_consensus, compartment_ic, pwm_dir


# ===== 保留原始的使用伪计数的函数（可选） =====
def create_pwm_from_cluster_with_pseudocount(cluster_sequences, cluster_scores=None, pseudocount=0.1):
    """
    从聚类中的序列创建PWM矩阵（使用伪计数）

    参数:
    - cluster_sequences: 聚类中的6-mer序列列表
    - cluster_scores: 对应的IG分数（可选，用于加权）
    - pseudocount: 伪计数值（默认0.1）

    返回:
    - pwm: 位置权重矩阵 [6, 4]
    - consensus: 共识序列
    """
    if not cluster_sequences:
        return None, ""

    # 初始化PWM矩阵
    pwm = np.zeros((6, 4))
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    total_weight = 0

    for seq_idx, seq in enumerate(cluster_sequences):
        # 如果有分数，使用分数作为权重
        weight = cluster_scores[seq_idx] if cluster_scores else 1.0
        total_weight += weight

        for pos in range(6):
            base = seq[pos]
            if base in base_map:
                pwm[pos, base_map[base]] += weight

    # 添加伪计数并归一化
    pwm += pseudocount
    pwm = pwm / pwm.sum(axis=1, keepdims=True)

    # 生成共识序列
    consensus = ""
    for pos in range(6):
        max_idx = np.argmax(pwm[pos])
        if max_idx == 0:
            consensus += 'A'
        elif max_idx == 1:
            consensus += 'C'
        elif max_idx == 2:
            consensus += 'G'
        else:
            consensus += 'T'

    return pwm, consensus

def calculate_information_content(pwm, background=None):
    """
    计算PWM的信息含量（以bits为单位）
    """
    if background is None:
        background = [0.25, 0.25, 0.25, 0.25]  # 假设均匀背景

    ic = 0
    for pos in range(pwm.shape[0]):
        pos_ic = 0
        for base_idx in range(4):
            if pwm[pos, base_idx] > 0:
                pos_ic += pwm[pos, base_idx] * np.log2(pwm[pos, base_idx] / background[base_idx])
        ic += pos_ic

    return ic


def build_pwm_from_clusters(clustered_results, output_dir, top_clusters_per_compartment=3):
    """
    从聚类结果构建PWM矩阵
    每个区室选择top_clusters_per_compartment个聚类构建PWM

    返回:
    - compartment_pwms: 每个区室的PWM列表
    - pwm_dir: PWM保存目录
    """
    print(f"\n{'=' * 60}")
    print(f"从聚类构建PWM矩阵（每个区室选择前{top_clusters_per_compartment}个聚类）")
    print(f"{'=' * 60}")

    pwm_dir = os.path.join(output_dir, 'clustered_pwm_matrices')
    os.makedirs(pwm_dir, exist_ok=True)

    compartment_pwms = {}
    compartment_consensus = {}
    compartment_ic = {}

    for comp_name, clusters in clustered_results.items():
        if not clusters:
            print(f"\n{comp_name}: 没有聚类数据，跳过")
            continue

        print(f"\n为{comp_name}构建PWM矩阵...")
        print(f"  聚类数量: {len(clusters)}")

        # 选择前top_clusters_per_compartment个聚类
        selected_clusters = clusters[:min(top_clusters_per_compartment, len(clusters))]

        pwms = []
        consensus_seqs = []
        ic_values = []

        for cluster_idx, cluster in enumerate(selected_clusters):
            print(f"  处理聚类 {cluster_idx + 1}: {cluster['size']} 个6-mer，平均分数: {cluster['avg_score']:.4f}")

            # 从聚类构建PWM
            pwm, consensus = create_pwm_from_cluster(cluster['sequences'], cluster['scores'])

            if pwm is not None:
                # 计算信息含量
                ic = calculate_information_content(pwm)

                pwms.append(pwm)
                consensus_seqs.append(consensus)
                ic_values.append(ic)

                print(f"    共识序列: {consensus}")
                print(f"    信息含量: {ic:.3f} bits")

                # 保存单个PWM矩阵
                pwm_file = os.path.join(pwm_dir, f'{comp_name}_cluster{cluster_idx + 1}_pwm.csv')
                pwm_df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])
                pwm_df.index.name = 'Position'
                pwm_df.to_csv(pwm_file, encoding='utf-8')

                # 保存聚类信息
                cluster_info_file = os.path.join(pwm_dir, f'{comp_name}_cluster{cluster_idx + 1}_info.txt')
                with open(cluster_info_file, 'w', encoding='utf-8') as f:
                    f.write(f"聚类 {cluster_idx + 1} - {comp_name}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"聚类大小: {cluster['size']}\n")
                    f.write(f"平均IG分数: {cluster['avg_score']:.4f}\n")
                    f.write(f"代表性序列: {cluster['representative_seq']}\n")
                    f.write(f"共识序列: {consensus}\n")
                    f.write(f"信息含量: {ic:.3f} bits\n")
                    f.write(f"聚类内相似性: {cluster['intra_similarity']:.3f}\n\n")
                    f.write("聚类中的序列:\n")
                    for i, seq in enumerate(cluster['sequences'][:20]):  # 只显示前20个
                        f.write(f"  {i + 1}. {seq}: {cluster['scores'][i]:.4f}\n")

        if pwms:
            compartment_pwms[comp_name] = pwms
            compartment_consensus[comp_name] = consensus_seqs
            compartment_ic[comp_name] = ic_values

    return compartment_pwms, compartment_consensus, compartment_ic, pwm_dir


# ===== 绘制Logo图 =====
def plot_motif_logos(compartment_pwms, compartment_consensus, output_dir):
    """
    为每个区室的每个PWM绘制logo图
    """
    print(f"\n{'=' * 60}")
    print("绘制motif logo图")
    print(f"{'=' * 60}")

    logo_dir = os.path.join(output_dir, 'motif_logos')
    os.makedirs(logo_dir, exist_ok=True)

    for comp_name, pwms in compartment_pwms.items():
        if not pwms:
            continue

        print(f"\n为{comp_name}绘制logo图...")

        for i, (pwm, consensus) in enumerate(zip(pwms, compartment_consensus[comp_name])):
            # 创建DataFrame
            pwm_df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])

            # 创建logo图
            plt.figure(figsize=(10, 3))

            # 创建logo对象
            logo = logomaker.Logo(pwm_df,
                                  font_name='Arial',
                                  color_scheme='classic',
                                  vpad=0.1,
                                  width=0.8)

            # 设置样式
            logo.style_spines(visible=False)
            logo.style_spines(spines=['left', 'bottom'], visible=True)
            logo.ax.set_xlabel('Position', fontsize=12)
            logo.ax.set_ylabel('Bits', fontsize=12)
            logo.ax.set_title(f'{comp_name} - Cluster {i + 1}\nConsensus: {consensus}', fontsize=14)
            logo.ax.set_xticks(range(6))
            logo.ax.set_xticklabels(range(1, 7))

            # 保存
            logo_path = os.path.join(logo_dir, f'{comp_name}_cluster{i + 1}_logo.png')
            plt.tight_layout()
            plt.savefig(logo_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  已保存: {logo_path}")

    print(f"\n所有logo图已保存到: {logo_dir}")


# ===== 创建MEME格式文件 =====
def create_meme_file_from_clusters(clustered_results, compartment_pwms, compartment_consensus, output_dir):
    """
    从聚类结果创建MEME格式文件
    每个聚类对应一个motif
    """
    print(f"\n{'=' * 60}")
    print("创建MEME格式文件（基于聚类）")
    print(f"{'=' * 60}")

    meme_dir = os.path.join(output_dir, 'meme_files')
    os.makedirs(meme_dir, exist_ok=True)

    background = [0.25, 0.25, 0.25, 0.25]

    meme_file = os.path.join(meme_dir, 'clustered_motifs.meme')

    with open(meme_file, 'w', encoding='utf-8') as f:
        # MEME文件头
        f.write("MEME version 5\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies:\n")
        f.write(f"A {background[0]:.3f} C {background[1]:.3f} G {background[2]:.3f} T {background[3]:.3f}\n\n")

        # 为每个区室的每个聚类写入motif
        motif_count = 0
        for comp_name, clusters in clustered_results.items():
            if comp_name not in compartment_pwms:
                continue

            pwms = compartment_pwms[comp_name]
            consensus_seqs = compartment_consensus.get(comp_name, [])

            for i, (cluster, pwm, consensus) in enumerate(zip(clusters[:len(pwms)], pwms, consensus_seqs)):
                motif_count += 1

                # Motif信息
                f.write(f"MOTIF {comp_name}_cluster{i + 1}\n")
                f.write(f"letter-probability matrix: alength= 4 w= {pwm.shape[0]} nsites= {cluster['size']} E= 0.0\n")
                f.write(f"# Consensus: {consensus}\n")
                f.write(f"# Cluster size: {cluster['size']}\n")
                f.write(f"# Avg score: {cluster['avg_score']:.4f}\n")

                # 写入PWM矩阵
                for pos in range(pwm.shape[0]):
                    f.write(f"  {pwm[pos, 0]:.6f}  {pwm[pos, 1]:.6f}  {pwm[pos, 2]:.6f}  {pwm[pos, 3]:.6f}\n")

                f.write("\n")

    print(f"已创建MEME文件: {meme_file}")
    print(f"共包含 {motif_count} 个motif（基于聚类）")

    return meme_file


# ===== 生成聚类分析报告 =====
def generate_clustering_report(clustered_results, output_dir):
    """
    生成聚类分析报告
    """
    print(f"\n{'=' * 60}")
    print("生成聚类分析报告")
    print(f"{'=' * 60}")

    report_file = os.path.join(output_dir, 'clustering_analysis_report.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("6-mer Clustering Analysis Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("SUMMARY:\n")
        f.write("-" * 30 + "\n")

        total_clusters = 0
        total_6mers = 0

        for comp_name, clusters in clustered_results.items():
            comp_clusters = len(clusters)
            comp_6mers = sum([c['size'] for c in clusters])

            total_clusters += comp_clusters
            total_6mers += comp_6mers

            f.write(f"{comp_name}:\n")
            f.write(f"  Clusters: {comp_clusters}\n")
            f.write(f"  6-mers: {comp_6mers}\n")

            if clusters:
                # 聚类统计
                cluster_sizes = [c['size'] for c in clusters]
                avg_sizes = [c['avg_score'] for c in clusters]

                f.write(f"  Avg cluster size: {np.mean(cluster_sizes):.1f}\n")
                f.write(f"  Max cluster size: {np.max(cluster_sizes)}\n")
                f.write(f"  Min cluster size: {np.min(cluster_sizes)}\n")
                f.write(f"  Avg cluster score: {np.mean(avg_sizes):.4f}\n\n")

                # 前3个聚类信息
                f.write("  Top 3 clusters:\n")
                for i, cluster in enumerate(clusters[:3]):
                    f.write(f"    Cluster {i + 1}:\n")
                    f.write(f"      Size: {cluster['size']}\n")
                    f.write(f"      Avg score: {cluster['avg_score']:.4f}\n")
                    f.write(f"      Representative: {cluster['representative_seq']}\n")
                    f.write(f"      Intra similarity: {cluster['intra_similarity']:.3f}\n")
            else:
                f.write("  No clusters found\n")

            f.write("\n")

        f.write(f"TOTAL:\n")
        f.write(f"  Total clusters: {total_clusters}\n")
        f.write(f"  Total 6-mers: {total_6mers}\n")
        if total_clusters > 0:
            f.write(f"  Avg 6-mers per cluster: {total_6mers / total_clusters:.1f}\n")
        else:
            f.write("  No clusters\n")

    print(f"聚类报告已保存到: {report_file}")


# ===== 添加清理检查点函数（可选） =====
def cleanup_checkpoints(output_dir, keep_last_n=3):
    """
    清理旧的检查点，只保留最近的几个

    参数:
    - output_dir: 输出目录
    - keep_last_n: 保留最近的几个检查点
    """
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return

    # 获取所有检查点文件
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith('batch_checkpoint') and file.endswith('.pkl'):
            file_path = os.path.join(checkpoint_dir, file)
            ctime = os.path.getctime(file_path)
            checkpoint_files.append((ctime, file_path))

    # 按创建时间排序
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)

    # 删除旧的检查点
    for i, (ctime, file_path) in enumerate(checkpoint_files):
        if i >= keep_last_n:
            try:
                os.remove(file_path)
                print(f"删除旧检查点: {os.path.basename(file_path)}")
            except:
                pass


# ===== 主函数 =====
def main():
    # 1. 参数与路径
    csv_path = os.path.join(base_dir, "dataset", "training_validation.csv")
    fasta_path = os.path.join(base_dir, "dataset", "training_validation_seqs")

    parser = argparse.ArgumentParser(description='6-mer motif analysis with clustering using integrated gradients')
    parser.add_argument('--input_fasta', default=fasta_path, help='training/validation fasta')
    parser.add_argument('--label_csv', default=csv_path, help='training/validation label csv')
    parser.add_argument('--output_path', default="6mer_clustering_results", help='output path')
    parser.add_argument('--device', default="cpu")  # 强制使用CPU
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--top_clusters_per_compartment', type=int, default=3,
                        help='Number of top clusters per compartment for PWM building')
    parser.add_argument('--max_6mers_per_compartment', type=int, default=50000,
                        help='每个区室最大保留的6-mer数量')
    parser.add_argument('--min_score_threshold', type=float, default=0.0001,
                        help='最小IG分数阈值')
    parser.add_argument('--model_fold', type=int, default=2, help='Model fold to use (2)')
    parser.add_argument('--debug_batches', type=int, default=None,
                        help='调试时处理的批次数量，None表示处理所有')
    # 新增参数
    parser.add_argument('--resume', action='store_true', help='从检查点恢复')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='检查点保存间隔（每处理多少批次保存一次）')
    parser.add_argument('--export_batch_results', action='store_true',
                        help='导出批次处理结果')

    opt = parser.parse_args()

    os.makedirs(opt.output_path, exist_ok=True)
    set_seed(42)

    # 设置设备
    device = torch.device(opt.device)
    print(f"使用设备: {device}")
    print(f"使用第{opt.model_fold}折模型")
    print(f"每个区室最多保留 {opt.max_6mers_per_compartment} 个6-mer")
    print(f"最小IG阈值: {opt.min_score_threshold}")
    print(f"检查点保存间隔: 每{opt.checkpoint_interval}个批次")
    print(f"从检查点恢复: {opt.resume}")

    # 2. 读取标签数据
    print("读取标签数据...")
    label_df = pd.read_csv(opt.label_csv)
    label_columns = label_df.columns[1:].tolist()
    labels_dict = {row[0]: row[1:].values.astype(np.float32)
                   for _, row in label_df.iterrows()}

    print(f"标签列 ({len(label_columns)}个细胞区室): {label_columns}")
    print(f"标签数量: {len(labels_dict)}")

    # 3. 读取序列数据
    def read_nucleotide_sequences(file):
        print(f"正在读取FASTA文件: {file}")
        if not os.path.exists(file):
            print(f'错误: 文件 {file} 不存在.')
            sys.exit(1)

        records = []
        for record in SeqIO.parse(file, "fasta"):
            seq_id = record.id
            sequence = str(record.seq).upper()
            sequence = re.sub('U', 'T', sequence)
            sequence = re.sub('[^ACGT]', '-', sequence)
            records.append([seq_id, sequence])

        print(f"成功读取 {len(records)} 条序列")
        return records

    fasta_records = read_nucleotide_sequences(opt.input_fasta)

    # 4. 加载模型
    print(f"\n加载第{opt.model_fold}折训练好的模型...")

    # 查找模型文件
    model_path = os.path.join("results", f'best_model_fold_{opt.model_fold}.pt')
    if not os.path.exists(model_path):
        # 尝试其他可能的模型文件
        import glob
        all_models = glob.glob(os.path.join("results", '*.pt'))
        if all_models:
            model_path = all_models[0]
            print(f"警告: 未找到第{opt.model_fold}折模型，使用找到的第一个模型: {model_path}")
        else:
            print("错误: 未找到任何训练好的模型文件")
            sys.exit(1)
    else:
        print(f"找到模型: {model_path}")

    # 加载状态字典
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        print("从checkpoint加载state_dict")

    print(f"状态字典键数量: {len(state_dict)}")

    # 5. 创建模型并加载权重
    print("\n创建完整模型...")
    model = SequenceModel(kmer_dim=5460, n_classes=len(label_columns)).to(device)

    # 获取模型当前的状态字典
    model_dict = model.state_dict()

    # 筛选可以加载的权重
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            pretrained_dict[k] = v

    # 更新模型字典
    if len(pretrained_dict) > 0:
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"成功加载 {len(pretrained_dict)} 个权重")
    else:
        print("警告: 未加载任何预训练权重，使用随机初始化模型")

    model.eval()
    print("模型准备完成")

    # 6. 加载特征数据
    print("\n加载特征数据...")

    # 加载k-mer特征（5460维）
    kmer_cache = os.path.join("results", 'train_kmer.pkl')
    if os.path.exists(kmer_cache):
        print(f"从缓存加载k-mer特征: {kmer_cache}")
        with open(kmer_cache, 'rb') as f:
            kmer_feat = pickle.load(f)
        print(f"成功加载 {len(kmer_feat)} 个序列的k-mer特征")
    else:
        print(f"错误: 未找到k-mer特征缓存文件 {kmer_cache}")
        sys.exit(1)

    # 加载CKSNAP特征
    cksnap_cache = os.path.join("results", 'train_cksnap.pkl')
    if os.path.exists(cksnap_cache):
        print(f"从缓存加载CKSNAP特征: {cksnap_cache}")
        with open(cksnap_cache, 'rb') as f:
            cksnap_feat = pickle.load(f)
        print(f"成功加载 {len(cksnap_feat)} 个序列的CKSNAP特征")
    else:
        print(f"错误: 未找到CKSNAP特征缓存文件 {cksnap_cache}")
        sys.exit(1)

    # 加载邻接矩阵
    adj_cache = os.path.join("results", 'train_adj.pkl')
    if os.path.exists(adj_cache):
        print(f"从缓存加载邻接矩阵: {adj_cache}")
        with open(adj_cache, 'rb') as f:
            adj_feat = pickle.load(f)
        print(f"成功加载 {len(adj_feat)} 个序列的邻接矩阵")
    else:
        # 如果没有邻接矩阵缓存，创建空矩阵
        print(f"未找到邻接矩阵缓存，创建空矩阵...")
        nucleotides = ['A', 'C', 'G', 'T']
        kmers = [''.join(combo) for combo in itertools.product(nucleotides, repeat=3)]
        n_kmers = len(kmers)

        adj_feat = {}
        for seq_id in kmer_feat.keys():
            adj_feat[seq_id] = np.zeros((n_kmers, n_kmers), dtype=np.float32)
        print(f"创建了 {len(adj_feat)} 个空邻接矩阵")

    # 7. 准备数据集
    print("\n准备数据集...")

    # 筛选有标签且特征完整的序列
    all_seqs = []
    for seq_id, seq in fasta_records:
        if (seq_id in labels_dict and
                seq_id in cksnap_feat and
                seq_id in kmer_feat and
                seq_id in adj_feat):
            all_seqs.append(seq_id)

    print(f"使用 {len(all_seqs)} 个序列进行分析")

    # 检查特征维度
    if all_seqs:
        sample_id = all_seqs[0]
        print(f"\n检查第一个样本的特征维度:")
        print(f"  cksnap特征长度: {len(cksnap_feat[sample_id])} (期望: 96)")
        print(f"  kmer特征长度: {len(kmer_feat[sample_id])} (期望: 5460)")
        print(f"  邻接矩阵形状: {adj_feat[sample_id].shape} (期望: (64, 64))")

    # 创建数据集
    dataset = SequenceDataset(all_seqs, cksnap_feat, kmer_feat, adj_feat, labels_dict)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    print(f"数据加载器准备完成: {len(dataset)} 个样本")

    # 8. 生成6-mer名称
    sixmer_names = generate_6mer_names()

    # 9. 提取和聚类6-mer（使用检查点版本）
    print(f"\n{'=' * 60}")
    print("开始提取和聚类6-mer（检查点版本）")
    print(f"{'=' * 60}")

    clustered_results, compartment_top_6mers = extract_and_cluster_6mers_by_compartment_optimized_with_checkpoint(
        model=model,
        dataloader=dataloader,
        label_columns=label_columns,
        device=device,
        sixmer_names=sixmer_names,
        output_dir=opt.output_path,
        max_6mers_per_compartment=opt.max_6mers_per_compartment,
        min_score_threshold=opt.min_score_threshold,
        max_batches=opt.debug_batches,  # 可以限制批次数量进行调试
        checkpoint_interval=opt.checkpoint_interval,
        resume_from_checkpoint=opt.resume
    )

    # 10. 如果启用，导出批次结果
    if opt.export_batch_results:
        batch_results_dir = export_batch_results(compartment_top_6mers, opt.output_path)
        print(f"\n批次结果已导出到: {batch_results_dir}")

    # 11. 从聚类构建PWM
    if any(len(clusters) > 0 for clusters in clustered_results.values()):
        # 保存原始6-mer数据
        save_original_6mers(compartment_top_6mers, opt.output_path)

        # 构建PWM
        compartment_pwms, compartment_consensus, compartment_ic, pwm_dir = build_pwm_from_clusters(
            clustered_results,
            opt.output_path,
            top_clusters_per_compartment=opt.top_clusters_per_compartment
        )

        # 12. 创建MEME格式文件
        meme_file = create_meme_file_from_clusters(
            clustered_results,
            compartment_pwms,
            compartment_consensus,
            opt.output_path
        )

        # 14. 生成聚类分析报告
        generate_clustering_report(clustered_results, opt.output_path)

        # 15. 输出总结
        print(f"\n{'=' * 60}")
        print("分析完成！")
        print(f"{'=' * 60}")
        print(f"\n主要输出文件:")
        print(f"  1. PWM矩阵: {pwm_dir}/")
        print(f"  2. MEME文件: {meme_file}")
        print(f"  3. 聚类报告: {opt.output_path}/clustering_analysis_report.txt")
        print(f"  4. 原始6-mer数据: {opt.output_path}/original_6mers/")

        print(f"\n每个区室的聚类统计:")
        for comp_name, clusters in clustered_results.items():
            if clusters:
                print(f"  {comp_name}: {len(clusters)} 个聚类")
                if comp_name in compartment_consensus and compartment_consensus[comp_name]:
                    for i, consensus in enumerate(compartment_consensus[comp_name][:3]):
                        print(f"    聚类 {i + 1}: 共识序列: {consensus}")
    else:
        print("错误: 未能提取到任何聚类")

    # 16. 清理旧的检查点文件（可选）
    if opt.resume:  # 只有在从检查点恢复后才清理
        cleanup_checkpoints(opt.output_path, keep_last_n=1)


if __name__ == "__main__":
    main()