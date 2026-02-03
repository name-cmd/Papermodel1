# -*- coding: utf-8 -*-
"""
流量生成模型评估模块 (Evaluation Module)

本模块提供流量生成质量的综合评估指标，支持：
1. 被其他模块 import 调用（训练/生成时实时记录）
2. 独立运行进行批量评估和报告生成

评估指标：
- 困惑度 (Perplexity): 衡量语言模型对序列的预测能力
- JS 散度 (Jensen-Shannon Divergence): 衡量生成流量与真实流量的分布差异
- 余弦相似度 (Cosine Similarity): 衡量嵌入空间中的语义相似性
- 统计保真度: 均值/标准差/分位数的相对误差
- 有效流比例: 生成流量的结构有效性

参考文献:
- Jensen-Shannon Divergence: Lin, J. (1991). Divergence measures based on the Shannon entropy.
- Perplexity: Jelinek et al. (1977). Perplexity—a measure of the difficulty of speech recognition tasks.
"""

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from datetime import datetime
import warnings

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 深度学习
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from scipy.spatial.distance import cosine

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 设置 matplotlib 样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# 1. 困惑度计算 (Perplexity)
# ============================================================================

def compute_perplexity(loss: float) -> float:
    """
    根据交叉熵损失计算困惑度。
    
    困惑度定义: PPL = exp(CrossEntropyLoss)
    
    参数:
        loss: 交叉熵损失值
        
    返回:
        perplexity: 困惑度值
        
    参考:
        Jelinek et al. (1977). Perplexity—a measure of the difficulty 
        of speech recognition tasks.
    """
    if loss < 0:
        warnings.warn(f"Loss 值为负 ({loss})，可能存在计算错误")
        return float('inf')
    
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float('inf')
    
    return perplexity


def compute_sequence_perplexity(
    model, 
    tokenizer, 
    sequences: List[str],
    device: torch.device = None,
    batch_size: int = 8
) -> Tuple[float, List[float]]:
    """
    计算一组序列的平均困惑度。
    
    参数:
        model: GPT-2 语言模型
        tokenizer: 对应的分词器
        sequences: 待评估的文本序列列表
        device: 计算设备
        batch_size: 批次大小
        
    返回:
        avg_perplexity: 平均困惑度
        per_sequence_ppl: 每个序列的困惑度列表
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)
    
    per_sequence_ppl = []
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            
            for seq in batch_seqs:
                inputs = tokenizer(
                    seq, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=1024
                ).to(device)
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss.item()
                num_tokens = inputs['input_ids'].shape[1]
                
                ppl = compute_perplexity(loss)
                per_sequence_ppl.append(ppl)
                
                total_loss += loss * num_tokens
                total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    avg_perplexity = compute_perplexity(avg_loss)
    
    return avg_perplexity, per_sequence_ppl


# ============================================================================
# 2. JS 散度计算 (Jensen-Shannon Divergence)
# ============================================================================

def compute_histogram(
    values: np.ndarray, 
    bins: int = 50,
    range_minmax: Tuple[float, float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算数值的直方图分布（归一化为概率分布）。
    
    参数:
        values: 数值数组
        bins: 直方图箱数
        range_minmax: 值域范围 (min, max)，若为 None 则自动计算
        
    返回:
        hist: 归一化的概率分布
        bin_edges: 箱边界
    """
    values = np.array(values, dtype=np.float64)
    values = values[~np.isnan(values)]  # 移除 NaN
    
    if len(values) == 0:
        return np.zeros(bins), np.linspace(0, 1, bins + 1)
    
    if range_minmax is None:
        range_minmax = (values.min(), values.max())
    
    # 防止 range 相等
    if range_minmax[0] == range_minmax[1]:
        range_minmax = (range_minmax[0] - 0.5, range_minmax[1] + 0.5)
    
    hist, bin_edges = np.histogram(values, bins=bins, range=range_minmax, density=True)
    
    # 归一化为概率分布（确保和为1）
    hist = hist / (hist.sum() + 1e-10)
    
    return hist, bin_edges


def compute_js_divergence(
    real_values: np.ndarray, 
    generated_values: np.ndarray,
    bins: int = 50
) -> float:
    """
    计算 Jensen-Shannon 散度。
    
    JS 散度是 KL 散度的对称版本，取值范围 [0, 1]（使用 log2 时）。
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), 其中 M = 0.5 * (P + Q)
    
    参数:
        real_values: 真实数据数值数组
        generated_values: 生成数据数值数组
        bins: 直方图箱数
        
    返回:
        js_divergence: JS 散度值 [0, 1]
        
    参考:
        Lin, J. (1991). Divergence measures based on the Shannon entropy.
        IEEE Transactions on Information Theory, 37(1), 145-151.
    """
    real_values = np.array(real_values, dtype=np.float64)
    generated_values = np.array(generated_values, dtype=np.float64)
    
    # 移除无效值
    real_values = real_values[~np.isnan(real_values) & ~np.isinf(real_values)]
    generated_values = generated_values[~np.isnan(generated_values) & ~np.isinf(generated_values)]
    
    if len(real_values) == 0 or len(generated_values) == 0:
        return 1.0  # 无数据时返回最大散度
    
    # 统一值域范围
    all_values = np.concatenate([real_values, generated_values])
    range_minmax = (all_values.min(), all_values.max())
    
    # 计算直方图
    p, _ = compute_histogram(real_values, bins=bins, range_minmax=range_minmax)
    q, _ = compute_histogram(generated_values, bins=bins, range_minmax=range_minmax)
    
    # 添加平滑项防止除零
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # 重新归一化
    p = p / p.sum()
    q = q / q.sum()
    
    # 计算中间分布 M
    m = 0.5 * (p + q)
    
    # 计算 JS 散度（使用 log2，结果在 [0, 1]）
    js_div = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)
    
    return float(js_div)


def compute_kl_divergence(
    real_values: np.ndarray, 
    generated_values: np.ndarray,
    bins: int = 50
) -> float:
    """
    计算 KL 散度 D_KL(P_real || P_generated)。
    
    注意: KL 散度是非对称的，这里计算的是真实分布到生成分布的 KL 散度。
    
    参数:
        real_values: 真实数据数值数组
        generated_values: 生成数据数值数组
        bins: 直方图箱数
        
    返回:
        kl_divergence: KL 散度值 [0, +∞)
    """
    real_values = np.array(real_values, dtype=np.float64)
    generated_values = np.array(generated_values, dtype=np.float64)
    
    real_values = real_values[~np.isnan(real_values) & ~np.isinf(real_values)]
    generated_values = generated_values[~np.isnan(generated_values) & ~np.isinf(generated_values)]
    
    if len(real_values) == 0 or len(generated_values) == 0:
        return float('inf')
    
    all_values = np.concatenate([real_values, generated_values])
    range_minmax = (all_values.min(), all_values.max())
    
    p, _ = compute_histogram(real_values, bins=bins, range_minmax=range_minmax)
    q, _ = compute_histogram(generated_values, bins=bins, range_minmax=range_minmax)
    
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()
    
    kl_div = entropy(p, q, base=2)
    
    return float(kl_div)


# ============================================================================
# 3. 余弦相似度计算 (Cosine Similarity)
# ============================================================================

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度。
    
    Cosine Similarity = (A · B) / (||A|| * ||B||)
    
    参数:
        vec1: 向量1
        vec2: 向量2
        
    返回:
        similarity: 余弦相似度 [-1, 1]
    """
    vec1 = np.array(vec1, dtype=np.float64).flatten()
    vec2 = np.array(vec2, dtype=np.float64).flatten()
    
    if len(vec1) != len(vec2):
        raise ValueError(f"向量维度不匹配: {len(vec1)} vs {len(vec2)}")
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = 1.0 - cosine(vec1, vec2)  # scipy.cosine 返回的是距离
    
    return float(similarity)


def compute_embedding_similarity(
    model,
    tokenizer,
    real_texts: List[str],
    generated_texts: List[str],
    device: torch.device = None,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    使用预训练模型计算真实文本和生成文本在嵌入空间中的相似度。
    
    方法:
    1. 提取所有文本的 [CLS] 或平均池化嵌入
    2. 计算真实文本嵌入的质心
    3. 计算生成文本嵌入的质心
    4. 计算质心间的余弦相似度
    5. 计算逐对相似度的统计量
    
    参数:
        model: 预训练模型（如 DistilBERT）
        tokenizer: 对应的分词器
        real_texts: 真实文本列表
        generated_texts: 生成文本列表
        device: 计算设备
        batch_size: 批次大小
        
    返回:
        包含多种相似度指标的字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)
    
    def get_embeddings(texts: List[str]) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(
                    batch, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=512,
                    padding=True
                ).to(device)
                
                outputs = model(**inputs, output_hidden_states=True)
                
                # 使用最后一层的 [CLS] token 或平均池化
                if hasattr(outputs, 'last_hidden_state'):
                    # 平均池化
                    hidden = outputs.last_hidden_state
                    mask = inputs['attention_mask'].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                else:
                    # 使用 pooler_output
                    pooled = outputs.pooler_output
                
                embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings)
    
    # 获取嵌入
    real_embeddings = get_embeddings(real_texts)
    gen_embeddings = get_embeddings(generated_texts)
    
    # 计算质心
    real_centroid = real_embeddings.mean(axis=0)
    gen_centroid = gen_embeddings.mean(axis=0)
    
    # 质心余弦相似度
    centroid_similarity = compute_cosine_similarity(real_centroid, gen_centroid)
    
    # 计算所有真实样本与生成质心的相似度
    real_to_gen_centroid = [
        compute_cosine_similarity(emb, gen_centroid) 
        for emb in real_embeddings
    ]
    
    # 计算所有生成样本与真实质心的相似度
    gen_to_real_centroid = [
        compute_cosine_similarity(emb, real_centroid) 
        for emb in gen_embeddings
    ]
    
    return {
        'centroid_cosine_similarity': centroid_similarity,
        'real_to_gen_centroid_mean': float(np.mean(real_to_gen_centroid)),
        'real_to_gen_centroid_std': float(np.std(real_to_gen_centroid)),
        'gen_to_real_centroid_mean': float(np.mean(gen_to_real_centroid)),
        'gen_to_real_centroid_std': float(np.std(gen_to_real_centroid)),
    }


# ============================================================================
# 4. 统计保真度指标 (Statistical Fidelity)
# ============================================================================

def compute_statistical_fidelity(
    real_values: np.ndarray, 
    generated_values: np.ndarray
) -> Dict[str, float]:
    """
    计算统计保真度指标。
    
    包括:
    - 均值相对误差 (Mean Relative Error)
    - 标准差相对误差 (Std Relative Error)
    - 各分位数的相对误差 (Quantile Relative Errors)
    
    参数:
        real_values: 真实数据数值数组
        generated_values: 生成数据数值数组
        
    返回:
        包含各统计指标相对误差的字典
    """
    real_values = np.array(real_values, dtype=np.float64)
    generated_values = np.array(generated_values, dtype=np.float64)
    
    real_values = real_values[~np.isnan(real_values) & ~np.isinf(real_values)]
    generated_values = generated_values[~np.isnan(generated_values) & ~np.isinf(generated_values)]
    
    if len(real_values) == 0 or len(generated_values) == 0:
        return {
            'mean_relative_error': float('inf'),
            'std_relative_error': float('inf'),
            'median_relative_error': float('inf'),
            'q25_relative_error': float('inf'),
            'q75_relative_error': float('inf'),
            'q99_relative_error': float('inf'),
        }
    
    def relative_error(real_val, gen_val):
        if abs(real_val) < 1e-10:
            return abs(gen_val - real_val)
        return abs(gen_val - real_val) / abs(real_val)
    
    # 基本统计量
    real_mean = np.mean(real_values)
    gen_mean = np.mean(generated_values)
    real_std = np.std(real_values)
    gen_std = np.std(generated_values)
    
    # 分位数
    quantiles = [0.25, 0.5, 0.75, 0.99]
    real_quantiles = np.quantile(real_values, quantiles)
    gen_quantiles = np.quantile(generated_values, quantiles)
    
    return {
        'real_mean': float(real_mean),
        'generated_mean': float(gen_mean),
        'mean_relative_error': relative_error(real_mean, gen_mean),
        'real_std': float(real_std),
        'generated_std': float(gen_std),
        'std_relative_error': relative_error(real_std, gen_std),
        'q25_relative_error': relative_error(real_quantiles[0], gen_quantiles[0]),
        'median_relative_error': relative_error(real_quantiles[1], gen_quantiles[1]),
        'q75_relative_error': relative_error(real_quantiles[2], gen_quantiles[2]),
        'q99_relative_error': relative_error(real_quantiles[3], gen_quantiles[3]),
    }


# ============================================================================
# 5. 有效流比例 (Valid Flow Ratio)
# ============================================================================

def compute_valid_flow_ratio(
    generated_flows: List[str],
    expected_field_count: int,
    tolerance: float = 0.3
) -> Dict[str, float]:
    """
    计算生成流量的结构有效性。
    
    有效性判断标准:
    1. 字段数量在预期范围内
    2. 数值字段可解析
    
    参数:
        generated_flows: 生成的流文本列表
        expected_field_count: 预期的字段数量
        tolerance: 字段数量容差比例
        
    返回:
        包含有效性统计的字典
    """
    if not generated_flows:
        return {
            'total_flows': 0,
            'valid_flows': 0,
            'valid_ratio': 0.0,
            'avg_field_count': 0.0,
            'field_count_std': 0.0,
        }
    
    valid_count = 0
    field_counts = []
    
    min_fields = int(expected_field_count * (1 - tolerance))
    max_fields = int(expected_field_count * (1 + tolerance))
    
    for flow in generated_flows:
        fields = flow.strip().split()
        field_count = len(fields)
        field_counts.append(field_count)
        
        # 检查字段数量
        if min_fields <= field_count <= max_fields:
            # 检查数值可解析性
            parseable = True
            for field in fields:
                try:
                    float(field)
                except ValueError:
                    # 允许某些非数值字段
                    if field not in ['<bos>', '<eos>', '<pad>']:
                        parseable = False
                        break
            
            if parseable:
                valid_count += 1
    
    return {
        'total_flows': len(generated_flows),
        'valid_flows': valid_count,
        'valid_ratio': valid_count / len(generated_flows),
        'avg_field_count': float(np.mean(field_counts)),
        'field_count_std': float(np.std(field_counts)),
        'expected_field_count': expected_field_count,
    }


# ============================================================================
# 6. 综合评估报告类
# ============================================================================

class EvaluationReport:
    """
    综合评估报告生成器。
    
    用于收集、汇总和导出所有评估指标。
    """
    
    def __init__(self, output_dir: str = './evaluation_results'):
        """
        初始化评估报告。
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.metrics = {
            'meta': {
                'timestamp': datetime.now().isoformat(),
                'random_seed': RANDOM_SEED,
            },
            'perplexity': {},
            'js_divergence': {},
            'kl_divergence': {},
            'cosine_similarity': {},
            'statistical_fidelity': {},
            'validity': {},
        }
        
        self.feature_distributions = {}
    
    def add_perplexity(self, name: str, value: float, details: Dict = None):
        """添加困惑度指标"""
        self.metrics['perplexity'][name] = {
            'value': value,
            'details': details or {}
        }
    
    def add_divergence(
        self, 
        feature_name: str, 
        real_values: np.ndarray, 
        generated_values: np.ndarray,
        bins: int = 50
    ):
        """计算并添加散度指标"""
        js_div = compute_js_divergence(real_values, generated_values, bins)
        kl_div = compute_kl_divergence(real_values, generated_values, bins)
        stat_fidelity = compute_statistical_fidelity(real_values, generated_values)
        
        self.metrics['js_divergence'][feature_name] = js_div
        self.metrics['kl_divergence'][feature_name] = kl_div
        self.metrics['statistical_fidelity'][feature_name] = stat_fidelity
        
        # 保存分布数据用于绘图
        self.feature_distributions[feature_name] = {
            'real': real_values,
            'generated': generated_values,
        }
    
    def add_cosine_similarity(self, metrics: Dict[str, float]):
        """添加余弦相似度指标"""
        self.metrics['cosine_similarity'] = metrics
    
    def add_validity(self, metrics: Dict[str, float]):
        """添加有效性指标"""
        self.metrics['validity'] = metrics
    
    def compute_summary(self) -> Dict[str, float]:
        """计算汇总指标"""
        summary = {}
        
        # 平均 JS 散度
        js_values = list(self.metrics['js_divergence'].values())
        if js_values:
            summary['avg_js_divergence'] = float(np.mean(js_values))
            summary['max_js_divergence'] = float(np.max(js_values))
            summary['min_js_divergence'] = float(np.min(js_values))
        
        # 平均统计误差
        stat_errors = []
        for feat, stats in self.metrics['statistical_fidelity'].items():
            if 'mean_relative_error' in stats:
                stat_errors.append(stats['mean_relative_error'])
        if stat_errors:
            summary['avg_mean_relative_error'] = float(np.mean(stat_errors))
        
        # 有效率
        if self.metrics['validity']:
            summary['valid_ratio'] = self.metrics['validity'].get('valid_ratio', 0.0)
        
        # 质心余弦相似度
        if self.metrics['cosine_similarity']:
            summary['centroid_cosine_similarity'] = self.metrics['cosine_similarity'].get(
                'centroid_cosine_similarity', 0.0
            )
        
        self.metrics['summary'] = summary
        return summary
    
    def save_report(self, filename: str = 'evaluation_report.json'):
        """保存评估报告为 JSON 文件"""
        self.compute_summary()
        
        # 将 numpy 类型转换为 Python 原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        serializable_metrics = convert_to_serializable(self.metrics)
        
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 评估报告已保存至: {report_path}")
        return report_path
    
    def plot_distributions(self, n_cols: int = 3, n_rows: int = 5):
        """
        绘制所有特征的累积分布函数 (CDF) 对比图。
        不使用分箱 (Binning)，直接基于原始数据绘制，以展示更精细的分布差异。
        
        参数:
            n_cols: 每行的子图数量，默认为 3
            n_rows: 总行数，默认为 5（共绘制 15 个特征）
        """
        if not self.feature_distributions:
            print("⚠️ 无分布数据可绘制")
            return
        
        # 按 JS 散度排序（从大到小），获取所有特征
        sorted_features = sorted(
            self.metrics['js_divergence'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        n_features = len(sorted_features)
        total_slots = n_cols * n_rows
        
        # 如果特征数量超过预设的 5x3=15 个，自动扩展行数
        if n_features > total_slots:
            n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (feature_name, js_div) in enumerate(sorted_features):
            if idx >= len(axes):
                break
            ax = axes[idx]
            
            real_data = np.sort(self.feature_distributions[feature_name]['real'])
            gen_data = np.sort(self.feature_distributions[feature_name]['generated'])
            
            # 计算 CDF
            real_y = np.arange(1, len(real_data) + 1) / len(real_data)
            gen_y = np.arange(1, len(gen_data) + 1) / len(gen_data)
            
            # 绘制 CDF
            ax.plot(real_data, real_y, label='Real', color='blue', linewidth=2)
            ax.plot(gen_data, gen_y, label='Generated', color='darkorange', linewidth=2, linestyle='--')
            
            ax.set_title(f'{feature_name}\nJS Divergence: {js_div:.4f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('CDF')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature CDF Comparison (Real vs Generated)', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'distribution_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 分布对比图 (CDF) 已保存至: {plot_path}")
        return plot_path
    
    def plot_divergence_heatmap(self):
        """绘制所有特征的散度热力图"""
        if not self.metrics['js_divergence']:
            print("⚠️ 无散度数据可绘制")
            return
        
        features = list(self.metrics['js_divergence'].keys())
        js_values = [self.metrics['js_divergence'][f] for f in features]
        
        # 创建数据框
        df = pd.DataFrame({
            'Feature': features,
            'JS Divergence': js_values
        })
        df = df.sort_values('JS Divergence', ascending=True)
        
        # 绘制水平条形图
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))
        bars = ax.barh(df['Feature'], df['JS Divergence'], color=colors)
        
        # 添加数值标签
        for bar, val in zip(bars, df['JS Divergence']):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)
        
        ax.set_xlabel('JS Divergence')
        ax.set_title('Jensen-Shannon Divergence by Feature\n(Lower is Better)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlim(0, max(js_values) * 1.2)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'divergence_summary.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 散度汇总图已保存至: {plot_path}")
        return plot_path
    
    def export_distributions_to_csv(self):
        """
        将特征分布数据导出为 CSV 文件，供 Origin 等绑图软件使用。
        
        导出三个文件：
        1. distribution_raw_data.csv: 每个特征的原始数值（Real 和 Generated）
        2. distribution_histogram.csv: 每个特征的直方图分箱数据
        3. distribution_summary.csv: 各特征的散度和统计汇总
        """
        if not self.feature_distributions:
            print("⚠️ 无分布数据可导出")
            return
        
        # === 1. 导出原始数据 ===
        raw_data_path = os.path.join(self.output_dir, 'distribution_raw_data.csv')
        
        # 找到最大长度
        max_len = 0
        for fname, data in self.feature_distributions.items():
            max_len = max(max_len, len(data['real']), len(data['generated']))
        
        # 构建 DataFrame
        raw_df_dict = {}
        for fname, data in self.feature_distributions.items():
            real_vals = list(data['real']) + [np.nan] * (max_len - len(data['real']))
            gen_vals = list(data['generated']) + [np.nan] * (max_len - len(data['generated']))
            raw_df_dict[f'{fname}_Real'] = real_vals
            raw_df_dict[f'{fname}_Generated'] = gen_vals
        
        raw_df = pd.DataFrame(raw_df_dict)
        raw_df.to_csv(raw_data_path, index=False, encoding='utf-8-sig')
        print(f"✅ 原始分布数据已导出: {raw_data_path}")
        
        # === 2. (已移除) 导出直方图数据 ===
        # 根据用户要求移除分箱操作
        hist_data_path = None
        
        # === 3. 导出汇总统计表 ===
        summary_path = os.path.join(self.output_dir, 'distribution_summary.csv')
        
        summary_records = []
        for fname in self.feature_distributions.keys():
            js_div = self.metrics['js_divergence'].get(fname, np.nan)
            kl_div = self.metrics['kl_divergence'].get(fname, np.nan)
            stat = self.metrics['statistical_fidelity'].get(fname, {})
            
            summary_records.append({
                'Feature': fname,
                'JS_Divergence': js_div,
                'KL_Divergence': kl_div,
                'Real_Mean': stat.get('real_mean', np.nan),
                'Generated_Mean': stat.get('generated_mean', np.nan),
                'Mean_Relative_Error': stat.get('mean_relative_error', np.nan),
                'Real_Std': stat.get('real_std', np.nan),
                'Generated_Std': stat.get('generated_std', np.nan),
                'Std_Relative_Error': stat.get('std_relative_error', np.nan),
                'Q25_Relative_Error': stat.get('q25_relative_error', np.nan),
                'Median_Relative_Error': stat.get('median_relative_error', np.nan),
                'Q75_Relative_Error': stat.get('q75_relative_error', np.nan),
                'Q99_Relative_Error': stat.get('q99_relative_error', np.nan),
            })
        
        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"✅ 汇总统计表已导出: {summary_path}")
        
        return raw_data_path, None, summary_path

    def print_summary(self):
        """打印评估摘要"""
        summary = self.compute_summary()
        
        print("\n" + "=" * 60)
        print(" " * 18 + "评估结果摘要")
        print("=" * 60)
        
        # 困惑度
        if self.metrics['perplexity']:
            print("\n📊 困惑度 (Perplexity):")
            for name, data in self.metrics['perplexity'].items():
                print(f"   {name}: {data['value']:.4f}")
        
        # JS 散度
        if self.metrics['js_divergence']:
            js_values = list(self.metrics['js_divergence'].values())
            print(f"\n📊 JS 散度 (Jensen-Shannon Divergence):")
            print(f"   平均: {np.mean(js_values):.4f}")
            print(f"   最大: {np.max(js_values):.4f} ({max(self.metrics['js_divergence'], key=self.metrics['js_divergence'].get)})")
            print(f"   最小: {np.min(js_values):.4f} ({min(self.metrics['js_divergence'], key=self.metrics['js_divergence'].get)})")
        
        # 余弦相似度
        if self.metrics['cosine_similarity']:
            print(f"\n📊 余弦相似度 (Cosine Similarity):")
            cs = self.metrics['cosine_similarity']
            if 'centroid_cosine_similarity' in cs:
                print(f"   质心相似度: {cs['centroid_cosine_similarity']:.4f}")
            if 'gen_to_real_centroid_mean' in cs:
                print(f"   生成→真实质心: {cs['gen_to_real_centroid_mean']:.4f} ± {cs['gen_to_real_centroid_std']:.4f}")
        
        # 有效性
        if self.metrics['validity']:
            v = self.metrics['validity']
            print(f"\n📊 有效性 (Validity):")
            print(f"   有效流比例: {v.get('valid_ratio', 0):.2%} ({v.get('valid_flows', 0)}/{v.get('total_flows', 0)})")
            print(f"   平均字段数: {v.get('avg_field_count', 0):.1f} (预期: {v.get('expected_field_count', 0)})")
        
        print("\n" + "=" * 60)


# ============================================================================
# 7. 流量解析工具函数
# ============================================================================

def parse_flow_file(
    file_path: str,
    feature_names: List[str] = None
) -> Tuple[List[str], Dict[str, List[float]]]:
    """
    解析流量文件，提取流文本和特征值。
    
    参数:
        file_path: 流量文件路径
        feature_names: 特征名列表
        
    返回:
        flows: 流文本列表
        features: {特征名: 数值列表} 字典
    """
    flows = []
    features = defaultdict(list)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line in ['<bos>', '<eos>']:
                continue
            if line.startswith('['):  # 跳过注释行
                continue
            
            flows.append(line)
            
            # 解析特征值
            if feature_names:
                fields = line.split()
                for i, fname in enumerate(feature_names):
                    if i < len(fields):
                        try:
                            features[fname].append(float(fields[i]))
                        except ValueError:
                            pass
    
    # 转换为 numpy 数组
    features = {k: np.array(v) for k, v in features.items()}
    
    return flows, features


def parse_generated_output(
    file_path: str,
    feature_names: List[str] = None
) -> Tuple[List[str], Dict[str, List[float]]]:
    """
    解析生成器输出文件。
    
    参数:
        file_path: 生成结果文件路径
        feature_names: 特征名列表
        
    返回:
        flows: 生成的流文本列表
        features: {特征名: 数值列表} 字典
    """
    flows = []
    features = defaultdict(list)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 跳过空行、会话头、注释
            if not line:
                continue
            if line.startswith('=') or line.startswith('[') or line.startswith('Session'):
                continue
            if '<bos>' in line or '<eos>' in line:
                continue
            
            # 这是一条流记录
            flows.append(line)
            
            if feature_names:
                fields = line.split()
                for i, fname in enumerate(feature_names):
                    if i < len(fields):
                        try:
                            features[fname].append(float(fields[i]))
                        except ValueError:
                            pass
    
    features = {k: np.array(v) for k, v in features.items()}
    
    return flows, features


# ============================================================================
# 8. 主函数：独立运行模式
# ============================================================================

# ============================================================================
# 配置常量（与 generator.py / distilbert_training.py 保持一致）
# ============================================================================
DEFAULT_DATASET = 'NF-UNSW-NB15-v3'  # 默认数据集名称
PROCESSED_DATA_BASE_DIR = './processed_data'  # 预处理数据基础目录
GENERATED_DATA_DIR = './generated_data'  # 生成数据目录
DEFAULT_BERT_MODEL_PATH = './models/distilbert-finetuned'  # DistilBERT 模型路径


def main():
    """独立运行评估模块"""
    parser = argparse.ArgumentParser(description='流量生成质量评估模块')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                       help=f'数据集名称（例如: NF-UNSW-NB15-v3），默认: {DEFAULT_DATASET}')
    parser.add_argument('--real_data', type=str, default=None,
                       help='真实流量数据文件路径（可选，默认根据数据集自动推断: ./processed_data/{dataset}/test_gpt2_input.txt）')
    parser.add_argument('--generated_data', type=str, default=None,
                       help='生成流量数据文件路径（可选，默认根据数据集自动推断: ./generated_data/generated_flows_only_{dataset}.txt）')
    parser.add_argument('--feature_config', type=str, default=None,
                       help='特征配置文件路径（可选，默认根据数据集自动推断: ./processed_data/{dataset}/feature_columns.json）')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='评估结果输出目录')
    parser.add_argument('--bert_model', type=str, default=None,
                       help=f'DistilBERT 模型路径（用于余弦相似度计算），默认: {DEFAULT_BERT_MODEL_PATH}')
    parser.add_argument('--skip_cosine', action='store_true',
                       help='跳过余弦相似度计算（节省时间）')
    
    args = parser.parse_args()
    
    # 获取数据集名称
    dataset_name = args.dataset
    
    print("\n" + "=" * 60)
    print(" " * 15 + "流量生成质量评估")
    print("=" * 60)
    print(f"数据集: {dataset_name}")
    
    # 根据数据集名称自动推断路径（与 generator.py / distilbert_training.py 一致）
    dataset_dir = os.path.join(PROCESSED_DATA_BASE_DIR, dataset_name)
    
    # 特征配置文件路径
    if args.feature_config:
        feature_config_path = args.feature_config
    else:
        feature_config_path = os.path.join(dataset_dir, 'feature_columns.json')
    
    # 真实数据文件路径
    if args.real_data:
        real_data_path = args.real_data
    else:
        real_data_path = os.path.join(dataset_dir, 'test_gpt2_input.txt')
    
    # 生成数据文件路径
    if args.generated_data:
        generated_data_path = args.generated_data
    else:
        generated_data_path = os.path.join(GENERATED_DATA_DIR, f'generated_flows_only_{dataset_name}.txt')
    
    # DistilBERT 模型路径
    bert_model_path = args.bert_model if args.bert_model else DEFAULT_BERT_MODEL_PATH
    
    # 加载特征配置
    print("\n📁 加载配置...")
    print(f"   特征配置文件: {feature_config_path}")
    if not os.path.exists(feature_config_path):
        print(f"❌ 特征配置文件不存在: {feature_config_path}")
        print(f"   提示: 请确认数据集 '{dataset_name}' 已完成预处理，")
        print(f"         或使用 --feature_config 参数指定特征配置文件路径")
        return
    
    with open(feature_config_path, 'r', encoding='utf-8') as f:
        feature_cfg = json.load(f)
    
    feature_names = feature_cfg.get('flow_text_features', [])
    print(f"   特征数量: {len(feature_names)}")
    print(f"   特征列表: {feature_names}")
    
    # 解析真实数据
    print(f"\n📁 解析真实数据: {real_data_path}")
    if not os.path.exists(real_data_path):
        print(f"❌ 真实数据文件不存在: {real_data_path}")
        print(f"   提示: 请确认数据集 '{dataset_name}' 已完成预处理，")
        print(f"         或使用 --real_data 参数指定真实数据文件路径")
        return
    
    real_flows, real_features = parse_flow_file(real_data_path, feature_names)
    print(f"   解析流数量: {len(real_flows)}")
    
    # 解析生成数据
    print(f"\n📁 解析生成数据: {generated_data_path}")
    if not os.path.exists(generated_data_path):
        print(f"❌ 生成数据文件不存在: {generated_data_path}")
        print(f"   提示: 请先运行 generator.py 生成流量数据，")
        print(f"         或使用 --generated_data 参数指定生成数据文件路径")
        return
    
    gen_flows, gen_features = parse_generated_output(generated_data_path, feature_names)
    print(f"   解析流数量: {len(gen_flows)}")
    
    if len(gen_flows) == 0:
        print("❌ 未解析到任何生成流量，请检查文件格式")
        return
    
    # 创建评估报告
    report = EvaluationReport(output_dir=args.output_dir)
    
    # 计算各特征的散度
    print("\n📊 计算分布散度...")
    for fname in feature_names:
        if fname in real_features and fname in gen_features:
            if len(real_features[fname]) > 0 and len(gen_features[fname]) > 0:
                report.add_divergence(fname, real_features[fname], gen_features[fname])
                print(f"   {fname}: JS={report.metrics['js_divergence'][fname]:.4f}")
    
    # 计算有效性
    print("\n📊 计算有效性...")
    validity = compute_valid_flow_ratio(gen_flows, len(feature_names))
    report.add_validity(validity)
    print(f"   有效流比例: {validity['valid_ratio']:.2%}")
    
    # 计算余弦相似度（可选）
    if not args.skip_cosine and os.path.exists(bert_model_path):
        print("\n📊 计算余弦相似度...")
        try:
            from transformers import AutoTokenizer, AutoModel
            
            tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
            model = AutoModel.from_pretrained(bert_model_path)
            
            # 采样以加速（如果数据量太大）
            sample_size = min(500, len(real_flows), len(gen_flows))
            sampled_real = np.random.choice(real_flows, sample_size, replace=False).tolist()
            sampled_gen = np.random.choice(gen_flows, sample_size, replace=False).tolist()
            
            cos_metrics = compute_embedding_similarity(
                model, tokenizer, sampled_real, sampled_gen
            )
            report.add_cosine_similarity(cos_metrics)
            print(f"   质心余弦相似度: {cos_metrics['centroid_cosine_similarity']:.4f}")
        except Exception as e:
            print(f"   ⚠️ 余弦相似度计算失败: {e}")
    else:
        if args.skip_cosine:
            print("\n⏭️ 跳过余弦相似度计算")
        else:
            print(f"\n⚠️ DistilBERT 模型不存在，跳过余弦相似度计算: {bert_model_path}")
    
    # 生成报告和图表
    print("\n📊 生成评估报告...")
    report.save_report()
    report.plot_distributions()
    report.plot_divergence_heatmap()
    
    # 导出 CSV 数据供 Origin 等绘图软件使用
    print("\n📊 导出 CSV 数据...")
    report.export_distributions_to_csv()
    
    report.print_summary()
    
    print(f"\n✅ 评估完成！结果保存在: {args.output_dir}")
    print(f"   - distribution_raw_data.csv: 原始数据（供 Origin 绘制分布曲线）")
    # print(f"   - distribution_histogram.csv: 直方图数据（已移除）")
    print(f"   - distribution_summary.csv: 散度汇总表")


if __name__ == "__main__":
    main()

