import os
# 静默 tokenizer 并行分叉告警
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 进度条显示优化：使用 ASCII 字符模式，固定宽度，提高终端兼容性
# 解决屏幕切换、黑屏等场景下进度条乱码、不显示、循环大量显示等问题
os.environ["TQDM_ASCII"] = "true"
os.environ["TQDM_NCOLS"] = "100"
import argparse
import json
import time
from pathlib import Path
import shutil
import csv # Added for saving training log

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed
)

import matplotlib.pyplot as plt
import seaborn as sns

# 配置（与 notebooks 思路对齐，但保留 BERT/DistilBERT）
MODEL_PATH = "./models/base/distilbert"  # 预训练 DistilBERT 目录（已在项目中）
BERT_BASE_DIR = "./processed_data/bert"  # (已废弃，保留用于兼容)
PROCESSED_DATA_BASE_DIR = "./processed_data"  # 预处理数据基础目录
# 根据 mode 自动选择子目录：binary -> bert/binary, multiclass -> bert/multiclass
OUTPUT_DIR_BINARY = "./models/distilbert-finetuned"  # 二分类微调输出
OUTPUT_DIR_MULTICLASS = "./models/distilbert-finetuned-multiclass"  # 多分类微调输出
# 训练模式：binary 或 multiclass（作为信息提示与一致性校验）
DEFAULT_MODE = "binary"
DEFAULT_DATASET = "NF-CICIDS2018-v3"  # 默认数据集名称/NF-CICIDS2018-v3
NUM_EPOCHS = 6
RANDOM_SEED = 42

# --- 调试/采样配置 ---
# 将 DEBUG_SAMPLE_RATIO 设为 < 1.0 即可只用部分样本，便于快速调试跑通流程。
# 正式实验时，请改回 1.0。
DEBUG_SAMPLE_RATIO = 0.4  # 例如 0.1 表示只用 10% 的样本


def get_bert_dir(dataset_name: str, mode: str) -> str:
    """根据数据集名称和训练模式返回对应的数据目录"""
    return os.path.join(PROCESSED_DATA_BASE_DIR, dataset_name, 'bert', mode)


def get_output_dir(mode: str) -> str:
    """根据训练模式返回对应的输出目录"""
    if mode == 'binary':
        return OUTPUT_DIR_BINARY
    else:
        return OUTPUT_DIR_MULTICLASS

class PacketDataset(Dataset):
    """将预处理后的 'text' 拼接特征送入 BERT 的数据集封装"""
    def __init__(self, tokenizer, texts, labels, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DetailedLoggingCallback(TrainerCallback):
    """记录每个 epoch 的训练与验证度量，保证指标按轮次变化。
    
    使用 flush=True 确保日志及时输出，避免缓冲区问题。
    """
    def __init__(self):
        self.epoch_start_time = None
        self.train_start_time = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.val_f1s = []
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        self.total_epochs = int(args.num_train_epochs)
        print("\n" + "="*80, flush=True)
        print("开始训练", flush=True)
        print("="*80, flush=True)
        print(f"配置: epochs={args.num_train_epochs} | batch={args.per_device_train_batch_size} | lr={args.learning_rate} | scheduler={args.lr_scheduler_type}", flush=True)
        print(f"总步数: {state.max_steps}", flush=True)
        print("-" * 80, flush=True)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        # 避免 int(state.epoch)==0 导致连续打印 0；使用本地计数器统一递增
        self.current_epoch += 1
        self.epoch_start_time = time.time()
        print(f"\n📍 Epoch {self.current_epoch}/{self.total_epochs} 开始...", flush=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 降低日志频率：不在每个 step 打印
        return
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and self.epoch_start_time:
            epoch_duration = time.time() - self.epoch_start_time
            total_elapsed = time.time() - self.train_start_time if self.train_start_time else 0
            val_loss = float(metrics.get('eval_loss', 0))
            val_acc = float(metrics.get('eval_accuracy', 0))
            val_f1 = float(metrics.get('eval_f1', 0))
            val_prec = float(metrics.get('eval_precision', 0))
            val_rec = float(metrics.get('eval_recall', 0))
            # 从 state.log_history 找最近的训练 loss
            train_loss = next((float(log['loss']) for log in reversed(state.log_history) if 'loss' in log), None)
            if train_loss is None:
                train_loss = float('nan')
            self.epochs.append(self.current_epoch)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.val_f1s.append(val_f1)
            # 计算预计剩余时间
            if self.current_epoch > 0:
                avg_epoch_time = total_elapsed / self.current_epoch
                eta = avg_epoch_time * (self.total_epochs - self.current_epoch)
            else:
                eta = 0
            print(f"  Train Loss: {train_loss:.4f}", flush=True)
            print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}", flush=True)
            print(f"  Precision: {val_prec:.4f} | Recall: {val_rec:.4f}", flush=True)
            print(f"  ⏱️  本轮耗时: {epoch_duration:.1f}s | 总已用时: {total_elapsed/60:.1f}min | 预计剩余: {eta/60:.1f}min", flush=True)
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start_time if self.train_start_time else 0
        best_val_loss = min(self.val_losses) if self.val_losses else float('nan')
        best_val_f1 = max(self.val_f1s) if self.val_f1s else float('nan')
        print("\n" + "="*80, flush=True)
        print(f"✓ 训练完成 | 总耗时: {total_time/60:.1f}min | 最佳: Val Loss={best_val_loss:.4f}, F1={best_val_f1:.4f}", flush=True)
        print("="*80, flush=True)
        
        # Save to CSV
        csv_path = os.path.join(args.output_dir, 'training_log.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_f1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(self.epochs)):
                writer.writerow({
                    'epoch': self.epochs[i],
                    'train_loss': self.train_losses[i],
                    'val_loss': self.val_losses[i],
                    'val_acc': self.val_accs[i],
                    'val_f1': self.val_f1s[i]
                })
        print(f"✅ 训练日志已保存至: {csv_path}", flush=True)
    
    def get_plot_data(self):
        return {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'val_f1s': self.val_f1s
        }
class FocalLoss(nn.Module):
    """多分类 Focal Loss。
    参数:
      gamma: 聚焦参数，越大越关注难样本（默认 2.0）
      alpha: 可选的每类权重张量，shape=[num_labels]；为 None 时不使用类权重
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        # 使用 register_buffer 确保 alpha 随模型自动迁移到正确设备（CPU/GPU）
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-ce)  # 预测为真类的概率
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            # alpha 通过 register_buffer 注册，随模型自动迁移设备
            # 此处仅做索引取权重，不修改 self.alpha
            alpha_weight = self.alpha[targets]
            focal = alpha_weight * focal
        if self.reduction == 'mean':
            return focal.mean()
        if self.reduction == 'sum':
            return focal.sum()
        return focal


def _ensure_int_labels(series: pd.Series) -> np.ndarray:
    """将 label 列转为 int numpy 数组（若存在脏数据则尽量容错）。"""
    # 注意：训练/评估都依赖 label 为从 0 开始的非负整数
    return pd.to_numeric(series, errors='coerce').fillna(0).astype(int).values


def _infer_num_labels_from_labels(all_labels: np.ndarray) -> int:
    """更稳健地推断 num_labels：
    - 若标签是非负整数：用 max+1（即使中间有缺失 id 也不会维度不匹配）
    - 否则：退化为 unique 数量
    """
    if all_labels.size == 0:
        return 0
    try:
        min_v = int(np.min(all_labels))
        max_v = int(np.max(all_labels))
        if min_v >= 0:
            return int(max_v + 1)
    except Exception:
        pass
    return int(np.unique(all_labels).size)


def load_bert_datasets(bert_dir: str, mode: str = 'binary'):
    """加载 processed_data/bert 下的 train/val/test_fusion.csv，返回 DataFrame 及基本统计。

    参数:
      bert_dir: processed_data/bert 目录
      mode: 'binary' 或 'multiclass'

    约定:
      - binary: 统一映射为 0=正常, 1=攻击（任何非0都视为攻击）
      - multiclass: 保留原始多类 label（应为从 0 开始的整数 id）
    """
    def read_csv(path):
        try:
            df = pd.read_csv(path)
            print(f"  ✓ {path}: {len(df)} 行")
            return df
        except Exception as e:
            print(f"  ✗ {path}: {e}")
            return None
    print(f"\n加载数据集...")
    train_df = read_csv(f"{bert_dir}/train_fusion.csv")
    val_df = read_csv(f"{bert_dir}/val_fusion.csv")
    test_df = read_csv(f"{bert_dir}/test_fusion.csv")
    if train_df is None or val_df is None or test_df is None:
        raise ValueError("数据集加载失败")
    
    # 调试模式：仅使用部分样本（分层采样，保持类别比例）
    if 0 < DEBUG_SAMPLE_RATIO < 1.0:
        n_train_orig = len(train_df)
        n_val_orig = len(val_df)
        n_test_orig = len(test_df)
        
        def stratified_sample(df, frac, label_col='label'):
            """分层采样：从每个类别中按相同比例抽取样本"""
            sampled_dfs = []
            for label_val in df[label_col].unique():
                label_subset = df[df[label_col] == label_val]
                n_sample = max(1, int(len(label_subset) * frac))  # 至少保留1个样本
                sampled_dfs.append(
                    label_subset.sample(n=n_sample, random_state=RANDOM_SEED)
                )
            return pd.concat(sampled_dfs, axis=0).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        
        train_df = stratified_sample(train_df, DEBUG_SAMPLE_RATIO)
        val_df = stratified_sample(val_df, DEBUG_SAMPLE_RATIO)
        test_df = stratified_sample(test_df, DEBUG_SAMPLE_RATIO)
        
        print(f"\n⚠️  调试模式启用 (DEBUG_SAMPLE_RATIO={DEBUG_SAMPLE_RATIO}, 分层采样):")
        print(f"  训练集: {n_train_orig} -> {len(train_df)}")
        print(f"  验证集: {n_val_orig} -> {len(val_df)}")
        print(f"  测试集: {n_test_orig} -> {len(test_df)}")
    for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        if not all(col in df.columns for col in ['text', 'label']):
            raise ValueError(f"{name}.csv 缺少必需的列: ['text','label']")
    # 统一清洗 label 为 int
    train_labels_raw = _ensure_int_labels(train_df['label'])
    val_labels_raw = _ensure_int_labels(val_df['label'])
    test_labels_raw = _ensure_int_labels(test_df['label'])

    if mode == 'binary':
        # 二分类：任何非0都视为攻击类 1
        train_labels = np.where(train_labels_raw == 0, 0, 1)
        val_labels = np.where(val_labels_raw == 0, 0, 1)
        test_labels = np.where(test_labels_raw == 0, 0, 1)

        # 覆写 DataFrame，保证后续 dataset / report 口径一致
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        train_df['label'] = train_labels
        val_df['label'] = val_labels
        test_df['label'] = test_labels

        label_counts = dict(zip(*np.unique(train_labels, return_counts=True)))
        print(f"\n训练集统计 (二分类模式):")
        print(f"  总样本数: {len(train_df)}")
        print(f"  类别数: 2 (0=正常, 1=攻击[非0])")
        print(f"  类别分布: {label_counts}")

        # 若原始标签不是 {0,1}，明确提示用户当前 CSV 实际上来自多分类预处理
        raw_unique = np.unique(train_labels_raw)
        if raw_unique.size > 2 or (raw_unique.size == 2 and not set(raw_unique.tolist()).issubset({0, 1})):
            print(f"  提示: 你的 CSV 原始 label 去重后为 {raw_unique.tolist()}，已在二分类模式下映射为 0/1。")
    else:
        # 多分类：保留原始标签
        train_labels = train_labels_raw
        val_labels = val_labels_raw
        test_labels = test_labels_raw

        label_counts = dict(zip(*np.unique(train_labels, return_counts=True)))
        unique_labels = np.unique(train_labels)
        print(f"\n训练集统计 (多分类模式):")
        print(f"  总样本数: {len(train_df)}")
        print(f"  类别数: {len(unique_labels)}")
        print(f"  类别分布: {label_counts}")
    if len(label_counts) > 0:
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        if min_count > 0:
            print(f"  不平衡度: {max_count}/{min_count} = {max_count/min_count:.1f}x")
    # num_labels：binary 固定为 2；multiclass 用 max+1 防止标签不连续
    all_labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)
    inferred_num_labels = 2 if mode == 'binary' else _infer_num_labels_from_labels(all_labels)

    return {
        # 原始 DataFrame，便于后续访问会话级前缀特征等数值列
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        # 兼容原有调用方式的 tuple
        'train': (train_df['text'].values, train_labels),
        'val': (val_df['text'].values, val_labels),
        'test': (test_df['text'].values, test_labels),
        'num_labels': int(inferred_num_labels),
        'label_counts': label_counts
    }


def compute_class_weights(labels, num_labels, power):
    """使用 sklearn 的 balanced 策略并进行幂平滑，返回长度=num_labels 的权重张量"""
    unique_labels = np.unique(labels)
    base_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=labels)
    smoothed = np.power(base_weights, power)
    smoothed = smoothed / smoothed.mean() if smoothed.mean() > 0 else smoothed
    weights = np.ones(num_labels, dtype=np.float32)
    for label, w in zip(unique_labels, smoothed):
        if 0 <= int(label) < num_labels:
            weights[int(label)] = float(w)
    print(f"\n类别权重 (平滑系数={power}): 范围[{weights.min():.4f},{weights.max():.4f}] 比例{(weights.max()/max(weights.min(),1e-6)):.2f}x")
    return torch.tensor(weights, dtype=torch.float32)


def _format_session_stat_value(v):
    """将会话级数值特征格式化为字符串，供伪 token 使用。"""
    try:
        fv = float(v)
        if abs(fv - round(fv)) < 1e-6:
            return str(int(round(fv)))
        return f"{fv:.4f}"
    except Exception:
        return str(v)


def build_text_with_session_tokens(df: pd.DataFrame, session_cols, base_text_col: str = 'text'):
    """
    将数值型会话前缀特征编码为伪 token，并与原始流级文本拼接。
    示例：
      原 text: "1 2 3 ..."
      会话特征: SESS_FLOW_COUNT=10, SESS_TOTAL_BYTES=1234
      拼接后: "1 2 3 ... [SEP] SESS_FLOW_COUNT=10 SESS_TOTAL_BYTES=1234"
    """
    if not session_cols:
        return df[base_text_col].astype(str).values

    df_local = df.copy()
    df_local[session_cols] = df_local[session_cols].fillna(0)

    def _row_to_session_str(row):
        parts = []
        for col in session_cols:
            if col in row:
                parts.append(f"{col}={_format_session_stat_value(row[col])}")
        return " ".join(parts)

    session_strings = df_local.apply(_row_to_session_str, axis=1)
    combined = df_local[base_text_col].astype(str) + " [SEP] " + session_strings
    return combined.values


def train_distilbert(model_path: str, bert_dir: str, output_dir: str, num_epochs: int,
                     mode: str = DEFAULT_MODE, dataset_name: str = DEFAULT_DATASET,
                     use_session_tokens: bool = True, loss_type: str = 'focal'):
    """使用 DistilBERT 进行（多）分类微调，支持可选的会话前缀特征伪 token 融合。"""
    set_seed(RANDOM_SEED)
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return
    
    # 根据 mode 自动选择对应的数据目录（若用户未通过 bert_dir 手动指定）
    actual_bert_dir = bert_dir if bert_dir else get_bert_dir(dataset_name, mode)
    if not os.path.exists(actual_bert_dir):
        print(f"错误: 数据目录不存在: {actual_bert_dir}")
        print(f"请先运行 data_preprocessing.py 生成 {mode} 模式的数据")
        return
    print(f"\n训练模式: {mode}")
    print(f"数据集: {dataset_name}")
    print(f"数据目录: {actual_bert_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 加载数据
    data = load_bert_datasets(actual_bert_dir, mode=mode)
    train_df, val_df, test_df = data['train_df'], data['val_df'], data['test_df']
    num_labels = data['num_labels']

    # 基础文本与标签
    # 注意：label 已在 load_bert_datasets 内按 mode 做过清洗/映射，这里直接使用即可
    train_labels = _ensure_int_labels(train_df['label'])
    val_labels = _ensure_int_labels(val_df['label'])
    test_labels = _ensure_int_labels(test_df['label'])

    # 二分类下再做一次断言式兜底，防止外部误改导致标签越界
    if mode == 'binary':
        train_labels = np.where(train_labels == 0, 0, 1)
        val_labels = np.where(val_labels == 0, 0, 1)
        test_labels = np.where(test_labels == 0, 0, 1)
        train_df = train_df.copy(); train_df['label'] = train_labels
        val_df = val_df.copy(); val_df['label'] = val_labels
        test_df = test_df.copy(); test_df['label'] = test_labels

    # 从特征配置中读取会话前缀特征名（若存在）
    # actual_bert_dir 是 processed_data/bert/binary 或 multiclass
    # feature_columns.json 在 processed_data/dataset_name/ 目录下
    feature_cfg_path = Path(PROCESSED_DATA_BASE_DIR) / dataset_name / 'feature_columns.json'
    session_stat_cols = []
    if feature_cfg_path.exists():
        try:
            with open(feature_cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            session_stat_cols = cfg.get('session_stat_features', [])
            print(f"\n从 {feature_cfg_path} 读取会话前缀特征列: {session_stat_cols}")
        except Exception as e:
            print(f"警告: 读取特征配置失败 ({e})，将不使用会话特征伪 token。")
            use_session_tokens = False
            session_stat_cols = []
    else:
        if use_session_tokens:
            print(f"警告: 未找到 {feature_cfg_path}，将不使用会话特征伪 token。")
        use_session_tokens = False

    # 根据开关，决定是否在文本中拼接会话前缀伪 token
    if use_session_tokens and session_stat_cols:
        print("\n启用会话前缀特征伪 token 融合到 BERT 输入文本中...")
        train_texts = build_text_with_session_tokens(train_df, session_stat_cols)
        val_texts = build_text_with_session_tokens(val_df, session_stat_cols)
        test_texts = build_text_with_session_tokens(test_df, session_stat_cols)
    else:
        print("\n不使用会话特征伪 token，仅基于单流级文本训练（可视为消融设置）...")
        train_texts = train_df['text'].astype(str).values
        val_texts = val_df['text'].astype(str).values
        test_texts = test_df['text'].astype(str).values
    # tokenizer/model: 使用 Auto* 以兼容 BERT/DistilBERT 权重；若模型无 pad_token，进行修复
    print("\n加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("加载模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=int(num_labels),
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    # 数据集
    print("创建数据集...")
    train_dataset = PacketDataset(tokenizer, train_texts, train_labels)
    val_dataset = PacketDataset(tokenizer, val_texts, val_labels)
    test_dataset = PacketDataset(tokenizer, test_texts, test_labels)

    # -----------------------------------------------------------------------------
    # 构造 WeightedRandomSampler，仅作用于训练集：
    #  - 依据类别频率计算每个样本的采样权重；
    #  - 频率越小，权重越大，从而在训练阶段过采样少数类、弱化多数类。
    # -----------------------------------------------------------------------------
    print("\n构建训练集 WeightedRandomSampler 以缓解类别不平衡...")
    train_label_tensor = torch.tensor(train_labels, dtype=torch.long)
    # 使用 num_labels 兜底，确保所有类别都被统计到
    if num_labels is not None and int(num_labels) > 0:
        class_counts = torch.bincount(train_label_tensor, minlength=int(num_labels)).float()
    else:
        class_counts = torch.bincount(train_label_tensor).float()
    # 避免除以 0
    class_counts = torch.clamp(class_counts, min=1.0)
    class_weights_for_sampler = 1.0 / class_counts
    sample_weights = class_weights_for_sampler[train_label_tensor]
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # 允许过采样少数类
    )
    # 根据 loss_type 选择损失函数
    focal_gamma = 2.0  # Focal Loss 的 gamma 参数，控制聚焦程度

    if loss_type == 'ce':
        # 纯交叉熵损失
        loss_fn = nn.CrossEntropyLoss().to(device)
        print("使用纯交叉熵损失函数")
    elif loss_type == 'focal':
        # 无权重版本：Focal Loss (gamma=2, alpha=None)
        loss_fn = FocalLoss(gamma=focal_gamma, alpha=None).to(device)
        print(f"使用无权重 Focal Loss (gamma={focal_gamma})")
    elif loss_type == 'focal_weighted':
        # 有权重版本：Focal Loss (gamma=2, alpha=class_weights)
        class_weights = compute_class_weights(train_labels, num_labels, power=0.25)
        # 将 FocalLoss 移到与模型相同的设备，确保 alpha buffer 也在正确设备上
        loss_fn = FocalLoss(gamma=focal_gamma, alpha=class_weights).to(device)
        print(f"使用有权重 Focal Loss (gamma={focal_gamma})")
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")

    class CustomLossTrainer(Trainer):
        def __init__(self, *args, loss_fn=None, train_sampler=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_fn = loss_fn
            # 仅用于训练集的数据加载；验证/测试仍使用默认 DataLoader
            self.train_sampler = train_sampler

        def get_train_dataloader(self):
            """
            只在训练阶段应用 WeightedRandomSampler，保证每个 batch 中少数类有更高出现频率。
            验证与测试 DataLoader 仍沿用 Trainer 默认实现（顺序/随机采样但不重采样）。
            """
            if self.train_sampler is not None:
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.args.train_batch_size,
                    sampler=self.train_sampler,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                )
            return super().get_train_dataloader()

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop('labels')
            outputs = model(**inputs)
            logits = outputs.logits
            # 统一通过 self.loss_fn 计算损失（支持 FocalLoss / CrossEntropyLoss）
            loss = self.loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss
    # 评估指标（与 notebooks 风格一致，监控 weighted F1）
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        # 防止“塌陷到单一类别”时指标表面稳定：输出预测类别多样性
        num_pred_classes = int(np.unique(preds).size)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'pred_classes': num_pred_classes}
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type='cosine',
        fp16=torch.cuda.is_available(),
        logging_strategy='epoch',
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        report_to='none',
        seed=RANDOM_SEED,
        dataloader_num_workers=0,
        dataloader_pin_memory=True
    )
    logging_cb = DetailedLoggingCallback()
    early_cb = EarlyStoppingCallback(early_stopping_patience=2)
    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[logging_cb, early_cb],
        loss_fn=loss_fn,
        train_sampler=train_sampler  # 传入 WeightedRandomSampler 以缓解类别不平衡
    )
    # 训练前 sanity check
    print("\n训练前检查...")
    sample = train_dataset[0]
    # 统计UNK占比
    with torch.no_grad():
        enc = tokenizer(
            str(train_texts[0]), max_length=512, padding='max_length', truncation=True, return_tensors='pt'
        )
        input_ids = enc['input_ids'][0]
        unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1
        unk_ratio = float((input_ids == unk_id).sum().item()) / float(len(input_ids)) if unk_id >= 0 else 0.0
        print(f"  UNK比例估计: {unk_ratio:.3f}")
    with torch.no_grad():
        test_inp = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if k != 'labels'}
        test_out = model(**test_inp)
        test_pred = int(torch.argmax(test_out.logits, dim=-1).item())
        print(f"  样本预测: 真实={int(sample['labels'].item())}, 预测={test_pred}")
    # 训练
    trainer.train()
    # 保存
    print(f"\n保存模型到 {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    # 保存/复制标签映射，供推理/生成器使用
    mapping_src = Path(PROCESSED_DATA_BASE_DIR) / dataset_name / 'bert' / mode / 'attack_type_mapping.json'
    mapping_dst = Path(output_dir) / 'attack_type_mapping.json'
    if mode == 'multiclass':
        if mapping_src.exists():
            try:
                shutil.copyfile(mapping_src, mapping_dst)
                print(f"已复制多分类标签映射到: {mapping_dst}")
            except Exception as e:
                print(f"复制标签映射失败: {e}")
        else:
            print(f"警告: 多分类模式未找到 {mapping_src}，下游若需要类别名映射请重新运行预处理生成。")
    else:
        # 二分类：无论源映射是否存在，都写入明确的二分类映射，避免误用多分类映射文件
        try:
            with open(mapping_dst, 'w', encoding='utf-8') as f:
                json.dump({"Normal": 0, "Attack": 1}, f, ensure_ascii=False, indent=2)
            print(f"已写入二分类标签映射: {mapping_dst}")
        except Exception:
            pass
    # 保存标签映射占位：若预处理阶段已生成 attack_type_mapping.json，可复制；
    # 此处保存 num_labels 及会话特征使用情况，供 generator 参考
    training_info = {
        'num_labels': int(num_labels),
        'use_session_tokens': bool(use_session_tokens),
        'session_stat_features': session_stat_cols,
        'loss_type': loss_type
    }
    with open(os.path.join(output_dir, 'training_info.json'), 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)
    # 详细评估
    print("\n" + "="*80)
    print("最终评估")
    print("="*80)
    def evaluate_detailed(dataset, labels, split_name):
        print(f"\n{split_name}集:")
        results = trainer.predict(dataset)
        preds = np.argmax(results.predictions, axis=-1)
        # 指定 labels 确保多分类时所有类别都出现在报告中
        print(classification_report(labels, preds, labels=list(range(num_labels)), digits=4, zero_division=0))
        return results, preds
    train_results, train_preds = evaluate_detailed(train_dataset, train_labels, "训练")
    val_results, val_preds = evaluate_detailed(val_dataset, val_labels, "验证")
    test_results, test_preds = evaluate_detailed(test_dataset, test_labels, "测试")
    # 绘图与混淆矩阵
    plot = logging_cb.get_plot_data()
    if len(plot['epochs']) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(plot['epochs'], plot['train_losses'], 'b-o', label='Train')
        axes[0, 0].plot(plot['epochs'], plot['val_losses'], 'r-o', label='Val')
        axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss'); axes[0, 0].set_title('Loss Curves'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(plot['epochs'], plot['val_accs'], 'g-o'); axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy'); axes[0, 1].set_title('Validation Accuracy'); axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(plot['epochs'], plot['val_f1s'], 'm-o'); axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('F1 Score'); axes[1, 0].set_title('Validation F1'); axes[1, 0].grid(True, alpha=0.3)
        cm = confusion_matrix(test_labels, test_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], cbar=True)
        axes[1, 1].set_xlabel('Predicted'); axes[1, 1].set_ylabel('True'); axes[1, 1].set_title('Confusion Matrix (Test)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        print(f"\n图表已保存: {output_dir}/training_curves.png")
        plt.close()
    # 保存指标
    metrics = {
        'train': {k.replace('test_', ''): float(v) for k, v in train_results.metrics.items()},
        'val': {k.replace('test_', ''): float(v) for k, v in val_results.metrics.items()},
        'test': {k.replace('test_', ''): float(v) for k, v in test_results.metrics.items()}
    }
    with open(f"{output_dir}/evaluation_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("\n" + "="*80)
    print("✓ 训练与评估全部完成")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--bert_dir', type=str, default=None,
                       help='数据目录（可选，默认根据 --mode 自动选择 processed_data/bert/binary 或 multiclass）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（可选，默认根据 --mode 自动选择 distilbert-finetuned 或 distilbert-finetuned-multiclass）')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass'], default=DEFAULT_MODE,
                       help='训练模式: binary(二分类) 或 multiclass(多分类)，自动选择对应数据目录')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                       help=f'数据集名称（例如: NF-UNSW-NB15-v3），默认: {DEFAULT_DATASET}')
    # 控制是否在 BERT 输入中拼接会话前缀特征伪 token，便于做消融实验
    parser.add_argument('--use_session_tokens', dest='use_session_tokens', action='store_true')
    parser.add_argument('--no_session_tokens', dest='use_session_tokens', action='store_false')
    parser.set_defaults(use_session_tokens=True)
    # 损失函数选择
    parser.add_argument('--loss_type', type=str, choices=['focal', 'focal_weighted', 'ce'],
                       default='focal',
                       help='损失函数类型: focal(无权重Focal Loss), focal_weighted(有权重Focal Loss), ce(纯交叉熵)')
    args = parser.parse_args()

    # 根据 mode 自动选择输出目录（如果用户未指定）
    output_dir = args.output_dir if args.output_dir else get_output_dir(args.mode)

    train_distilbert(
        model_path=args.model_path,
        bert_dir=args.bert_dir,  # 此参数现在仅用于向后兼容，实际由 dataset 和 mode 决定
        output_dir=output_dir,
        num_epochs=args.epochs,
        mode=args.mode,
        dataset_name=args.dataset,
        use_session_tokens=args.use_session_tokens,
        loss_type=args.loss_type
    )