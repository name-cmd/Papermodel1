import pandas as pd
import os
import sys
import random
import numpy as np
import json
from tqdm import tqdm

# --- 配置参数 ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 文件路径配置（支持多数据集）
INPUT_DATA_DIR = './datasets'
INPUT_FILENAME = 'NF-UNSW-NB15-v3.csv'  # 默认数据集，可通过命令行参数覆盖
OUTPUT_DATA_DIR = './processed_data'  # 基础输出目录

# 关键列定义（确认标签列为"Label"，数值型：0=正常，1=攻击）
TIME_COLUMN = 'FLOW_START_MILLISECONDS'
LABEL_COLUMN = 'Label'  
ATTACK_CAT_COLUMN = 'Attack' 

# 序列长度与会话定义
MIN_SEQ_LEN = 3
MAX_SEQ_LEN = 64
SESSION_TIMEOUT = 120  # 秒

# 数据采样配置
# - DATA_SAMPLE_RATIO: 采样比例（0.0-1.0），1.0表示读取全部
# - NUM_CHUNKS: 分块数量，用于大文件分块随机抽样
# - LARGE_FILE_THRESHOLD_MB: 超过此大小启用分块抽样（单位MB）
DATA_SAMPLE_RATIO = 1.0  # 读取完整数据集
NUM_CHUNKS = 5
LARGE_FILE_THRESHOLD_MB = 500

# 数据平衡与划分配置
BENIGN_RETAIN_RATIO = 0.05
SPLIT_RATIOS = [
    ('train', 0.7),
    ('val', 0.10),
    ('test', 0.20),
]

# --- 数据类型优化配置 ---
# 指定各列的优化数据类型，减少内存占用
# 整数字段使用整数类型，浮点字段（平均值、比率等）使用 float32
DTYPE_SPEC = {
    # 协议和标志位（整数）
    'PROTOCOL': 'int8',
    'TCP_FLAGS': 'int16',
    'CLIENT_TCP_FLAGS': 'int16',
    'SERVER_TCP_FLAGS': 'int16',
    'Label': 'int8',
    # 端口（整数）
    'L4_SRC_PORT': 'uint16',
    'L4_DST_PORT': 'uint16',
    # 包和字节计数（整数）
    'IN_PKTS': 'int32',
    'OUT_PKTS': 'int32',
    'IN_BYTES': 'int64',
    'OUT_BYTES': 'int64',
    # 持续时间（整数，毫秒）
    'FLOW_DURATION_MILLISECONDS': 'int64',
    'DURATION_IN': 'int64',
    'DURATION_OUT': 'int64',
    # 重传统计（整数）
    'RETRANSMITTED_IN_BYTES': 'int64',
    'RETRANSMITTED_OUT_BYTES': 'int64',
    'RETRANSMITTED_IN_PKTS': 'int32',
    'RETRANSMITTED_OUT_PKTS': 'int32',
    # IAT 统计（浮点，因为是平均值/标准差）
    'SRC_TO_DST_IAT_AVG': 'float32',
    'DST_TO_SRC_IAT_AVG': 'float32',
    'SRC_TO_DST_IAT_MIN': 'float32',
    'SRC_TO_DST_IAT_MAX': 'float32',
    'SRC_TO_DST_IAT_STDDEV': 'float32',
    'DST_TO_SRC_IAT_MIN': 'float32',
    'DST_TO_SRC_IAT_MAX': 'float32',
    'DST_TO_SRC_IAT_STDDEV': 'float32',
    # TTL（整数）
    'MIN_TTL': 'int16',
    'MAX_TTL': 'int16',
    # 包长度统计（整数）
    'LONGEST_FLOW_PKT': 'int32',
    'SHORTEST_FLOW_PKT': 'int32',
    'MIN_IP_PKT_LEN': 'int32',
    'MAX_IP_PKT_LEN': 'int32',
    # 包大小分布计数（整数）
    'NUM_PKTS_UP_TO_128_BYTES': 'int32',
    'NUM_PKTS_128_TO_256_BYTES': 'int32',
    'NUM_PKTS_256_TO_512_BYTES': 'int32',
    'NUM_PKTS_512_TO_1024_BYTES': 'int32',
    'NUM_PKTS_1024_TO_1514_BYTES': 'int32',
    # TCP 窗口（整数）
    'TCP_WIN_MAX_IN': 'int32',
    'TCP_WIN_MAX_OUT': 'int32',
    # ICMP（整数）
    'ICMP_TYPE': 'int16',
    'ICMP_IPV4_TYPE': 'int16',
    # 吞吐量（浮点，因为是计算值）
    'SRC_TO_DST_AVG_THROUGHPUT': 'int64',
    'DST_TO_SRC_AVG_THROUGHPUT': 'int64',
    # 每秒字节数（浮点，因为是比率）
    'SRC_TO_DST_SECOND_BYTES': 'float32',
    'DST_TO_SRC_SECOND_BYTES': 'float32',
    # DNS 相关（整数）
    'DNS_QUERY_ID': 'int32',
    'DNS_QUERY_TYPE': 'int16',
    'DNS_TTL_ANSWER': 'int32',
    # FTP（整数）
    'FTP_COMMAND_RET_CODE': 'int16',
    # L7 协议（浮点，因为数据中有小数如 5.0, 11.0）
    'L7_PROTO': 'float32',
}

# --- 特征选择策略 ---
EXCLUDE_COLS = [
    'id', 'label', LABEL_COLUMN, ATTACK_CAT_COLUMN, 
    'IPV4_SRC_ADDR', 'IPV4_DST_ADDR',
    'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
    'service_port', 'ip_min', 'ip_max', 'aggr_key', 'sub_session_id', 
    'new_session_flag', 'time_diff', 'Session_ID'
]

# 核心流级特征列表（14个）
CORE_FLOW_FEATURES = [
    'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
    'FLOW_DURATION_MILLISECONDS', 'PROTOCOL',
    'TCP_FLAGS_ENCODED', 'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_OUT_BYTES',
    'PKT_SIZE_AVG_IN', 'PKT_SIZE_AVG_OUT',
    'SRC_PORT_CATEGORY', 'DST_PORT_CATEGORY',
    'SRC_TO_DST_IAT_AVG', 'DST_TO_SRC_IAT_AVG'
]

# 会话级特征列表（前缀累积窗口版本）
SESSION_STAT_COLS = [
    'SESS_FLOW_COUNT',     # 当前会话已出现的流数量
    'SESS_UNIQUE_PORT',    # 当前会话前缀内不同目标端口类别数量
    'SESS_TCP_RATIO',      # 当前会话前缀内 TCP 流比例
    'SESS_FLOW_DUR_VAR',   # 当前会话前缀内流持续时间方差
    'SESS_TOTAL_BYTES'     # 当前会话前缀内 IN+OUT 总字节数
]

# TCP_FLAGS 编码映射表
TCP_FLAGS_MAP = {
    1: 1, 2: 2, 4: 3, 8: 4, 16: 5, 32: 6, 64: 7, 128: 8, 256: 9,
    3: 10, 18: 11, 20: 12, 17: 13, 34: 14
}


# --- 辅助函数 ---
def get_dataset_name(filename):
    """从文件名提取数据集名称"""
    return filename[:-4] if filename.endswith('.csv') else filename


def load_column_names_from_data(data_path):
    """读取CSV文件的列名"""
    try:
        try:
            df_head = pd.read_csv(data_path, encoding='utf-8', nrows=0)
        except UnicodeDecodeError:
            df_head = pd.read_csv(data_path, encoding='cp1252', nrows=0)
        return [c.strip() for c in df_head.columns]
    except Exception as e:
        print(f"读取列名失败: {e}")
        return None


def get_file_line_count(file_path):
    """快速统计文件行数"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f) - 1  # 减去表头


def read_data_chunked_sampling(data_path, col_names, sample_ratio, num_chunks=5):
    """
    分块随机抽样读取大文件
    将文件分成 num_chunks 块，从每块中随机抽取 sample_ratio 比例的数据
    确保覆盖文件各个部分的数据（包括不同攻击类型）
    """
    print(f"启用分块随机抽样模式（{num_chunks} 块，每块抽取 {sample_ratio*100:.0f}%）...")
    
    # 统计总行数
    total_lines = get_file_line_count(data_path)
    print(f"数据集总行数: {total_lines:,}")
    
    # 计算每块的行数和每块要抽取的行数
    chunk_size = total_lines // num_chunks
    rows_per_chunk = int(chunk_size * sample_ratio)
    
    print(f"每块约 {chunk_size:,} 行，每块抽取 {rows_per_chunk:,} 行")
    
    # 过滤出数据集中存在的列的类型定义
    dtype_filtered = {k: v for k, v in DTYPE_SPEC.items() if k in col_names}
    
    chunks = []
    for i in range(num_chunks):
        skip_start = i * chunk_size + 1  # +1 跳过表头
        
        # 读取当前块
        try:
            chunk_df = pd.read_csv(
                data_path, 
                header=0, 
                names=col_names,
                skiprows=range(1, skip_start),
                nrows=chunk_size if i < num_chunks - 1 else None,
                encoding='utf-8',
                dtype=dtype_filtered,
                low_memory=True
            )
        except UnicodeDecodeError:
            chunk_df = pd.read_csv(
                data_path, 
                header=0, 
                names=col_names,
                skiprows=range(1, skip_start),
                nrows=chunk_size if i < num_chunks - 1 else None,
                encoding='cp1252',
                dtype=dtype_filtered,
                low_memory=True
            )
        
        # 从当前块随机抽样
        if len(chunk_df) > rows_per_chunk:
            chunk_df = chunk_df.sample(n=rows_per_chunk, random_state=RANDOM_SEED + i)
        
        chunks.append(chunk_df)
        print(f"  块 {i+1}/{num_chunks}: 读取 {len(chunks[-1]):,} 行")
    
    # 合并所有块
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    
    print(f"分块抽样完成，共 {len(df):,} 行")
    return df


def read_data_simple(data_path, col_names, nrows=None):
    """简单读取数据（用于小文件或读取全部）"""
    dtype_filtered = {k: v for k, v in DTYPE_SPEC.items() if k in col_names}
    
    try:
        df = pd.read_csv(
            data_path, header=0, names=col_names, 
            encoding='utf-8', dtype=dtype_filtered,
            low_memory=True, nrows=nrows
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            data_path, header=0, names=col_names, 
            encoding='cp1252', dtype=dtype_filtered,
            low_memory=True, nrows=nrows
        )
    return df


# --- 向量化特征加工函数 ---
def encode_tcp_flags_vectorized(flags_series):
    """向量化 TCP FLAGS 编码"""
    flags = flags_series.fillna(0).astype(int)
    # 使用 map + 默认值
    encoded = flags.map(TCP_FLAGS_MAP).fillna(15).astype(int)
    return encoded


def encode_port_vectorized(port_series):
    """向量化端口分类编码"""
    port = port_series.fillna(0).astype(int)
    # 使用 np.select 进行向量化条件判断
    conditions = [
        (port >= 1) & (port <= 1024),
        port.isin([8080, 3306]),
        (port >= 1025) & (port <= 65535)
    ]
    choices = [1, 2, 3]
    return np.select(conditions, choices, default=4).astype(np.int8)


def calculate_pkt_size_avg_vectorized(bytes_series, pkts_series):
    """向量化计算平均包大小"""
    bytes_vals = bytes_series.fillna(0).astype(float)
    pkts_vals = pkts_series.fillna(0).astype(float)
    # 避免除零
    result = np.where(pkts_vals != 0, bytes_vals / pkts_vals, 0.0)
    return np.round(result, 4)


def format_numeric_value(x):
    """
    智能格式化数值：
    - 整数或可无损转为整数的浮点数 -> 输出整数字符串
    - 真正的小数 -> 保留有效小数位（去除尾部无效0）
    """
    if pd.isna(x):
        return '0'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    # 浮点数处理
    if x == int(x):  # 可无损转为整数
        return str(int(x))
    # 真正的小数，保留有效位数（最多4位，去除尾部0）
    formatted = f"{x:.4f}".rstrip('0').rstrip('.')
    return formatted


def build_text_column_vectorized(df, feature_cols):
    """向量化构建文本列，智能处理数值格式"""
    print("  使用向量化方式构建文本列...")
    
    # 准备特征列的字符串版本
    str_cols = []
    for col in feature_cols:
        col_data = df[col].fillna(0)
        # 对于浮点数，智能格式化
        if col_data.dtype in ['float32', 'float64']:
            # 检查是否所有值都可以无损转为整数
            is_all_int = (col_data == col_data.astype(np.int64)).all()
            if is_all_int:
                col_str = col_data.astype(np.int64).astype(str)
            else:
                col_str = col_data.apply(format_numeric_value)
        else:
            col_str = col_data.astype(str)
        str_cols.append(col_str)
    
    # 使用 DataFrame 的 agg 方法拼接
    text_df = pd.concat(str_cols, axis=1)
    text_df.columns = feature_cols
    return text_df.apply(lambda row: ' '.join(row.values), axis=1)


def compute_prefix_session_stats_vectorized(df):
    """
    向量化计算前缀累积会话特征
    使用 groupby + transform 替代 Python 循环，大幅提升性能
    """
    print("正在按会话计算前缀累积会话特征（向量化）...")
    
    # 确保按会话与时间排序
    df = df.sort_values(by=['Session_ID', TIME_COLUMN]).reset_index(drop=True)
    
    n_sessions = df['Session_ID'].nunique()
    print(f"  共 {n_sessions:,} 个会话，{len(df):,} 条流记录")
    
    # 1. 前缀流计数
    print("  计算 SESS_FLOW_COUNT...")
    df['SESS_FLOW_COUNT'] = df.groupby('Session_ID').cumcount() + 1
    
    # 2. TCP 前缀比例
    print("  计算 SESS_TCP_RATIO...")
    df['_is_tcp'] = (df['PROTOCOL'] == 6).astype(np.float32)
    df['_tcp_cum'] = df.groupby('Session_ID')['_is_tcp'].cumsum()
    df['SESS_TCP_RATIO'] = np.round(df['_tcp_cum'] / df['SESS_FLOW_COUNT'], 4)
    df.drop(columns=['_is_tcp', '_tcp_cum'], inplace=True)
    
    # 3. 前缀总字节数
    print("  计算 SESS_TOTAL_BYTES...")
    df['_total_bytes'] = df['IN_BYTES'].fillna(0).astype(np.float32) + df['OUT_BYTES'].fillna(0).astype(np.float32)
    df['SESS_TOTAL_BYTES'] = df.groupby('Session_ID')['_total_bytes'].cumsum()
    df.drop(columns=['_total_bytes'], inplace=True)
    
    # 4. 端口类别前缀唯一数（使用 expanding nunique）
    print("  计算 SESS_UNIQUE_PORT...")
    df['SESS_UNIQUE_PORT'] = df.groupby('Session_ID')['DST_PORT_CATEGORY'].transform(
        lambda x: x.expanding().apply(lambda s: s.nunique(), raw=False)
    ).astype(np.int32)
    
    # 5. 持续时间前缀方差（使用 expanding var）
    print("  计算 SESS_FLOW_DUR_VAR...")
    df['_dur'] = df['FLOW_DURATION_MILLISECONDS'].fillna(0).astype(np.float32)
    df['SESS_FLOW_DUR_VAR'] = df.groupby('Session_ID')['_dur'].transform(
        lambda x: x.expanding().var().fillna(0)
    ).round(4)
    df.drop(columns=['_dur'], inplace=True)
    
    # 确保所有会话特征列存在且无缺失
    for col in SESSION_STAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0
    
    print("  前缀累积特征计算完成")
    return df


# --- 数据划分函数 ---
def _compute_split_counts(total, ratios):
    """根据比例列表返回每个 split 应分配的样本数"""
    if total <= 0:
        return [0] * len(ratios)
    counts = [int(total * r) for r in ratios]
    allocated = sum(counts)
    idx = 0
    while allocated < total:
        counts[idx] += 1
        allocated += 1
        idx = (idx + 1) % len(counts)
    return counts


def stratified_split_by_label(df, label_col):
    """按照 SPLIT_RATIOS 对每个标签独立划分 train/val/test"""
    ratios = [r for _, r in SPLIT_RATIOS]
    split_frames = {name: [] for name, _ in SPLIT_RATIOS}
    
    for _, group in df.groupby(label_col):
        group = group.sample(frac=1.0, random_state=RANDOM_SEED)
        counts = _compute_split_counts(len(group), ratios)
        start = 0
        for (split_name, _), count in zip(SPLIT_RATIOS, counts):
            if count > 0:
                split_frames[split_name].append(group.iloc[start:start+count])
                start += count
    
    for split_name in split_frames:
        if split_frames[split_name]:
            split_frames[split_name] = pd.concat(split_frames[split_name], axis=0)
        else:
            split_frames[split_name] = pd.DataFrame(columns=df.columns)
        split_frames[split_name] = split_frames[split_name].sample(
            frac=1.0, random_state=RANDOM_SEED
        ).reset_index(drop=True)
    
    return split_frames


def export_balanced_fusion_data(df, output_dir, label_mode):
    """为分类模型生成平衡后的 train/val/test CSV"""
    output_cols = ['text', 'label'] + SESSION_STAT_COLS
    fusion_df = df[output_cols].copy()
    total_samples = len(fusion_df)
    benign_mask = fusion_df['label'] == 0
    benign_total = int(benign_mask.sum())

    benign_keep = 0
    benign_subset = fusion_df.iloc[0:0]
    if benign_total > 0:
        benign_keep = max(1, int(total_samples * BENIGN_RETAIN_RATIO))
        benign_keep = min(benign_keep, benign_total)
        benign_subset = fusion_df[benign_mask].sample(
            n=benign_keep, random_state=RANDOM_SEED, replace=False
        )

    attack_subset = fusion_df[~benign_mask]
    balanced_df = pd.concat([benign_subset, attack_subset], axis=0).sample(
        frac=1.0, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(f"\n[平衡策略统计 - {label_mode}]")
    print(f"  原始样本数: {total_samples}")
    print(f"  Benign 总数: {benign_total}")
    print(f"  攻击总数: {len(attack_subset)}")
    print(f"  Benign 保留数量 (~{BENIGN_RETAIN_RATIO*100:.1f}%): {benign_keep}")
    print(f"  平衡后总样本数: {len(balanced_df)}")

    splits = stratified_split_by_label(balanced_df, 'label')
    for split_name, split_df in splits.items():
        out_path = os.path.join(output_dir, f"{split_name}_fusion.csv")
        split_df[output_cols].to_csv(out_path, index=False, encoding='utf-8')
        print(f"\n[{split_name} 集标签分布 - {label_mode}]")
        total = len(split_df)
        if total == 0:
            print("  (空集)")
            continue
        for lbl, cnt in split_df['label'].value_counts().sort_index().items():
            pct = cnt / total * 100
            print(f"  label={lbl:<3d} 数量={cnt:<8d} 占比={pct:6.2f}%")

    print(f"\nFusion 数据已导出（路径位于 {output_dir}/[train|val|test]_fusion.csv）")


# --- 核心数据处理函数 ---
def process_netflow_data(input_filename=None):
    """处理网络流量数据"""
    # 确定输入文件名和数据集名称
    if input_filename is None:
        input_filename = INPUT_FILENAME
    dataset_name = get_dataset_name(input_filename)
    
    print(f"[{'='*20} 数据预处理（适配数值标签） {'='*20}]")
    print(f"数据集名称: {dataset_name}")
    
    # 创建输出目录结构
    dataset_output_dir = os.path.join(OUTPUT_DATA_DIR, dataset_name)
    bert_output_dir = os.path.join(dataset_output_dir, 'bert')
    bert_binary_dir = os.path.join(bert_output_dir, 'binary')
    bert_multiclass_dir = os.path.join(bert_output_dir, 'multiclass')
    
    for d in [dataset_output_dir, bert_output_dir, bert_binary_dir, bert_multiclass_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 1. 加载数据与列名验证
    data_file_path = os.path.join(INPUT_DATA_DIR, input_filename)
    col_names = load_column_names_from_data(data_file_path)
    if not col_names:
        print("列名读取失败，终止程序")
        return

    print(f"数据集原始列数: {len(col_names)}")
    print(f"确认标签列 '{LABEL_COLUMN}' 存在: {LABEL_COLUMN in col_names}")
    
    # 2. 读取数据（根据文件大小和采样比例选择策略）
    file_size_mb = os.path.getsize(data_file_path) / (1024 * 1024)
    print(f"文件大小: {file_size_mb:.1f} MB")
    
    if DATA_SAMPLE_RATIO >= 1.0:
        # 读取全部数据
        print("读取比例: 100%（全部数据）")
        print("正在读取数据集...")
        df = read_data_simple(data_file_path, col_names)
    elif file_size_mb > LARGE_FILE_THRESHOLD_MB:
        # 大文件：分块随机抽样
        df = read_data_chunked_sampling(data_file_path, col_names, DATA_SAMPLE_RATIO, NUM_CHUNKS)
    else:
        # 小文件：简单按比例读取前N行
        total_lines = get_file_line_count(data_file_path)
        nrows = int(total_lines * DATA_SAMPLE_RATIO)
        print(f"数据集总行数: {total_lines:,}")
        print(f"读取比例: {DATA_SAMPLE_RATIO*100:.0f}%，将读取前 {nrows:,} 行")
        print("正在读取数据集...")
        df = read_data_simple(data_file_path, col_names, nrows=nrows)
    
    print(f"成功读取 {len(df):,} 行数据")

    # 3. 基础数据清洗
    # 时间戳处理（毫秒转秒）
    df[TIME_COLUMN] = pd.to_numeric(df[TIME_COLUMN], errors='coerce') / 1000.0
    df = df.dropna(subset=[TIME_COLUMN])

    # 关键数值列转换
    numeric_cols = [
        'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'PROTOCOL',
        'TCP_FLAGS', 'L4_SRC_PORT', 'L4_DST_PORT', 'FLOW_DURATION_MILLISECONDS',
        'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_OUT_BYTES',
        'SRC_TO_DST_IAT_AVG', 'DST_TO_SRC_IAT_AVG', LABEL_COLUMN
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 验证原始标签分布
    print(f"\n[原始标签分布验证（数值列 {LABEL_COLUMN}）]")
    label_counts = df[LABEL_COLUMN].value_counts().sort_index()
    print(f"原始标签=0（正常）数量: {label_counts.get(0, 0)}")
    print(f"原始标签=1（攻击）数量: {label_counts.get(1, 0)}")
    if label_counts.get(0, 0) == 0:
        print("⚠️  警告：原始数据中未找到标签=0的样本，可能标签列或数值格式错误！")

    # 4. 预处理 Attack 列（多分类标签）
    if ATTACK_CAT_COLUMN in df.columns:
        attack_series = df[ATTACK_CAT_COLUMN].fillna('Benign').astype(str).str.strip()
        unique_types = attack_series.unique()
        
        # 构建映射：Benign=0，其他按字典序编号
        attack_type_mapping = {}
        for t in unique_types:
            if t.lower() == 'benign':
                attack_type_mapping[t] = 0
                break
        
        current_id = 0
        for t in sorted(unique_types):
            if t not in attack_type_mapping:
                current_id += 1
                attack_type_mapping[t] = current_id
        
        df[ATTACK_CAT_COLUMN] = attack_series
        df['label_multiclass'] = df[ATTACK_CAT_COLUMN].map(attack_type_mapping).fillna(0).astype(int)
        
        # 打印多分类标签分布
        print(f"\n[Attack 列多类别分布统计]")
        for t, label_id in sorted(attack_type_mapping.items(), key=lambda x: x[1]):
            cnt = (df['label_multiclass'] == label_id).sum()
            pct = cnt / len(df) * 100
            print(f"  - 类型: {t:<20} -> label_id={label_id:<3d} 数量={cnt:<8d} 占比={pct:6.2f}%")

        # 保存映射
        mapping_path = os.path.join(bert_multiclass_dir, 'attack_type_mapping.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(attack_type_mapping, f, ensure_ascii=False, indent=2)
        print(f"多分类标签映射已保存至: {mapping_path}")
    else:
        print(f"⚠️ 警告: 数据集中不存在列 '{ATTACK_CAT_COLUMN}'，将只生成二分类数据。")
        df['label_multiclass'] = None

    # 二分类标签：0=正常，1=攻击（向量化）
    df['label_binary'] = (df[LABEL_COLUMN] != 0).astype(np.int8)

    # 5. 流级核心特征加工（向量化）
    print("\n正在加工流级核心特征（向量化）...")
    df['TCP_FLAGS_ENCODED'] = encode_tcp_flags_vectorized(df['TCP_FLAGS'])
    df['SRC_PORT_CATEGORY'] = encode_port_vectorized(df['L4_SRC_PORT'])
    df['DST_PORT_CATEGORY'] = encode_port_vectorized(df['L4_DST_PORT'])
    df['PKT_SIZE_AVG_IN'] = calculate_pkt_size_avg_vectorized(df['IN_BYTES'], df['IN_PKTS'])
    df['PKT_SIZE_AVG_OUT'] = calculate_pkt_size_avg_vectorized(df['OUT_BYTES'], df['OUT_PKTS'])
    df.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT', 'TCP_FLAGS'], errors='ignore', inplace=True)

    # 6. 会话聚合与会话级特征计算
    print("正在构建会话...")
    df['ip_min'] = np.where(
        df['IPV4_SRC_ADDR'] < df['IPV4_DST_ADDR'], 
        df['IPV4_SRC_ADDR'], df['IPV4_DST_ADDR']
    )
    df['ip_max'] = np.where(
        df['IPV4_SRC_ADDR'] < df['IPV4_DST_ADDR'], 
        df['IPV4_DST_ADDR'], df['IPV4_SRC_ADDR']
    )
    
    # 使用向量化字符串拼接构建 aggr_key
    df['aggr_key'] = (
        df['ip_min'].astype(str) + '_' + 
        df['ip_max'].astype(str) + '_' +
        df['DST_PORT_CATEGORY'].astype(str) + '_' + 
        df['PROTOCOL'].astype(str)
    )

    df.sort_values(by=['aggr_key', TIME_COLUMN], inplace=True)
    df['time_diff'] = df.groupby('aggr_key')[TIME_COLUMN].diff().fillna(0)
    df['new_session_flag'] = (df['time_diff'] > SESSION_TIMEOUT).astype(np.int8)
    df['sub_session_id'] = df.groupby('aggr_key')['new_session_flag'].cumsum()
    df['Session_ID'] = df['aggr_key'] + '_' + df['sub_session_id'].astype(str)

    # 过滤短会话（流数 < MIN_SEQ_LEN 的会话）
    # 这样可以避免对无意义的短会话计算会话级特征，同时保持 GPT-2 和 Fusion 数据一致性
    print(f"正在过滤短会话（流数 < {MIN_SEQ_LEN}）...")
    session_counts = df.groupby('Session_ID').size()
    valid_sessions = session_counts[session_counts >= MIN_SEQ_LEN].index
    n_sessions_before = df['Session_ID'].nunique()
    n_flows_before = len(df)
    df = df[df['Session_ID'].isin(valid_sessions)].reset_index(drop=True)
    n_sessions_after = len(valid_sessions)
    n_flows_after = len(df)
    print(f"  过滤前: {n_sessions_before:,} 个会话，{n_flows_before:,} 条流")
    print(f"  过滤后: {n_sessions_after:,} 个会话，{n_flows_after:,} 条流")
    print(f"  过滤掉: {n_sessions_before - n_sessions_after:,} 个短会话，{n_flows_before - n_flows_after:,} 条流")

    # 计算会话级前缀特征（向量化）
    df = compute_prefix_session_stats_vectorized(df)

    # 7. 核心特征筛选
    print("正在筛选核心流级特征...")
    all_cols = df.columns
    missing_features = [feat for feat in CORE_FLOW_FEATURES if feat not in all_cols]
    if missing_features:
        print(f"⚠️  警告：以下核心特征在数据集中不存在，已自动跳过：{missing_features}")
    
    text_feature_cols = [
        feat for feat in CORE_FLOW_FEATURES 
        if feat in all_cols and feat not in EXCLUDE_COLS
    ]

    print(f"\n[特征筛选结果]")
    print(f"最终流级核心特征列（{len(text_feature_cols)}个）: {text_feature_cols}")
    print(f"最终会话级特征列（{len(SESSION_STAT_COLS)}个）: {SESSION_STAT_COLS}")

    # 构建文本列（向量化）
    print("\n构建流级文本列用于 BERT/Fusion 训练...")
    df['text'] = build_text_column_vectorized(df, text_feature_cols)

    # 保存特征列表
    with open(os.path.join(dataset_output_dir, 'feature_columns.json'), 'w') as f:
        json.dump({
            'flow_text_features': text_feature_cols,
            'session_stat_features': SESSION_STAT_COLS,
            'missing_features': missing_features
        }, f, indent=4, ensure_ascii=False)

    # 8. 数据集划分
    print("\n正在划分训练/验证/测试/生成集...")
    unique_sessions = df['Session_ID'].unique()
    np.random.shuffle(unique_sessions)

    n_total = len(unique_sessions)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    n_test = int(n_total * 0.1)

    train_ids = unique_sessions[:n_train]
    val_ids = unique_sessions[n_train:n_train+n_val]
    test_ids = unique_sessions[n_train+n_val:n_train+n_val+n_test]
    gen_ids = unique_sessions[n_train+n_val+n_test:]

    print(f"数据量统计：")
    print(f"  - 总会话数: {n_total}")
    print(f"  - 训练会话数: {n_train}")
    print(f"  - 验证会话数: {n_val}")
    print(f"  - 测试会话数: {n_test}")
    print(f"  - 生成会话数: {len(gen_ids)}")

    # 9. 生成 GPT-2 数据
    def generate_gpt_file(ids, filename):
        output_path = os.path.join(dataset_output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            subset = df[df['Session_ID'].isin(ids)].sort_values(['Session_ID', TIME_COLUMN])
            for _, session_data in tqdm(
                subset.groupby('Session_ID'), desc=f"生成 {filename}"
            ):
                session_unique = session_data.drop_duplicates(subset=text_feature_cols)
                if len(session_unique) < MIN_SEQ_LEN:
                    continue
                if len(session_unique) > MAX_SEQ_LEN:
                    session_unique = session_unique.iloc[:MAX_SEQ_LEN]

                f.write("<bos>\n")
                # 直接使用预先构建好的 text 列
                flow_texts = session_unique['text'].tolist()
                f.write("\n".join(flow_texts))
                f.write("\n<eos>\n\n")

    print("\n生成 GPT-2 数据集...")
    generate_gpt_file(train_ids, 'train_gpt2_input.txt')
    generate_gpt_file(val_ids, 'val_gpt2_input.txt')
    generate_gpt_file(test_ids, 'test_gpt2_input.txt')
    generate_gpt_file(gen_ids, 'gen_input.txt')

    # 10. 生成 Fusion 模型数据
    print("\n" + "="*60)
    print("生成 Fusion 模型数据集（binary + multiclass）")
    print("="*60)
    
    # 生成二分类数据
    print("\n>>> 生成二分类 (binary) 数据...")
    df['label'] = df['label_binary']
    export_balanced_fusion_data(df, bert_binary_dir, 'binary')
    
    # 生成多分类数据
    if df['label_multiclass'].notna().any():
        print("\n>>> 生成多分类 (multiclass) 数据...")
        df['label'] = df['label_multiclass'].astype(int)
        export_balanced_fusion_data(df, bert_multiclass_dir, 'multiclass')
    else:
        print("\n⚠️ 跳过多分类数据生成（缺少 Attack 列）")

    # 11. 结果汇总
    print(f"\n[{'='*20} 数据预处理完成 {'='*20}]")
    print(f"数据集: {dataset_name}")
    print(f"1. GPT-2 数据路径: {dataset_output_dir}/[train/val/test/gen]_gpt2_input.txt")
    print(f"2. Fusion 数据路径（二分类）: {bert_binary_dir}/[train/val/test]_fusion.csv")
    print(f"3. Fusion 数据路径（多分类）: {bert_multiclass_dir}/[train/val/test]_fusion.csv")
    print(f"4. 特征列表路径: {dataset_output_dir}/feature_columns.json")
    print(f"5. 标签说明:")
    print(f"   - binary: 0=正常，1=攻击")
    print(f"   - multiclass: 0=Benign，1..K-1=各攻击类型（详见 attack_type_mapping.json）")
    print(f"\n提示: DistilBERT 训练时使用 --mode binary 或 --mode multiclass 自动选择对应数据")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='网络流量数据预处理')
    parser.add_argument('--dataset', type=str, default=None,
                       help='数据集文件名（例如: NF-UNSW-NB15-v3.csv）')
    parser.add_argument('--sample-ratio', type=float, default=None,
                       help='读取数据集的比例（0.0-1.0），例如 0.2 表示采样20%%')
    parser.add_argument('--num-chunks', type=int, default=None,
                       help='大文件分块数量（默认5）')
    args = parser.parse_args()
    
    # 覆盖全局配置
    if args.sample_ratio is not None:
        if 0.0 < args.sample_ratio <= 1.0:
            DATA_SAMPLE_RATIO = args.sample_ratio
        else:
            print(f"错误: --sample-ratio 必须在 0.0 到 1.0 之间，当前值: {args.sample_ratio}")
            sys.exit(1)
    
    if args.num_chunks is not None:
        if args.num_chunks >= 1:
            NUM_CHUNKS = args.num_chunks
        else:
            print(f"错误: --num-chunks 必须 >= 1，当前值: {args.num_chunks}")
            sys.exit(1)
    
    input_filename = args.dataset if args.dataset else INPUT_FILENAME
    process_netflow_data(input_filename=input_filename)
