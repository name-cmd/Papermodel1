"""
NF-CICIDS2018-v3 数据集预处理脚本
采用分块处理策略，解决大数据集内存不足问题
- GPT-2数据：完整会话，无采样
- Fusion数据：按流级别分层采样

输出路径: ./processed_data/NF-CICIDS2018-v3/
"""
import pandas as pd
import os
import gc
import random
import numpy as np
import json
from tqdm import tqdm

# --- 配置参数 ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 文件路径配置
INPUT_DATA_DIR = './datasets'
INPUT_FILENAME = 'NF-CICIDS2018-v3.csv'
OUTPUT_DATA_DIR = './processed_data'

# 关键列定义
TIME_COLUMN = 'FLOW_START_MILLISECONDS'
LABEL_COLUMN = 'Label'  
ATTACK_CAT_COLUMN = 'Attack' 

# 序列长度与会话定义
MIN_SEQ_LEN = 3
MAX_SEQ_LEN = 64
SESSION_TIMEOUT = 120  # 秒

# GPT-2 数据采样配置
GPT2_SESSION_SAMPLE_RATIO = 0.10  # 从合并后的会话中采样10%用于GPT-2训练

# ========== 分块处理配置 ==========
NUM_PROCESS_CHUNKS = 10  # 将数据分成 10 块处理

# ========== 分层采样配置（基于全数据集统计，按流级别采样） ==========
# 目标：采样后总数控制在 50 万左右
# 采样阈值和比例（优化后）
SAMPLE_THRESHOLDS = [
    # (阈值下限, 阈值上限, 采样比例, 描述)
    (100000, float('inf'), 0.10, '大类(>10万)'),    # 超过10万：采样10%（降低）
    (20000, 100000, 0.20, '中类(2-10万)'),          # 2-10万：采样20%（降低）
    (1000, 20000, 0.50, '小类(1千-2万)'),           # 1千-2万：采样50%
    (0, 1000, 1.00, '极小类(<1千)'),                # 小于1千：全部保留
]
BENIGN_SAMPLE_RATIO = 0.02  # 正常样本采样 2%（降低以控制总数）

# 小样本保护：对于会话过滤后可能大量损失的类别，提前过采样
RARE_ATTACK_BOOST = 3.0  # 对极小类别在流级别过采样3倍，补偿会话过滤损失

# 数据划分配置
SPLIT_RATIOS = [('train', 0.7), ('val', 0.10), ('test', 0.20)]

# --- 数据类型优化配置 ---
# 整数字段使用整数类型，浮点字段（平均值、比率等）使用 float32
DTYPE_SPEC = {
    # 协议和标志位（整数）
    'PROTOCOL': 'int8', 'TCP_FLAGS': 'int16', 'CLIENT_TCP_FLAGS': 'int16',
    'SERVER_TCP_FLAGS': 'int16', 'Label': 'int8',
    # 端口（整数）
    'L4_SRC_PORT': 'uint16', 'L4_DST_PORT': 'uint16',
    # 包和字节计数（整数）
    'IN_PKTS': 'int32', 'OUT_PKTS': 'int32',
    'IN_BYTES': 'int64', 'OUT_BYTES': 'int64',
    # 持续时间（整数，毫秒）
    'FLOW_DURATION_MILLISECONDS': 'int64',
    'DURATION_IN': 'int64', 'DURATION_OUT': 'int64',
    # 重传统计（整数）
    'RETRANSMITTED_IN_BYTES': 'int64', 'RETRANSMITTED_OUT_BYTES': 'int64',
    'RETRANSMITTED_IN_PKTS': 'int32', 'RETRANSMITTED_OUT_PKTS': 'int32',
    # IAT 统计（浮点，因为是平均值/标准差）
    'SRC_TO_DST_IAT_AVG': 'float32', 'DST_TO_SRC_IAT_AVG': 'float32',
    'SRC_TO_DST_IAT_MIN': 'float32', 'SRC_TO_DST_IAT_MAX': 'float32',
    'SRC_TO_DST_IAT_STDDEV': 'float32',
    'DST_TO_SRC_IAT_MIN': 'float32', 'DST_TO_SRC_IAT_MAX': 'float32',
    'DST_TO_SRC_IAT_STDDEV': 'float32',
    # TTL（整数）
    'MIN_TTL': 'int16', 'MAX_TTL': 'int16',
    # 包长度统计（整数）
    'LONGEST_FLOW_PKT': 'int32', 'SHORTEST_FLOW_PKT': 'int32',
    'MIN_IP_PKT_LEN': 'int32', 'MAX_IP_PKT_LEN': 'int32',
    # 包大小分布计数（整数）
    'NUM_PKTS_UP_TO_128_BYTES': 'int32', 'NUM_PKTS_128_TO_256_BYTES': 'int32',
    'NUM_PKTS_256_TO_512_BYTES': 'int32', 'NUM_PKTS_512_TO_1024_BYTES': 'int32',
    'NUM_PKTS_1024_TO_1514_BYTES': 'int32',
    # TCP 窗口（整数）
    'TCP_WIN_MAX_IN': 'int32', 'TCP_WIN_MAX_OUT': 'int32',
    # ICMP（整数）
    'ICMP_TYPE': 'int16', 'ICMP_IPV4_TYPE': 'int16',
    # 吞吐量（整数）
    'SRC_TO_DST_AVG_THROUGHPUT': 'int64', 'DST_TO_SRC_AVG_THROUGHPUT': 'int64',
    # 每秒字节数（浮点，因为是比率）
    'SRC_TO_DST_SECOND_BYTES': 'float32', 'DST_TO_SRC_SECOND_BYTES': 'float32',
    # DNS 相关（整数）
    'DNS_QUERY_ID': 'int32', 'DNS_QUERY_TYPE': 'int16', 'DNS_TTL_ANSWER': 'int32',
    # FTP（整数）
    'FTP_COMMAND_RET_CODE': 'int16',
    # L7 协议（浮点，因为数据中有小数）
    'L7_PROTO': 'float32',
}

# --- 特征选择策略 ---
EXCLUDE_COLS = [
    'id', 'label', LABEL_COLUMN, ATTACK_CAT_COLUMN, 
    'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
    'service_port', 'ip_min', 'ip_max', 'aggr_key', 'sub_session_id', 
    'new_session_flag', 'time_diff', 'Session_ID'
]

CORE_FLOW_FEATURES = [
    'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS', 'PROTOCOL',
    'TCP_FLAGS_ENCODED', 'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_OUT_BYTES',
    'PKT_SIZE_AVG_IN', 'PKT_SIZE_AVG_OUT', 'SRC_PORT_CATEGORY', 'DST_PORT_CATEGORY',
    'SRC_TO_DST_IAT_AVG', 'DST_TO_SRC_IAT_AVG'
]

SESSION_STAT_COLS = [
    'SESS_FLOW_COUNT', 'SESS_UNIQUE_PORT', 'SESS_TCP_RATIO', 'SESS_FLOW_DUR_VAR', 'SESS_TOTAL_BYTES'
]

TCP_FLAGS_MAP = {1: 1, 2: 2, 4: 3, 8: 4, 16: 5, 32: 6, 64: 7, 128: 8, 256: 9, 3: 10, 18: 11, 20: 12, 17: 13, 34: 14}


# --- 辅助函数 ---
def get_dataset_name(filename):
    return filename[:-4] if filename.endswith('.csv') else filename

def load_column_names_from_data(data_path):
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
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f) - 1


def get_sample_ratio_for_category(category_name, category_count):
    """根据类别名称和样本数量确定采样比例"""
    if category_name.lower() == 'benign':
        return BENIGN_SAMPLE_RATIO
    
    for threshold_min, threshold_max, ratio, _ in SAMPLE_THRESHOLDS:
        if threshold_min <= category_count < threshold_max:
            return ratio
    
    return 1.0  # 默认全部保留


# --- 向量化特征加工函数 ---
def encode_tcp_flags_vectorized(flags_series):
    flags = flags_series.fillna(0).astype(int)
    return flags.map(TCP_FLAGS_MAP).fillna(15).astype(int)

def encode_port_vectorized(port_series):
    port = port_series.fillna(0).astype(int)
    conditions = [(port >= 1) & (port <= 1024), port.isin([8080, 3306]), (port >= 1025) & (port <= 65535)]
    return np.select(conditions, [1, 2, 3], default=4).astype(np.int8)

def calculate_pkt_size_avg_vectorized(bytes_series, pkts_series):
    bytes_vals = bytes_series.fillna(0).astype(float)
    pkts_vals = pkts_series.fillna(0).astype(float)
    return np.round(np.where(pkts_vals != 0, bytes_vals / pkts_vals, 0.0), 4)


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


def convert_column_to_string(col_data):
    """
    智能转换列数据为字符串，保留有效数值
    """
    col_data = col_data.fillna(0)
    # 对于浮点数，智能格式化
    if col_data.dtype in ['float32', 'float64']:
        # 检查是否所有值都可以无损转为整数
        try:
            is_all_int = (col_data == col_data.astype(np.int64)).all()
            if is_all_int:
                return col_data.astype(np.int64).astype(str)
        except (ValueError, OverflowError):
            pass
        # 有小数值，逐个格式化
        return col_data.apply(format_numeric_value)
    else:
        return col_data.astype(str)


def build_text_column_vectorized(df, feature_cols, batch_size=None):
    """
    分批构建文本列，避免内存溢出
    智能处理数值格式：整数保持整数，小数去除尾部无效0
    batch_size: 批大小，None则自动根据数据量决定
    """
    n_rows = len(df)
    
    # 自动确定批大小
    if batch_size is None:
        if n_rows < 500000:  # 小于50万，不分批
            batch_size = n_rows
            print(f"  使用向量化方式构建文本列（数据量小，不分批）...")
        elif n_rows < 2000000:  # 50万-200万，使用较大批大小
            batch_size = 200000
            print(f"  使用向量化方式构建文本列（分批处理，批大小={batch_size:,}）...")
        else:  # 超过200万，使用较小批大小
            batch_size = 100000
            print(f"  使用向量化方式构建文本列（分批处理，批大小={batch_size:,}）...")
    else:
        print(f"  使用向量化方式构建文本列（分批处理，批大小={batch_size:,}）...")
    
    n_batches = (n_rows + batch_size - 1) // batch_size
    
    if n_batches == 1:
        # 不分批，直接处理
        str_cols = []
        for col in feature_cols:
            col_str = convert_column_to_string(df[col])
            str_cols.append(col_str)
        text_df = pd.concat(str_cols, axis=1)
        text_df.columns = feature_cols
        return text_df.apply(lambda row: ' '.join(row.values), axis=1)
    
    # 分批处理
    text_results = []
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_rows)
        
        if n_batches <= 10 or batch_idx % max(1, n_batches // 10) == 0:  # 最多显示10次进度
            print(f"    处理批次 {batch_idx + 1}/{n_batches} ({start_idx:,}-{end_idx:,})")
        
        df_batch = df.iloc[start_idx:end_idx]
        
        str_cols = []
        for col in feature_cols:
            col_str = convert_column_to_string(df_batch[col])
            str_cols.append(col_str)
        
        text_df_batch = pd.concat(str_cols, axis=1)
        text_df_batch.columns = feature_cols
        text_batch = text_df_batch.apply(lambda row: ' '.join(row.values), axis=1)
        text_results.append(text_batch)
        
        del df_batch, str_cols, text_df_batch, text_batch
        gc.collect()
    
    print("    合并所有批次...")
    return pd.concat(text_results, ignore_index=True)

def compute_prefix_session_stats_vectorized(df):
    """向量化计算前缀累积会话特征"""
    print("  正在计算会话前缀特征...")
    df = df.sort_values(by=['Session_ID', TIME_COLUMN]).reset_index(drop=True)
    
    # 1. 前缀流计数
    df['SESS_FLOW_COUNT'] = df.groupby('Session_ID').cumcount() + 1
    
    # 2. TCP 前缀比例
    df['_is_tcp'] = (df['PROTOCOL'] == 6).astype(np.float32)
    df['_tcp_cum'] = df.groupby('Session_ID')['_is_tcp'].cumsum()
    df['SESS_TCP_RATIO'] = np.round(df['_tcp_cum'] / df['SESS_FLOW_COUNT'], 4)
    df.drop(columns=['_is_tcp', '_tcp_cum'], inplace=True)
    
    # 3. 前缀总字节数
    df['_total_bytes'] = df['IN_BYTES'].fillna(0).astype(np.float32) + df['OUT_BYTES'].fillna(0).astype(np.float32)
    df['SESS_TOTAL_BYTES'] = df.groupby('Session_ID')['_total_bytes'].cumsum()
    df.drop(columns=['_total_bytes'], inplace=True)
    
    # 4. 端口类别前缀唯一数（优化版：避免 expanding().apply()）
    df['SESS_UNIQUE_PORT'] = _compute_prefix_nunique_optimized(df, 'Session_ID', 'DST_PORT_CATEGORY')
    
    # 5. 持续时间前缀方差（优化版：使用 Welford 算法）
    df['_dur'] = df['FLOW_DURATION_MILLISECONDS'].fillna(0).astype(np.float32)
    df['SESS_FLOW_DUR_VAR'] = _compute_prefix_variance_optimized(df, 'Session_ID', '_dur')
    df.drop(columns=['_dur'], inplace=True)
    
    for col in SESSION_STAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0
    
    return df


def _compute_prefix_nunique_optimized(df, group_col, value_col):
    """
    高效计算前缀唯一值数量
    使用标记数组避免重复计算，时间复杂度从 O(n²) 降到 O(n)
    """
    result = np.zeros(len(df), dtype=np.int32)
    
    for _, group in df.groupby(group_col, sort=False):
        indices = group.index.values
        values = group[value_col].values
        
        # 使用集合追踪已见过的值
        seen = set()
        for i, (idx, val) in enumerate(zip(indices, values)):
            seen.add(val)
            result[idx] = len(seen)
    
    return result


def _compute_prefix_variance_optimized(df, group_col, value_col):
    """
    高效计算前缀方差（Welford's online algorithm）
    避免 expanding().var() 的重复计算
    """
    result = np.zeros(len(df), dtype=np.float32)
    
    for _, group in df.groupby(group_col, sort=False):
        indices = group.index.values
        values = group[value_col].values.astype(np.float64)
        
        # Welford 算法计算在线方差
        n = 0
        mean = 0.0
        M2 = 0.0
        
        for i, (idx, val) in enumerate(zip(indices, values)):
            n += 1
            delta = val - mean
            mean += delta / n
            delta2 = val - mean
            M2 += delta * delta2
            
            # 方差 = M2 / n (或 M2 / (n-1) 用于样本方差)
            if n > 1:
                result[idx] = M2 / n
            else:
                result[idx] = 0.0
    
    return np.round(result, 4)


# --- 数据划分函数 ---
def _compute_split_counts(total, ratios):
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
        split_frames[split_name] = split_frames[split_name].sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return split_frames

def export_fusion_data(df, output_dir, label_mode):
    """导出 Fusion 数据"""
    output_cols = ['text', 'label'] + SESSION_STAT_COLS
    fusion_df = df[output_cols].copy()
    
    print(f"\n[数据统计 - {label_mode}]")
    print(f"  总样本数: {len(fusion_df):,}")
    for lbl, cnt in fusion_df['label'].value_counts().sort_index().items():
        print(f"  label={lbl:<3d} 数量={cnt:<8d} 占比={cnt/len(fusion_df)*100:6.2f}%")
    
    # 按标签分层划分
    splits = stratified_split_by_label(fusion_df, 'label')
    for split_name, split_df in splits.items():
        out_path = os.path.join(output_dir, f"{split_name}_fusion.csv")
        split_df[output_cols].to_csv(out_path, index=False, encoding='utf-8')
        print(f"\n[{split_name} 集标签分布 - {label_mode}]")
        total = len(split_df)
        if total == 0:
            print("  (空集)")
            continue
        for lbl, cnt in split_df['label'].value_counts().sort_index().items():
            print(f"  label={lbl:<3d} 数量={cnt:<8d} 占比={cnt/total*100:6.2f}%")
    
    print(f"\nFusion 数据已导出: {output_dir}/[train|val|test]_fusion.csv")


def process_single_chunk_for_gpt2(df_chunk, chunk_idx, attack_type_mapping):
    """
    处理单个数据块（用于GPT-2）：会话聚合 + 特征计算，不进行采样
    返回完整的会话数据
    """
    print(f"\n  --- 块 {chunk_idx} 处理开始 ---")
    print(f"  原始行数: {len(df_chunk):,}")
    
    # 1. 基础数据清洗
    df_chunk[TIME_COLUMN] = pd.to_numeric(df_chunk[TIME_COLUMN], errors='coerce') / 1000.0
    df_chunk = df_chunk.dropna(subset=[TIME_COLUMN])
    
    numeric_cols = ['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'PROTOCOL', 'TCP_FLAGS',
                    'L4_SRC_PORT', 'L4_DST_PORT', 'FLOW_DURATION_MILLISECONDS',
                    'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_OUT_BYTES',
                    'SRC_TO_DST_IAT_AVG', 'DST_TO_SRC_IAT_AVG', LABEL_COLUMN]
    for col in numeric_cols:
        if col in df_chunk.columns:
            df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce').fillna(0)
    
    # 2. 处理 Attack 列
    if ATTACK_CAT_COLUMN in df_chunk.columns:
        attack_series = df_chunk[ATTACK_CAT_COLUMN].fillna('Benign').astype(str).str.strip()
        df_chunk[ATTACK_CAT_COLUMN] = attack_series
        df_chunk['label_multiclass'] = df_chunk[ATTACK_CAT_COLUMN].map(attack_type_mapping).fillna(0).astype(int)
    else:
        df_chunk['label_multiclass'] = 0
    
    df_chunk['label_binary'] = (df_chunk[LABEL_COLUMN] != 0).astype(np.int8)
    
    # 3. 流级特征加工
    df_chunk['TCP_FLAGS_ENCODED'] = encode_tcp_flags_vectorized(df_chunk['TCP_FLAGS'])
    df_chunk['SRC_PORT_CATEGORY'] = encode_port_vectorized(df_chunk['L4_SRC_PORT'])
    df_chunk['DST_PORT_CATEGORY'] = encode_port_vectorized(df_chunk['L4_DST_PORT'])
    df_chunk['PKT_SIZE_AVG_IN'] = calculate_pkt_size_avg_vectorized(df_chunk['IN_BYTES'], df_chunk['IN_PKTS'])
    df_chunk['PKT_SIZE_AVG_OUT'] = calculate_pkt_size_avg_vectorized(df_chunk['OUT_BYTES'], df_chunk['OUT_PKTS'])
    df_chunk.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT', 'TCP_FLAGS'], errors='ignore', inplace=True)
    
    # 4. 会话聚合
    df_chunk['ip_min'] = np.where(df_chunk['IPV4_SRC_ADDR'] < df_chunk['IPV4_DST_ADDR'], 
                                   df_chunk['IPV4_SRC_ADDR'], df_chunk['IPV4_DST_ADDR'])
    df_chunk['ip_max'] = np.where(df_chunk['IPV4_SRC_ADDR'] < df_chunk['IPV4_DST_ADDR'], 
                                   df_chunk['IPV4_DST_ADDR'], df_chunk['IPV4_SRC_ADDR'])
    df_chunk['aggr_key'] = (df_chunk['ip_min'].astype(str) + '_' + df_chunk['ip_max'].astype(str) + '_' +
                            df_chunk['DST_PORT_CATEGORY'].astype(str) + '_' + df_chunk['PROTOCOL'].astype(str))
    
    df_chunk.sort_values(by=['aggr_key', TIME_COLUMN], inplace=True)
    df_chunk['time_diff'] = df_chunk.groupby('aggr_key')[TIME_COLUMN].diff().fillna(0)
    df_chunk['new_session_flag'] = (df_chunk['time_diff'] > SESSION_TIMEOUT).astype(np.int8)
    df_chunk['sub_session_id'] = df_chunk.groupby('aggr_key')['new_session_flag'].cumsum()
    
    # 添加块标识避免 Session_ID 冲突
    df_chunk['Session_ID'] = f"c{chunk_idx}_" + df_chunk['aggr_key'] + '_' + df_chunk['sub_session_id'].astype(str)
    
    # 5. 过滤短会话
    session_counts = df_chunk.groupby('Session_ID').size()
    valid_sessions = session_counts[session_counts >= MIN_SEQ_LEN].index
    df_chunk = df_chunk[df_chunk['Session_ID'].isin(valid_sessions)].reset_index(drop=True)
    print(f"  过滤短会话后: {len(df_chunk):,} 行, {len(valid_sessions):,} 个会话")
    
    if len(df_chunk) == 0:
        return None
    
    # 6. 计算会话特征
    df_chunk = compute_prefix_session_stats_vectorized(df_chunk)
    
    print(f"  处理完成: {len(df_chunk):,} 行")
    return df_chunk


def sample_fusion_data_from_chunk(df_chunk, sampling_strategy, rare_categories, chunk_idx):
    """
    对单个块的数据按流级别进行分层采样（用于Fusion模型）
    sampling_strategy: {attack_type: sample_ratio}
    rare_categories: set of attack types that need boosting
    """
    print(f"  块 {chunk_idx} Fusion数据采样...")
    
    sampled_dfs = []
    
    for attack_type, ratio in sampling_strategy.items():
        # 筛选该类别的流
        type_mask = df_chunk[ATTACK_CAT_COLUMN] == attack_type
        type_df = df_chunk[type_mask]
        
        if len(type_df) == 0:
            continue
        
        # 对极小类别应用保护机制（过采样以补偿会话过滤损失）
        effective_ratio = ratio
        if attack_type in rare_categories:
            effective_ratio = min(ratio * RARE_ATTACK_BOOST, 1.0)
            print(f"    {attack_type}: 应用小样本保护 (×{RARE_ATTACK_BOOST})")
        
        if effective_ratio >= 1.0:
            # 全部保留
            sampled_dfs.append(type_df)
            print(f"    {attack_type}: {len(type_df):,} -> {len(type_df):,} (全部保留)")
        else:
            # 按比例采样
            keep_n = max(1, int(len(type_df) * effective_ratio))
            sampled = type_df.sample(n=keep_n, random_state=RANDOM_SEED + chunk_idx, replace=False)
            sampled_dfs.append(sampled)
            print(f"    {attack_type}: {len(type_df):,} -> {len(sampled):,} (采样 {effective_ratio*100:.0f}%)")
    
    if sampled_dfs:
        result = pd.concat(sampled_dfs, ignore_index=True)
        print(f"  块 {chunk_idx} 采样后: {len(result):,} 行")
        return result
    return None


def process_netflow_data_chunked():
    """分块处理大数据集"""
    dataset_name = get_dataset_name(INPUT_FILENAME)
    print(f"[{'='*20} 分块数据预处理 - {dataset_name} {'='*20}]")
    print(f"分块数: {NUM_PROCESS_CHUNKS}")
    print(f"\n采样策略:")
    print(f"  - 正常样本 (Benign): {BENIGN_SAMPLE_RATIO*100:.0f}%")
    for threshold_min, threshold_max, ratio, desc in SAMPLE_THRESHOLDS:
        if threshold_max == float('inf'):
            print(f"  - {desc}: {ratio*100:.0f}%")
        else:
            print(f"  - {desc}: {ratio*100:.0f}%")
    
    # 创建输出目录
    dataset_output_dir = os.path.join(OUTPUT_DATA_DIR, dataset_name)
    bert_output_dir = os.path.join(dataset_output_dir, 'bert')
    bert_binary_dir = os.path.join(bert_output_dir, 'binary')
    bert_multiclass_dir = os.path.join(bert_output_dir, 'multiclass')
    for d in [dataset_output_dir, bert_output_dir, bert_binary_dir, bert_multiclass_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 1. 读取列名和统计总行数
    data_file_path = os.path.join(INPUT_DATA_DIR, INPUT_FILENAME)
    col_names = load_column_names_from_data(data_file_path)
    if not col_names:
        print("列名读取失败，终止程序")
        return
    
    total_lines = get_file_line_count(data_file_path)
    chunk_size = total_lines // NUM_PROCESS_CHUNKS
    print(f"\n数据集总行数: {total_lines:,}")
    print(f"每块约: {chunk_size:,} 行")
    
    dtype_filtered = {k: v for k, v in DTYPE_SPEC.items() if k in col_names}
    
    # ========== 第一遍扫描：统计攻击类型分布，确定采样策略 ==========
    print("\n[第一遍扫描] 统计攻击类型分布...")
    attack_type_counts = {}
    
    for chunk in tqdm(pd.read_csv(data_file_path, chunksize=chunk_size, header=0, names=col_names,
                                   dtype=dtype_filtered, low_memory=True, encoding='utf-8'),
                      total=NUM_PROCESS_CHUNKS, desc="统计攻击类型"):
        if ATTACK_CAT_COLUMN in chunk.columns:
            attack_series = chunk[ATTACK_CAT_COLUMN].fillna('Benign').astype(str).str.strip()
            for attack_type, count in attack_series.value_counts().items():
                attack_type_counts[attack_type] = attack_type_counts.get(attack_type, 0) + count
    
    # 构建攻击类型映射
    attack_type_mapping = {}
    for t in attack_type_counts.keys():
        if t.lower() == 'benign':
            attack_type_mapping[t] = 0
            break
    current_id = 0
    for t in sorted(attack_type_counts.keys()):
        if t not in attack_type_mapping:
            current_id += 1
            attack_type_mapping[t] = current_id
    
    # 确定每个类别的采样比例（基于全数据集统计）
    sampling_strategy = {}
    rare_categories = set()  # 需要小样本保护的类别
    
    print("\n[攻击类型分布及采样策略]")
    for attack_type, count in sorted(attack_type_counts.items(), key=lambda x: -x[1]):
        ratio = get_sample_ratio_for_category(attack_type, count)
        sampling_strategy[attack_type] = ratio
        label_id = attack_type_mapping.get(attack_type, -1)
        
        # 识别极小类别（<1000），需要保护
        if count < 1000 and attack_type.lower() != 'benign':
            rare_categories.add(attack_type)
        
        # 确定类别描述
        if attack_type.lower() == 'benign':
            category_desc = "正常"
        else:
            for threshold_min, threshold_max, _, desc in SAMPLE_THRESHOLDS:
                if threshold_min <= count < threshold_max:
                    category_desc = desc
                    break
            else:
                category_desc = "未知"
        
        boost_marker = " [小样本保护×3]" if attack_type in rare_categories else ""
        print(f"  {attack_type:<25} -> label={label_id:<3d} 数量={count:<10,} [{category_desc}] 采样{ratio*100:.0f}%{boost_marker}")
    
    # 保存映射
    mapping_path = os.path.join(bert_multiclass_dir, 'attack_type_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(attack_type_mapping, f, ensure_ascii=False, indent=2)
    print(f"\n攻击类型映射已保存: {mapping_path}")
    
    # 保存采样策略
    strategy_path = os.path.join(dataset_output_dir, 'sampling_strategy.json')
    with open(strategy_path, 'w', encoding='utf-8') as f:
        json.dump({
            'attack_type_counts': attack_type_counts,
            'sampling_strategy': sampling_strategy,
            'rare_categories': list(rare_categories)
        }, f, ensure_ascii=False, indent=2)
    print(f"采样策略已保存: {strategy_path}")
    
    # 预估采样后的数据量
    estimated_total = 0
    print("\n[预估采样后数据量]")
    for attack_type, count in attack_type_counts.items():
        ratio = sampling_strategy.get(attack_type, 1.0)
        if attack_type in rare_categories:
            ratio = min(ratio * RARE_ATTACK_BOOST, 1.0)
        estimated = int(count * ratio * 0.7)  # 0.7系数：考虑短会话过滤损失
        estimated_total += estimated
        if estimated > 1000:  # 只显示超过1000的类别
            print(f"  {attack_type:<25}: ~{estimated:,}")
    print(f"  {'总计':<25}: ~{estimated_total:,}")
    print(f"  注：实际数量可能因会话过滤略有不同")
    
    # ========== 第二遍扫描：分块处理 ==========
    print(f"\n[第二遍扫描] 分块处理数据...")
    all_gpt2_chunks = []  # 用于GPT-2的完整数据
    all_fusion_chunks = []  # 用于Fusion的采样数据
    
    chunk_idx = 0
    for chunk_df in pd.read_csv(data_file_path, chunksize=chunk_size, header=0, names=col_names,
                                 dtype=dtype_filtered, low_memory=True, encoding='utf-8'):
        chunk_idx += 1
        print(f"\n{'='*60}")
        print(f"处理第 {chunk_idx}/{NUM_PROCESS_CHUNKS} 块")
        print(f"{'='*60}")
        
        # 处理块数据（会话聚合、特征计算，不采样）
        processed_chunk = process_single_chunk_for_gpt2(chunk_df, chunk_idx, attack_type_mapping)
        
        if processed_chunk is not None and len(processed_chunk) > 0:
            # 保存用于GPT-2的完整数据
            all_gpt2_chunks.append(processed_chunk.copy())
            
            # 对Fusion数据进行流级别采样
            fusion_chunk = sample_fusion_data_from_chunk(processed_chunk, sampling_strategy, rare_categories, chunk_idx)
            if fusion_chunk is not None:
                all_fusion_chunks.append(fusion_chunk)
        
        del chunk_df, processed_chunk
        gc.collect()
    
    # ========== 合并GPT-2数据 ==========
    print(f"\n[合并GPT-2数据] 合并 {len(all_gpt2_chunks)} 个块...")
    df_gpt2 = pd.concat(all_gpt2_chunks, ignore_index=True)
    del all_gpt2_chunks
    gc.collect()
    
    print(f"GPT-2数据合并完成:")
    print(f"  总行数: {len(df_gpt2):,}")
    print(f"  总会话数: {df_gpt2['Session_ID'].nunique():,}")
    
    # ========== GPT-2数据会话级采样（按标签分层）==========
    if GPT2_SESSION_SAMPLE_RATIO < 1.0:
        print(f"\n[GPT-2会话采样] 采样 {GPT2_SESSION_SAMPLE_RATIO*100:.0f}% 会话用于训练...")
        
        # 按会话聚合，获取每个会话的标签（取众数）
        session_labels = df_gpt2.groupby('Session_ID')['label_binary'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        ).reset_index()
        session_labels.columns = ['Session_ID', 'session_label']
        
        # 按标签分层采样会话
        sampled_sessions = []
        for label in session_labels['session_label'].unique():
            label_sessions = session_labels[session_labels['session_label'] == label]['Session_ID'].values
            n_sample = max(1, int(len(label_sessions) * GPT2_SESSION_SAMPLE_RATIO))
            sampled = np.random.choice(label_sessions, size=n_sample, replace=False)
            sampled_sessions.extend(sampled)
            print(f"  标签{label}: {len(label_sessions):,} 会话 -> 采样 {len(sampled):,} 会话")
        
        # 过滤数据
        df_gpt2_original_size = len(df_gpt2)
        df_gpt2 = df_gpt2[df_gpt2['Session_ID'].isin(sampled_sessions)].reset_index(drop=True)
        
        print(f"\n采样后GPT-2数据:")
        print(f"  总行数: {len(df_gpt2):,} (原始: {df_gpt2_original_size:,}, 压缩: {len(df_gpt2)/df_gpt2_original_size*100:.1f}%)")
        print(f"  总会话数: {df_gpt2['Session_ID'].nunique():,}")
        
        gc.collect()
    else:
        print(f"\n使用全部会话数据用于GPT-2")
    
    # ========== 合并Fusion数据 ==========
    print(f"\n[合并Fusion数据] 合并 {len(all_fusion_chunks)} 个块...")
    df_fusion = pd.concat(all_fusion_chunks, ignore_index=True)
    del all_fusion_chunks
    gc.collect()
    
    print(f"Fusion数据总行数: {len(df_fusion):,}")
    
    # 统计Fusion数据标签分布
    print(f"\n[Fusion数据标签分布]")
    print(f"二分类:")
    for lbl, cnt in df_fusion['label_binary'].value_counts().sort_index().items():
        print(f"  label={lbl}: {cnt:,} ({cnt/len(df_fusion)*100:.2f}%)")
    
    print(f"\n多分类:")
    for lbl, cnt in df_fusion['label_multiclass'].value_counts().sort_index().items():
        type_name = [k for k, v in attack_type_mapping.items() if v == lbl]
        type_name = type_name[0] if type_name else "Unknown"
        print(f"  label={lbl} ({type_name}): {cnt:,} ({cnt/len(df_fusion)*100:.2f}%)")
    
    # ========== 构建文本列 ==========
    print("\n[构建特征]")
    all_cols = df_gpt2.columns
    text_feature_cols = [feat for feat in CORE_FLOW_FEATURES if feat in all_cols and feat not in EXCLUDE_COLS]
    print(f"流级特征: {len(text_feature_cols)} 个")
    print(f"会话级特征: {len(SESSION_STAT_COLS)} 个")
    
    print("\n构建GPT-2文本列...")
    df_gpt2['text'] = build_text_column_vectorized(df_gpt2, text_feature_cols)
    
    print("构建Fusion文本列...")
    df_fusion['text'] = build_text_column_vectorized(df_fusion, text_feature_cols)
    
    # 保存特征列表
    with open(os.path.join(dataset_output_dir, 'feature_columns.json'), 'w') as f:
        json.dump({'flow_text_features': text_feature_cols, 'session_stat_features': SESSION_STAT_COLS}, f, indent=4)
    
    # ========== GPT-2数据集划分和生成 ==========
    print("\n[GPT-2数据集划分]")
    unique_sessions = df_gpt2['Session_ID'].unique()
    np.random.shuffle(unique_sessions)
    n_total = len(unique_sessions)
    n_train, n_val, n_test = int(n_total * 0.7), int(n_total * 0.1), int(n_total * 0.1)
    train_ids = unique_sessions[:n_train]
    val_ids = unique_sessions[n_train:n_train+n_val]
    test_ids = unique_sessions[n_train+n_val:n_train+n_val+n_test]
    gen_ids = unique_sessions[n_train+n_val+n_test:]
    print(f"总会话: {n_total:,}, 训练: {n_train:,}, 验证: {n_val:,}, 测试: {n_test:,}, 生成: {len(gen_ids):,}")
    
    def generate_gpt_file(ids, filename):
        output_path = os.path.join(dataset_output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            subset = df_gpt2[df_gpt2['Session_ID'].isin(ids)].sort_values(['Session_ID', TIME_COLUMN])
            for _, session_data in tqdm(subset.groupby('Session_ID'), desc=f"生成 {filename}"):
                session_unique = session_data.drop_duplicates(subset=text_feature_cols)
                if len(session_unique) < MIN_SEQ_LEN:
                    continue
                if len(session_unique) > MAX_SEQ_LEN:
                    session_unique = session_unique.iloc[:MAX_SEQ_LEN]
                f.write("<bos>\n")
                f.write("\n".join(session_unique['text'].tolist()))
                f.write("\n<eos>\n\n")
    
    print("\n[生成 GPT-2 数据]")
    generate_gpt_file(train_ids, 'train_gpt2_input.txt')
    generate_gpt_file(val_ids, 'val_gpt2_input.txt')
    generate_gpt_file(test_ids, 'test_gpt2_input.txt')
    generate_gpt_file(gen_ids, 'gen_input.txt')
    
    # 释放GPT-2数据内存
    del df_gpt2
    gc.collect()
    
    # ========== 生成 Fusion 模型数据 ==========
    print("\n" + "="*60)
    print("[生成 Fusion 模型数据]")
    print("="*60)
    
    print("\n>>> 二分类 (binary)")
    df_fusion['label'] = df_fusion['label_binary']
    export_fusion_data(df_fusion, bert_binary_dir, 'binary')
    
    if df_fusion['label_multiclass'].notna().any():
        print("\n>>> 多分类 (multiclass)")
        df_fusion['label'] = df_fusion['label_multiclass'].astype(int)
        export_fusion_data(df_fusion, bert_multiclass_dir, 'multiclass')
    
    # ========== 结果汇总 ==========
    print(f"\n[{'='*20} 数据预处理完成 {'='*20}]")
    print(f"数据集: {dataset_name}")
    print(f"\n输出文件:")
    print(f"1. GPT-2 数据（完整会话）: {dataset_output_dir}/[train/val/test/gen]_gpt2_input.txt")
    print(f"2. Fusion 二分类（采样后）: {bert_binary_dir}/[train/val/test]_fusion.csv")
    print(f"3. Fusion 多分类（采样后）: {bert_multiclass_dir}/[train/val/test]_fusion.csv")
    print(f"4. 特征列表: {dataset_output_dir}/feature_columns.json")
    print(f"5. 攻击类型映射: {mapping_path}")
    print(f"6. 采样策略: {strategy_path}")


if __name__ == "__main__":
    process_netflow_data_chunked()
