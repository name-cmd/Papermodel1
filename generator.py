# -*- coding: utf-8 -*-
"""
NetGPT 主动式流量生成与检测框架 (Paper Implementation)
阶段 1: GPT-2 基于会话上下文主动生成下一跳流量
阶段 2: DistilBERT 对生成流量进行实时恶意检测

输出文件:
- active_detection_results.txt: 带检测结果的完整日志
- generated_flows_only.txt: 仅包含生成的流量文本（供评估使用）
"""
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DistilBertTokenizer, DistilBertForSequenceClassification
import os
import json

# 导入评估模块中的有效性检查函数
from evaluation import compute_valid_flow_ratio

# --- 配置 ---
# 默认数据集名称，可通过命令行参数覆盖
DEFAULT_DATASET = 'NF-UNSW-NB15-v3'
DEFAULT_GPT2_MODEL_PATH = './models/gpt2-finetuned-traffic/best-model'
DEFAULT_BERT_MODEL_PATH = './models/distilbert-finetuned'
PROCESSED_DATA_BASE_DIR = './processed_data'
OUTPUT_DIR = './generated_data'

CONTEXT_WINDOW = 2        # 修改为2，适应短会话
MAX_GEN_LEN = 10          # 主动预测未来多少个流
MAX_FLOW_TOKENS = 128     # 单个流的最大Token数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. 加载模型（将在 run_active_detection 函数中调用）---
def load_models(dataset_name, gpt2_model_path, bert_model_path, mode='binary'):
    """加载 GPT-2 和 DistilBERT 模型"""
    print("正在加载模型...")
    # GPT-2 (生成器)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path).to(device)
    gpt2_model.eval()

    # DistilBERT (检测器)
    bert_tokenizer = DistilBertTokenizer.from_pretrained(bert_model_path)
    bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_path).to(device)
    bert_model.eval()

    # 加载特征列定义与训练配置信息
    feature_cols_path = os.path.join(PROCESSED_DATA_BASE_DIR, dataset_name, 'feature_columns.json')
    with open(feature_cols_path, 'r', encoding='utf-8') as f:
        feature_cfg = json.load(f)
    
    return gpt2_tokenizer, gpt2_model, bert_tokenizer, bert_model, feature_cfg

# 全局变量（将在 run_active_detection 函数中设置）
FLOW_TEXT_FEATURES = []
SESSION_STAT_FEATURES = []
EXPECTED_FIELD_COUNT = 0
USE_SESSION_TOKENS = False
SESSION_STAT_FEATURES_FOR_MODEL = []
NUM_LABELS = 2

# --- 2. 辅助函数 ---

def _format_session_stat_value(v):
    """与 distilbert_training.py 中一致的数值格式化逻辑。"""
    try:
        fv = float(v)
        if abs(fv - round(fv)) < 1e-6:
            return str(int(round(fv)))
        return f"{fv:.4f}"
    except Exception:
        return str(v)


def build_flow_with_session_tokens(flow_text, session_stats: dict):
    """
    将当前流文本与前缀会话统计编码为伪 token 拼接，格式需与训练阶段保持一致。
    """
    if not USE_SESSION_TOKENS or not SESSION_STAT_FEATURES_FOR_MODEL:
        return flow_text

    parts = []
    for col in SESSION_STAT_FEATURES_FOR_MODEL:
        if col in session_stats:
            parts.append(f"{col}={_format_session_stat_value(session_stats[col])}")
    if not parts:
        return flow_text
    session_str = " ".join(parts)
    return f"{flow_text} [SEP] {session_str}"


def detect_anomaly(flow_text, bert_tokenizer, bert_model, session_stats=None):
    """
    阶段 2: 实时检测
    输入:
      - flow_text: 单条流的 values-only 文本
      - bert_tokenizer: DistilBERT tokenizer
      - bert_model: DistilBERT 模型
      - session_stats: 当前会话到此流为止的前缀统计（字典形式）
    输出: Label - "Normal" 或 "ATTACK"
    """
    full_text = build_flow_with_session_tokens(flow_text, session_stats or {})
    inputs = bert_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        pred_label = torch.argmax(outputs.logits, dim=1).item()
    
    label_str = "ATTACK" if pred_label == 1 else "Normal"
    return label_str

def is_valid_flow(flow_text):
    """
    简单的结构校验：检查生成的流是否包含大致正确的字段数量。
    使用 evaluation 模块中的逻辑保持一致性。
    """
    parts = flow_text.strip().split()
    # 允许一定的误差 (例如某些字段包含空格被误切分，或者生成了额外空格)
    # 这里设置宽松界限，主要防止生成完全乱码
    min_fields = int(EXPECTED_FIELD_COUNT * 0.7)
    max_fields = int(EXPECTED_FIELD_COUNT * 1.3)
    return min_fields <= len(parts) <= max_fields


def parse_flow_to_dict(flow_text):
    """
    将单行 values-only 文本解析为 {特征名: 数值字符串} 映射。
    解析失败时，仅返回能对齐长度的部分，其余特征缺失。
    """
    tokens = flow_text.strip().split()
    feat_count = min(len(tokens), len(FLOW_TEXT_FEATURES))
    values = tokens[:feat_count]
    return {FLOW_TEXT_FEATURES[i]: values[i] for i in range(feat_count)}


def update_prefix_session_stats(prefix_state, flow_feat_dict):
    """
    根据当前流的特征字典，更新前缀会话统计，返回用于拼接伪 token 的统计字典。
    该逻辑需与 data_preprocessing.compute_prefix_session_stats 保持语义一致：
      - SESS_FLOW_COUNT     : 前缀流数
      - SESS_UNIQUE_PORT    : 前缀内不同 DST_PORT_CATEGORY 数量
      - SESS_TCP_RATIO      : 前缀 TCP 比例
      - SESS_FLOW_DUR_VAR   : 前缀 FLOW_DURATION_MILLISECONDS 方差（Welford）
      - SESS_TOTAL_BYTES    : 前缀 IN+OUT 字节总和
    """
    # 初始化前缀状态
    if not prefix_state:
        prefix_state.update({
            'count': 0,
            'ports': set(),
            'tcp_count': 0,
            'mean_dur': 0.0,
            'm2_dur': 0.0,
            'total_bytes': 0.0
        })

    # 解析所需字段
    def _to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    in_bytes = _to_float(flow_feat_dict.get('IN_BYTES', 0.0))
    out_bytes = _to_float(flow_feat_dict.get('OUT_BYTES', 0.0))
    dur = _to_float(flow_feat_dict.get('FLOW_DURATION_MILLISECONDS', 0.0))
    proto = _to_float(flow_feat_dict.get('PROTOCOL', 0.0))
    dst_port_cat = flow_feat_dict.get('DST_PORT_CATEGORY', None)

    # 更新计数
    prefix_state['count'] += 1
    n = prefix_state['count']

    if dst_port_cat is not None:
        prefix_state['ports'].add(dst_port_cat)

    if int(proto) == 6:
        prefix_state['tcp_count'] += 1

    # Welford 算法更新时长方差
    delta = dur - prefix_state['mean_dur']
    prefix_state['mean_dur'] += delta / n
    delta2 = dur - prefix_state['mean_dur']
    prefix_state['m2_dur'] += delta * delta2

    prefix_state['total_bytes'] += (in_bytes + out_bytes)

    # 派生统计输出
    sess_flow_count = n
    sess_unique_port = len(prefix_state['ports'])
    sess_tcp_ratio = round(prefix_state['tcp_count'] / n, 4) if n > 0 else 0.0
    sess_flow_dur_var = round(prefix_state['m2_dur'] / (n - 1), 4) if n > 1 else 0.0
    sess_total_bytes = prefix_state['total_bytes']

    stats_for_tokens = {
        'SESS_FLOW_COUNT': sess_flow_count,
        'SESS_UNIQUE_PORT': sess_unique_port,
        'SESS_TCP_RATIO': sess_tcp_ratio,
        'SESS_FLOW_DUR_VAR': sess_flow_dur_var,
        'SESS_TOTAL_BYTES': sess_total_bytes
    }
    return stats_for_tokens

# --- 3. 主流程 ---

def run_active_detection(dataset_name=None, gpt2_model_path=None, bert_model_path=None, mode='binary'):
    """
    运行主动检测流程
    
    参数:
        dataset_name: 数据集名称（例如: NF-UNSW-NB15-v3）
        gpt2_model_path: GPT-2 模型路径
        bert_model_path: DistilBERT 模型路径
        mode: 训练模式 ('binary' 或 'multiclass')
    """
    global FLOW_TEXT_FEATURES, SESSION_STAT_FEATURES, EXPECTED_FIELD_COUNT
    global USE_SESSION_TOKENS, SESSION_STAT_FEATURES_FOR_MODEL, NUM_LABELS
    
    # 设置默认值
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET
    if gpt2_model_path is None:
        gpt2_model_path = DEFAULT_GPT2_MODEL_PATH.replace('gpt2-finetuned-traffic', f'gpt2-finetuned-traffic-{dataset_name}')
    if bert_model_path is None:
        bert_model_path = DEFAULT_BERT_MODEL_PATH
    
    # 加载模型和配置
    gpt2_tokenizer, gpt2_model, bert_tokenizer, bert_model, feature_cfg = load_models(
        dataset_name, gpt2_model_path, bert_model_path, mode
    )
    
    # 设置全局变量
    FLOW_TEXT_FEATURES = feature_cfg.get('flow_text_features', [])
    SESSION_STAT_FEATURES = feature_cfg.get('session_stat_features', [])
    EXPECTED_FIELD_COUNT = len(FLOW_TEXT_FEATURES)
    
    # 从 DistilBERT 训练目录读取训练配置（是否使用会话伪 token 及具体列名）
    training_info_path = os.path.join(bert_model_path, 'training_info.json')
    USE_SESSION_TOKENS = False
    SESSION_STAT_FEATURES_FOR_MODEL = SESSION_STAT_FEATURES
    NUM_LABELS = 2
    if os.path.exists(training_info_path):
        with open(training_info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        NUM_LABELS = int(info.get('num_labels', 2))
        USE_SESSION_TOKENS = bool(info.get('use_session_tokens', False))
        # 若训练阶段显式记录了使用的会话特征列，则以训练配置为准
        if 'session_stat_features' in info:
            SESSION_STAT_FEATURES_FOR_MODEL = info['session_stat_features']
    
    print(f"模型加载完成。预期流特征数量: {EXPECTED_FIELD_COUNT}")
    print(f"分类标签数: {NUM_LABELS} | 使用会话伪 token: {USE_SESSION_TOKENS}")
    if USE_SESSION_TOKENS:
        print(f"会话前缀特征列: {SESSION_STAT_FEATURES_FOR_MODEL}")
    
    # 设置输入和输出文件路径
    input_file = os.path.join(PROCESSED_DATA_BASE_DIR, dataset_name, 'gen_input.txt')
    output_file = os.path.join(OUTPUT_DIR, f'active_detection_results_{dataset_name}.txt')
    generated_flows_file = os.path.join(OUTPUT_DIR, f'generated_flows_only_{dataset_name}.txt')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 用于收集所有生成的流量（供评估模块使用）
    all_generated_flows = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        current_session = []
        session_count = 0
        
        for line in f_in:
            line = line.strip()
            if not line: continue
            
            if line == '<bos>':
                current_session = []
                # 为当前会话初始化前缀状态
                prefix_state = {}
            elif line == '<eos>':
                session_count += 1
                if len(current_session) <= CONTEXT_WINDOW:
                    continue # 上下文太短，无法预测
                
                # --- 开始主动预测实验 ---
                # 1. 构建初始上下文 (模拟实时环境，已知前 N 个流)
                history = current_session[:CONTEXT_WINDOW]
                
                # 会话头信息
                f_out.write(f"\n{'='*20} Session #{session_count} {'='*20}\n")

                # 先原样写出前 CONTEXT_WINDOW 个真实流，并逐行给出分类结果
                for flow in history:
                    # 使用前缀状态更新会话统计，并按与训练阶段一致的方式拼接伪 token
                    feat_dict = parse_flow_to_dict(flow)
                    sess_stats = update_prefix_session_stats(prefix_state, feat_dict)
                    f_out.write(f"{flow}\n")
                    label_real = detect_anomaly(flow, bert_tokenizer, bert_model, sess_stats)
                    validity_real = "Valid" if is_valid_flow(flow) else "Broken"
                    f_out.write(f"[分类: {label_real}, 结构:{validity_real}]\n")
                
                # 2. 循环生成未来流
                generated_flows = []
                # 初始 Prompt
                prompt_text = "\n".join(history) + "\n"
                
                stop_generation = False
                
                for step in range(MAX_GEN_LEN):
                    if stop_generation: break
                    
                    # Tokenize
                    inputs = gpt2_tokenizer(prompt_text, return_tensors='pt').to(device)
                    input_len = inputs['input_ids'].shape[1]
                    
                    if input_len > 900: # 防止超出长度
                        break
                        
                    # Generate Next Flow (一行)
                    # 我们希望生成到换行符为止，或者遇到 eos
                    with torch.no_grad():
                        outputs = gpt2_model.generate(
                            **inputs,
                            max_new_tokens=MAX_FLOW_TOKENS,
                            pad_token_id=gpt2_tokenizer.eos_token_id,
                            eos_token_id=gpt2_tokenizer.eos_token_id,
                            do_sample=True,
                            top_p=0.92,
                            top_k=50,
                            temperature=0.8, # 稍微降低温度以提高结构稳定性
                            return_dict_in_generate=True
                        )
                    
                    # 提取新生成的 Tokens
                    new_tokens = outputs.sequences[0][input_len:]
                    
                    # 解码
                    gen_text = gpt2_tokenizer.decode(new_tokens, skip_special_tokens=False)
                    
                    # 检查是否生成了 EOS (会话结束)
                    if '<eos>' in gen_text:
                        stop_generation = True
                        gen_text = gen_text.split('<eos>')[0] # 截取有效部分
                    
                    # 提取第一行作为生成的流 (防止一次生成多行)
                    gen_flow = gen_text.split('\n')[0].strip()
                    
                    if not gen_flow:
                        continue

                    # 更新前缀会话统计
                    gen_feat_dict = parse_flow_to_dict(gen_flow)
                    gen_sess_stats = update_prefix_session_stats(prefix_state, gen_feat_dict)

                    # 3. 实时检测 (DistilBERT)，输入为"单流文本 + 会话前缀伪 token"
                    label = detect_anomaly(gen_flow, bert_tokenizer, bert_model, gen_sess_stats)
                    validity = "Valid" if is_valid_flow(gen_flow) else "Broken"
                    
                    # 生成流：先输出完整流，再单独换行给出分类结果
                    f_out.write(f"{gen_flow}\n")
                    f_out.write(f"[分类: {label}, 结构:{validity}]\n")
                    
                    # 更新上下文 (滑动窗口)
                    history.append(gen_flow)
                    generated_flows.append(gen_flow)
                    
                    # 收集生成的流量（供评估使用）
                    all_generated_flows.append(gen_flow)
                    
                    # 更新 Prompt (保持固定窗口大小，或者累积)
                    # 这里选择保留最近 N 个作为 Prompt，模拟无限流
                    recent_history = history[-CONTEXT_WINDOW:] 
                    prompt_text = "\n".join(recent_history) + "\n"
                
                f_out.write(f"[Done] 预测结束。生成了 {len(generated_flows)} 个流。\n")
                
            else:
                current_session.append(line)
                
            if session_count >= 100: # 仅测试前100个会话
                print("测试完成，停止。")
                break
    
    # 保存仅包含生成流量的文件（供 evaluation.py 使用）
    with open(generated_flows_file, 'w', encoding='utf-8') as f_gen:
        for flow in all_generated_flows:
            f_gen.write(f"{flow}\n")
    
    print(f"\n生成统计:")
    print(f"  总会话数: {session_count}")
    print(f"  生成流量数: {len(all_generated_flows)}")
    print(f"  结果已保存至: {output_file}")
    print(f"  生成流量文件: {generated_flows_file}")
    print(f"\n提示: 运行 'python evaluation.py --generated_data {generated_flows_file}' 进行质量评估")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='NetGPT 主动式流量生成与检测')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                       help=f'数据集名称（例如: NF-UNSW-NB15-v3），默认: {DEFAULT_DATASET}')
    parser.add_argument('--gpt2_model', type=str, default=None,
                       help='GPT-2 模型路径（可选，默认根据数据集名称自动推断）')
    parser.add_argument('--bert_model', type=str, default=None,
                       help='DistilBERT 模型路径（可选，默认: ./models/distilbert-finetuned）')
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass'], default='binary',
                       help='训练模式: binary(二分类) 或 multiclass(多分类)')
    args = parser.parse_args()
    
    run_active_detection(
        dataset_name=args.dataset,
        gpt2_model_path=args.gpt2_model,
        bert_model_path=args.bert_model,
        mode=args.mode
    )