# -*- coding: utf-8 -*-
"""
用于在网络流量数据上微调 GPT-2 模型的训练脚本（会话级序列建模）。

说明：
- 训练数据来自会话级聚合后的文本：每个样本形如
    <bos>
    flow_line_1
    flow_line_2
    ...
    <eos>
  GPT-2 学习的是「会话内按时间排序的流序列」的联合分布。
- GPT-2 内部自带绝对位置编码，配合预处理阶段固定的特征列顺序，
  可以让模型学习到“同一行中不同位置对应不同特征字段”的隐式结构。
- 本脚本同时提供调试用的小样本训练开关，便于快速跑通端到端流程。
"""
import torch
import os
import json
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import platform
from tqdm import tqdm
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from datetime import datetime
import traceback
import time

# 导入评估模块中的困惑度计算函数
from evaluation import compute_perplexity

# --- 1. 全局设置与随机种子 ---
# 设置随机种子以确保实验结果的可复现性
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 设置matplotlib字体以正确显示中文字符
# 如果在非Windows/macOS系统上，可能需要手动安装中文字体


# --- 1.1 调试／采样配置 ---
# 将 DEBUG_SAMPLE_RATIO 设为 < 1.0 即可在训练阶段只用部分会话样本，便于快速调试。
# 正式实验时，请改回 1.0。
DEBUG_SAMPLE_RATIO = 0.1  # 例如 0.1 表示只用 10% 的会话样本

# --- 2. 自定义数据集 ---
class FlowDataset(Dataset):
    """
    自定义数据集，用于加载和处理「会话级」网络流序列。
    每个由 <bos> 和 <eos> 包围的完整会话被视为一个独立样本。
    样本内部是按时间排序的多行流特征文本（Values-only 格式）。
    """
    def __init__(self, tokenizer: GPT2Tokenizer, file_path: str, block_size: int = 1024):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"输入文件未找到: {file_path}")

        print(f"正在从 {file_path} 加载数据...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 按双换行符分割「会话」，每个块中应包含 <bos> ... <eos>
        sessions = text.split('\n\n')
        all_examples = [sess for sess in sessions if sess.strip()]

        # 调试模式：仅使用部分会话样本
        if 0 < DEBUG_SAMPLE_RATIO < 1.0:
            n_total = len(all_examples)
            n_keep = max(1, int(n_total * DEBUG_SAMPLE_RATIO))
            self.examples = all_examples[:n_keep]
            print(f"调试模式启用: 仅使用 {n_keep}/{n_total} 个会话样本进行 GPT-2 训练。")
        else:
            self.examples = all_examples
        
        self.tokenizer = tokenizer
        self.block_size = block_size
        print(f"数据加载完成，共找到 {len(self.examples)} 个会话样本。")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        flow_text = self.examples[i]
        
        # 对单个流文本进行分词
        # 返回一个字典，包含 'input_ids' 和 'attention_mask'
        tokenized_output = self.tokenizer(
            flow_text,
            max_length=self.block_size,
            truncation=True,
            padding=False, # DataCollator会处理批次内的填充
        )
        return tokenized_output

# --- 3. 自定义训练回调函数 ---
class TrainingProgressCallback(TrainerCallback):
    """
    一个统一的回调函数，用于：
    - 显示详细的训练配置。
    - 管理一个总的训练进度条。
    - 在每个评估点和epoch结束时打印格式化的日志。
    - 训练结束后生成并保存损失和学习率曲线图。
    """
    def __init__(self):
        self.pbar = None
        self.training_start_time = None
        self.epoch_start_time = None
        self.log_history = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start_time = time.time()
        self.pbar = tqdm(total=state.max_steps, desc="总训练进度", unit="步")

        print(f"\n{'='*80}")
        print(" " * 28 + "GPT-2 微调任务开始")
        print(f"{'='*80}")
        print(f"模型输出目录: {args.output_dir}")
        print(f"训练轮数 (Epochs): {args.num_train_epochs}")
        print(f"设备批次大小 (Batch Size): {args.per_device_train_batch_size}")
        print(f"梯度累积步数: {args.gradient_accumulation_steps}")
        print(f"总优化步数 (Total Steps): {state.max_steps}")
        print(f"学习率 (Learning Rate): {args.learning_rate}")
        print(f"评估与保存策略: 每 {args.eval_steps} 步")
        print(f"设备: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
        print(f"使用半精度训练 (FP16): {args.fp16}")
        print(f"{'='*80}\n")
        
    def on_step_end(self, args, state, control, **kwargs):
        self.pbar.update(1)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        # 每次评估后，打印带困惑度的日志
        if logs and 'eval_loss' in logs:
            eval_loss = logs['eval_loss']
            perplexity = compute_perplexity(eval_loss)
            print(f"  [评估] > 步数: {state.global_step}, "
                  f"验证损失: {eval_loss:.4f}, "
                  f"困惑度 (Perplexity): {perplexity:.2f}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 收集所有日志，用于后续分析和绘图
        if logs:
            self.log_history.append({**logs, 'step': state.global_step})

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_duration = time.time() - self.epoch_start_time
        current_epoch_logs = [
            log for log in self.log_history 
            if 'loss' in log and int(log.get('epoch', 0)) == int(state.epoch)
        ]
        if current_epoch_logs:
            avg_train_loss = np.mean([log['loss'] for log in current_epoch_logs])
            print(f"\n--- Epoch {int(state.epoch)}/{int(state.num_train_epochs)} 完成 "
                  f"(耗时: {epoch_duration:.2f}s) ---")
            print(f"  [总结] > 平均训练损失: {avg_train_loss:.4f}")
            print("-" * (40 + len(str(int(state.epoch))) + len(str(int(state.num_train_epochs)))))
            

    def on_train_end(self, args, state, control, **kwargs):
        self.pbar.close()
        total_training_time = time.time() - self.training_start_time
        
        print(f"\n{'='*80}")
        print(" " * 32 + "训练完成")
        print(f"{'='*80}")
        print(f"总耗时: {total_training_time / 60:.2f} 分钟")
        print(f"最终模型保存在: {args.output_dir}")
        best_model_checkpoint = state.best_model_checkpoint
        print(f"性能最佳的模型检查点: {best_model_checkpoint}")
        
        # 从log历史中提取最终的评估指标
        final_eval_logs = next((log for log in reversed(self.log_history) if 'eval_loss' in log), None)
        if final_eval_logs:
            final_eval_loss = final_eval_logs['eval_loss']
            final_perplexity = compute_perplexity(final_eval_loss)
            print(f"最终验证损失: {final_eval_loss:.4f}")
            print(f"最终困惑度 (Perplexity): {final_perplexity:.2f}")
        print(f"{'='*80}\n")
        
        self.plot_curves(args.output_dir)
        
        # New: Save log_history to CSV
        csv_path = os.path.join(args.output_dir, 'training_log.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['step', 'epoch', 'loss', 'eval_loss', 'learning_rate', 'perplexity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for log in self.log_history:
                row = {k: log.get(k, '') for k in fieldnames}
                if 'eval_loss' in log:
                    row['perplexity'] = compute_perplexity(log['eval_loss'])
                writer.writerow(row)
        print(f"✅ 训练日志已保存至: {csv_path}")

    def plot_curves(self, output_dir):
        """绘制并保存训练曲线"""
        print("正在生成训练曲线图...")
        os.makedirs(output_dir, exist_ok=True)

        train_steps = [log['step'] for log in self.log_history if 'loss' in log]
        train_losses = [log['loss'] for log in self.log_history if 'loss' in log]
        eval_steps = [log['step'] for log in self.log_history if 'eval_loss' in log]
        eval_losses = [log['eval_loss'] for log in self.log_history if 'eval_loss' in log]
        lr_steps = [log['step'] for log in self.log_history if 'learning_rate' in log]
        learning_rates = [log['learning_rate'] for log in self.log_history if 'learning_rate' in log]

        if not train_steps or not eval_steps:
            print("日志数据不足，无法生成曲线图。")
            return

        fig, ax1 = plt.subplots(figsize=(12, 8))

        # 绘制损失曲线
        ax1.plot(train_steps, train_losses, 'b-', alpha=0.6, label='Training Loss')
        ax1.plot(eval_steps, eval_losses, 'r-o', linewidth=2, label='Validation Loss')
        ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12, color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        # 只在主轴(ax1)上绘制X轴网格线，避免双Y轴网格线冲突
        ax1.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        ax1.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

        # 绘制学习率曲线
        ax2 = ax1.twinx()
        ax2.plot(lr_steps, learning_rates, 'g--', alpha=0.7, label='Learning Rate')
        ax2.set_ylabel('Learning Rate', fontsize=12, color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        # 不在第二个Y轴上绘制网格线，避免与ax1的网格线重叠造成混乱
        ax2.grid(False)
        
        fig.suptitle('GPT-2 Fine-Tuning Training Curves', fontsize=16, fontweight='bold')
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9), 
                   frameon=True, fancybox=True, framealpha=0.9, edgecolor='black')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        plot_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"✅ 训练曲线图已保存至: {plot_path}")


# --- 4. 模型初始化 ---
def initialize_model_and_tokenizer(model_path: str):
    """从本地路径加载预训练的GPT-2模型和分词器。"""
    print(f"正在从 '{model_path}' 初始化模型和分词器...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    # 添加并设置特殊tokens
    special_tokens_dict = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    model = GPT2LMHeadModel.from_pretrained(model_path)
    # 调整模型嵌入层大小以匹配新的分词器
    model.resize_token_embeddings(len(tokenizer))
    # 确保模型配置知道新的pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"✅ 模型和分词器初始化完成。新增了 {num_added_toks} 个特殊 Token。")
    return model, tokenizer

# --- 5. 核心训练函数 ---
def fine_tune_gpt2(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, 
                   train_file: str, val_file: str,
                   output_dir: str, num_epochs: int):
    """
    使用准备好的数据集对GPT-2模型进行微调。
    """
    # 加载数据集
    train_dataset = FlowDataset(tokenizer=tokenizer, file_path=train_file)
    val_dataset = FlowDataset(tokenizer=tokenizer, file_path=val_file)

    # 数据整理器，用于动态填充和创建labels
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 动态计算评估步数：每轮评估2次
    steps_per_epoch = len(train_dataset) // (8 * 1) # 假设BS=8, GradAccum=1
    eval_steps = max(10, steps_per_epoch // 2) 

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        
        # 同步日志、评估和保存策略
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        
        # 设置步数
        logging_steps=eval_steps,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        
        save_total_limit=2, # 只保留最新的2个检查点
        load_best_model_at_end=True, # 训练结束时加载最佳模型
        metric_for_best_model="eval_loss",
        
        learning_rate=5e-5,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        
        fp16=torch.cuda.is_available(),
        
        # 优化项
        gradient_checkpointing=True, # 节省显存
        dataloader_num_workers=2 if platform.system() != 'Windows' else 0, # Windows下设为0
        dataloader_pin_memory=True,

        report_to="none", # 禁用wandb等报告
        disable_tqdm=True, # 使用我们自定义的回调进度条
        log_level="error",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[TrainingProgressCallback()],
    )

    print("开始训练...")
    trainer.train()
    
    # 训练结束后，保存最终的最佳模型
    final_model_dir = os.path.join(output_dir, "best-model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"✅ 训练完成，最佳模型已保存至: {final_model_dir}")

    return trainer

# --- 6. 主执行函数 ---
def main():
    print(f"\n{'='*80}")
    print(" " * 25 + "GPT-2 网络流量微调程序 (优化版)")
    print(f"{'='*80}\n")
    
    # --- 文件路径配置 ---
    # 支持通过命令行参数指定数据集名称
    import argparse
    parser = argparse.ArgumentParser(description='GPT-2 网络流量微调程序')
    parser.add_argument('--dataset', type=str, default='NF-UNSW-NB15-v3',
                       help='数据集名称（例如: NF-UNSW-NB15-v3），默认: NF-UNSW-NB15-v3')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    train_file = f'./processed_data/{dataset_name}/train_gpt2_input.txt'
    val_file = f'./processed_data/{dataset_name}/val_gpt2_input.txt'
    local_model_path = './models/base/gpt2'
    output_dir = f"./models/gpt2-finetuned-traffic-{dataset_name}"

    # --- 路径检查 ---
    paths_to_check = {
        "训练文件": train_file,
        "验证文件": val_file,
        "本地预训练模型": local_model_path,
    }
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print(f"❌ 错误: {name}未找到! 路径: {path}")
            print("请确保已下载预训练模型并运行了数据预处理脚本。")
            return
            
    try:
        model, tokenizer = initialize_model_and_tokenizer(local_model_path)
        
        fine_tune_gpt2(
            model=model,
            tokenizer=tokenizer,
            train_file=train_file,
            val_file=val_file,
            output_dir=output_dir,
            num_epochs=5,
        )
        
        print("\n🎉 微调任务成功完成!")
        print(f"\n下一步:\n1. 查看 '{output_dir}' 目录下的训练曲线图和日志。")
        print(f"2. 使用 '{os.path.join(output_dir, 'best-model')}' 路径下的模型进行流量生成。")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被用户中断。")
    except Exception as e:
        print(f"\n❌ 训练过程中发生严重错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
