import kagglehub
import os

# 下载数据集：athena21/netflow-v3-datasets
print("开始下载数据集...")
path = kagglehub.dataset_download("athena21/netflow-v3-datasets")

print(f"\n下载完成！文件保存在: {path}")

# 列出下载的文件
print("文件夹内容:")
print(os.listdir(path))