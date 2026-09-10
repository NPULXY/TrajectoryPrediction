"""
训练启动脚本 —— 封装 train.py，重定向所有输出到文件，避免终端缓冲问题。
用法: python launch_train.py
"""
import sys
import os
import time

# 切换到项目目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 将 stdout 和 stderr 重定向到文件，同时仍打印到终端
log_file = open("output/stdout.log", "w", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

print(f"训练启动: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python: {sys.version}")
print(f"工作目录: {os.getcwd()}")
sys.stdout.flush()

# 导入并运行训练
import train
train.train()

print(f"训练完成: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log_file.close()
