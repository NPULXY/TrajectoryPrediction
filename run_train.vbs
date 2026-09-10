' 无窗口运行训练脚本
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c cd /d D:\Office\2026.1\意图识别论文\TrajectoryPrediction && C:\Users\Hasee\anaconda3\envs\torch\python.exe train.py > output/stdout.log 2>&1", 0, False
