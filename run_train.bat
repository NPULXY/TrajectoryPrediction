@echo off
cd /d D:\Office\2026.1\意图识别论文\TrajectoryPrediction
C:\Users\Hasee\anaconda3\envs\torch\python.exe train.py > output/stdout.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> output/stdout.log
