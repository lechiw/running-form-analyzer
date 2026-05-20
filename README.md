# 🏃 Running Form Analyzer

AI 跑姿分析工具——上传跑步视频，自动提取骨架、计算生物力学指标、生成分析报告。

## 功能

- **姿态提取**：基于 MediaPipe Pose，从视频中提取 33 个身体关键点
- **跑姿指标**：步频、躯干前倾角、手臂对称性、垂直振幅、触地距离、着地类型、膝关节角度
- **自动评分**：0-100 综合跑姿评分，每项指标独立打分
- **可视化**：骨架叠加 + 实时数据面板
- **分析报告**：中文报告，指出优势和改进建议（支持模板版 / LLM 版）

## 快速开始

```bash
pip install -r requirements.txt

# 测试（合成数据验证）
python3 test.py

# 分析跑步视频
python3 run.py /path/to/your/running_video.mp4

# 分析并生成带骨架的可视化视频
python3 run.py /path/to/your/running_video.mp4 --render
```

## 拍摄建议

为获得准确分析，请用手机**横屏、侧面**拍摄跑步视频：

```
     手机（三脚架固定）
          📱
           ↓
   🏃 →  跑  → 🏃
      镜头侧面
```

- 跑步机最佳：手机架在侧面，拍 30 秒
- 穿浅色紧身衣
- 不要正对/背对镜头

## 项目结构

```
running-form-analyzer/
├── run.py              # CLI 入口
├── pose_extractor.py   # MediaPipe 骨架提取
├── metrics.py          # 跑姿指标 + 评分
├── visualizer.py       # 骨架可视化叠加
├── analyzer.py         # AI 报告生成
├── test.py             # 合成数据测试
├── requirements.txt
└── output/             # 分析结果输出
```

## 技术栈

- MediaPipe Pose（姿态估计）
- OpenCV（视频处理）
- NumPy（数值计算）

## License

MIT
