# 车牌识别：小型端到端网络（CCPD2019）

单张图片输入，一次前向同时输出 **车牌位置（4 个角点）** 与 **车牌文本**。
模型 2.97 M 参数，输入 320×512，RTX 5060 Laptop（8 GB）上 batch=32 约 140 ms/step。

```
图片 (3,320,512)
   │
   ├─ TinyBackbone (ResNet 风格, stride 2/4/8/16)   ← 1.2 M 参数
   │     c3(/8,96ch)  +  c4(/16,160ch) ──1x1 上采样融合──> 128ch @ stride 8
   │
   ├─ DetectionHead (3x3 conv ×2 → 1x1)
   │     obj (1,40,64) + 角点偏移 (8,40,64)
   │     argmax 取最佳格子 → 4 角点（可微）
   │
   ├─ 透视变换 warp_quads (torch grid_sample, 单应矩阵解析求解) → 3×32×96 校正图
   │
   └─ RecognitionHead (小型 CRNN: CNN + BiLSTM) → (T=24, C=66) → CTC
```

检测与识别共享同一次前向，透视变换可微 —— 识别损失会把梯度回传给检测头，
因此端到端联合训练（不需要 CPU 裁剪，也不需要两套模型）。


## 预训练权重

checkpoint 下载：[Google Drive](https://drive.google.com/file/d/1Y4A1yz6jKJmH5kXJx9Y8fOOVsxqNjelj/view?usp=sharing)

## 目录结构

```
plate_detection/
├── lpr/
│   ├── charset.py      # 65 类字符表（31 省 + 24 字母 + 10 数字），CTC 贪心解码
│   ├── ccpd.py         # 文件名标注解析、letterbox、几何/光照增强、CTC collate
│   ├── geometry.py     # 角点排序、点在四边形内、GIoU、批量单应矩阵、可微透视变换
│   ├── model.py        # TinyBackbone / DetectionHead / RecognitionHead / PlateNet
│   ├── loss.py         # Focal(obj) + L1(角点) + GIoU + CTC(预测裁剪) + CTC(GT裁剪)
│   ├── evaluate.py     # IoU / 检出率 / 端到端准确率 / CER
│   └── paths.py        # 路径解析：相对路径先按 cwd 找，找不到再回落到本目录
├── configs/default.yaml
├── train.py            # 训练入口（AMP、cosine+warmup、验证、断点续训）
├── evaluate.py         # 在 val/test 上评测
├── predict.py          # 单图/文件夹推理，画框并写文本
├── tools/
│   ├── check_dataset.py   # 数据集完整性校验
│   ├── check_pipeline.py  # 可视化标注 + 校正结果（肉眼验收数据管线）
│   ├── bench.py           # 吞吐 / 显存基准
│   ├── profile_step.py    # 单步耗时拆解（取数 / 前向 / 反向）
│   └── diagnose_det.py    # 检测分支诊断
└── data/CCPD2019/...      # 需自行放置（仓库里不含数据集）
```


## 环境

RTX 50 系（Blackwell, sm_120）必须用 cu128 及以上构建的 torch，cu126 会报
`CUDA error: no kernel image is available for execution on the device`。

```bash
# 在仓库根目录（algo/）执行
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy opencv-python-headless pillow pyyaml tqdm
```

macOS / 无 NVIDIA 显卡：直接 `pip install torch torchvision`（默认 CPU 版）即可，
`predict.py` / `evaluate.py` 会自动回落到 CPU（不启用 MPS），功能一致、速度慢些。
也可以直接用 `pip install -r plate_detection/requirements.txt`。

## 复现

```bash
# 0) 数据审计（11 s）
python tools/audit_dataset.py

# 1) 重建 split（含省份均衡诊断集）
python tools/make_splits.py --per-province 150 --val-base 4000

# 2) 重训（8 epoch × 215 179，约 2.3 h / RTX 5060 Laptop 8 GB）
python train.py --config configs/balanced.yaml

# 3) 消融：同数据池、同步数，只把 α 改成 0（自然先验）
python train.py --config configs/ablation_alpha0.yaml --max-steps 12000

# 4) 评测（多 split + 逐省 + 事后先验校正扫描）
python tools/eval_suite.py --weights runs/plate_bal/checkpoints/best.pth  --name new_bal --tau-sweep 0,0.5,1.0
python tools/eval_suite.py --weights runs/smoke_fix/checkpoints/best.pth  --name old_smoke --tau-sweep 0,1.0
python tools/pos_analysis.py --weights runs/plate_bal/checkpoints/best.pth --split test_prov --name new_bal
python tools/pos_analysis.py --weights runs/smoke_fix/checkpoints/best.pth --split test_prov --name old_smoke

# 5) 出图
python tools/make_figures.py --new-eval runs/eval/new_bal.json
python tools/make_ablation.py
python tools/make_qualitative.py --old runs/smoke_fix/checkpoints/best.pth \
                                 --new runs/plate_bal/checkpoints/best.pth \
                                 --out runs/report_figs/fig8_qualitative.png
```
