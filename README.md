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

## 数据集核查结论（先做的一步）

`tools/check_dataset.py` 对全部 355,013 个文件做了逐项校验，结论是 **数据完整，没有缺失**：

| 项目 | 结果 |
|---|---|
| splits 中列出的路径总数 | 341,978（train 100,000 / val 99,996 / test 141,982） |
| 磁盘上不存在的条目 | **0** |
| 截断 / 无法解码的图片 | **0**（抽样 800 张完整解码，全部 720×1160） |
| 重复路径 | 0 |
| ccpd_base 引用覆盖率 | 199,996 / 199,996（100%） |

以下是三个需要知道的“非缺失”事实，不影响训练：

1. `ccpd_base/` 多套了一层（实际在 `ccpd_base/ccpd_base/*.jpg`），dataloader 已自动兼容。
2. `ccpd_np/` 有 **15 个 4 KB 的 macOS AppleDouble 垃圾文件**（magic `00 05 16 07`，开头是
   `Mac OS X`），它们不是图片。该目录本身没有标注（文件名是纯数字），也不在任何 split 里。
3. `ccpd_weather/`（9,999 张，带完整标注）和 `ccpd_np/`（3,036 张，无标注）**不在官方 split 文件里**。
   weather 可以当额外训练数据用，np 只能当负样本。

复现（命令均在**仓库根目录**执行，`algo/` 下）：

```bash
python plate_detection/tools/check_dataset.py --sample 2000
```

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

## 关键设计选择

**输入分辨率 320×512。** CCPD 图统一 720×1160（宽高比 1.611），512/320 = 1.60，
letterbox 几乎无拉伸；车牌的边长中位数 245×91 px，缩放后仍有 ~109×40 px，
小目标（db/fn 子集）也有 ~25 px 高。再大分辨率收益有限，代价是平方增长的计算量。

**检测用 stride 8 的 anchor-free 四角回归。** 不用 anchor / NMS：CCPD 每图一块车牌，
格子中心落在四边形内即为正样本（每图 20–50 个），每个正样本回归 4 个角点相对格心的偏移。
直接回归四边形而不是矩形框，tilt/rotate 子集（横滚角可到 30°+）也能贴合。

**角点初值给了“车牌形状”先验。** 检测头 bias 初始化为以格心为中心的 ±0.105W × ±0.063H，
否则初始四角重合 → 单应矩阵奇异 → 裁剪出 NaN，CTC 分支从第一步就崩。
这个坑实际踩到过，`PlateNet.__init__` 里有注释。

**检测与识别共用一份角点计算。** `PlateNet.dense_corners()` 是唯一实现，loss 和 decode
都调它。早期版本 loss 里自己写了一遍 `permute+reshape`，布局解释错了（日志显示 corner loss
3.2，真实误差 221），网络朝着错误目标发散、IoU 卡在 0.02。这是本项目最值得记住的教训：
**凡是 loss 与推理共享的几何量，只能有一份实现。**

**识别用 GT 裁剪 + 预测裁剪双路 CTC。** `ctc_gt`（用真值角点裁剪）保证识别分支早期就有干净梯度，
`ctc`（用预测角点裁剪）负责把检测和识别对齐。二者权重 0.5 : 1.0。

**字母数字表是 34 类而不是 36 类。** 中国车牌不用 `I`、`O`，CCPD 的编码是
`ABCDEFGHJKLMNPQRSTUVWXYZ` + `0123456789`（字母在前、数字在后），加上 31 个省份共 65 类，
再留 1 个 CTC blank → 66 路 logits。用错顺序会让标签整体错位
（`皖APV930` 被解成 `皖56UUV8`）；本项目的映射已用图上的真实车牌逐字核对过。

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

## 用法

**以下命令都在仓库根目录（`algo/`）执行**，脚本一律按 `plate_detection/xxx.py` 调用。

`--config / --weights / --source / --out / --resume` 的相对路径会**先按当前目录解析，
找不到再回落到 `plate_detection/`**，所以 `runs/smoke_fix/checkpoints/best.pth`（项目内短路径）
和 `plate_detection/runs/smoke_fix/checkpoints/best.pth` 都能跑通，绝对路径不受影响。

数据集需自行放在 `plate_detection/data/CCPD2019`（`splits/{train,val,test}.txt` + 各子目录图像），
仓库里不含数据集；缺数据时各脚本会打印 `[FATAL] ... not found` 并返回退出码 2（不再抛 traceback）。

```bash
# 1) 数据自查（会扫全部文件，约 15 分钟）
python plate_detection/tools/check_dataset.py --sample 2000

# 2) 肉眼验收数据管线：画 GT 四边形 + 校正后的车牌 + 文本
python plate_detection/tools/check_pipeline.py --n 8 --split val

# 3) 训练（默认 12 epoch × 3125 step，8 GB 显存够用）
python plate_detection/train.py --config configs/default.yaml

# 冒烟：800 步，几分钟（产物写到 plate_detection/runs/smoke）
python plate_detection/train.py --max-steps 800 --batch-size 32 --train-limit 8000 \
                --lr 1.5e-3 --warmup-steps 80 --val-limit 512 \
                --val-every 100 --val-batches 16 --output-dir runs/smoke

# 断点续训（--output-dir 必须指回原目录，否则新日志/ckpt 会写进 configs 里的 runs/plate_net）
python plate_detection/train.py --resume runs/smoke_fix/checkpoints/last.pth \
                --output-dir runs/smoke_fix

# 4) 评测
python plate_detection/evaluate.py --weights runs/smoke_fix/checkpoints/best.pth --split val --limit 5000

# 5) 推理（单图或文件夹），结果画框存到 plate_detection/runs/pred
python plate_detection/predict.py --weights runs/smoke_fix/checkpoints/best.pth --source test.jpg
python plate_detection/predict.py --weights runs/smoke_fix/checkpoints/best.pth \
                                  --source some_dir/ --out runs/pred
```

推理输出会在车牌四边形上方画绿底白字标签。车牌文本以中文省份开头，必须用 TrueType 中文字体渲染
（OpenCV 自带的 Hershey 字体只有 ASCII），程序会按 Windows → macOS → Linux 顺序自动找系统字体，
全都找不到时退回 OpenCV 字体并打印告警（此时省份汉字会缺失），也可以显式指定：

```bash
python plate_detection/predict.py --weights runs/smoke_fix/checkpoints/best.pth \
    --source test.jpg --font "/System/Library/Fonts/Hiragino Sans GB.ttc"
```

## 实测结果（800 步冒烟，仅见 25.6 k 样本 ≈ 0.26 epoch）

`runs/smoke_fix`，val 512 张，IoU>0.7 计检出：

| step | loss | IoU | 检出率 | 端到端准确率 | CER |
|---|---|---|---|---|---|
| 100 | 7.78 | 0.339 | 0.6% | 0.0% | 0.863 |
| 200 | — | 0.825 | 92.0% | 0.0% | 0.732 |
| 400 | — | 0.759 | 74.8% | 0.4% | 0.396 |
| 600 | 2.57 | 0.822 | 94.5% | 30.3% | 0.148 |
| 800 | 1.92 | 0.863 | 98.1% | **57.0%** | **0.078** |

趋势明确：检测先收敛（边框任务简单），识别随后跟上。完整 12 epoch 有望把端到端准确率推到
0.85+（RPnet 论文在 CCPD 上报 99.3% 检测 / 94.7% 识别，但其骨干与输入分辨率都比这里大一个量级）。

推理样例见 `runs/demo_out/`（含 tilt/rotate/fn 子集样本），`runs/sanity/` 可看数据管线验收图。

## 显存与速度（RTX 5060 Laptop 8 GB，实测）

| batch | ms/step | img/s | 峰值显存（AMP 训练） |
|---|---|---|---|
| 16 | 73 | 218 | 0.68 GB |
| 32 | 140 | 229 | 1.27 GB |
| 64 | 258 | 248 | 2.45 GB |

结论：

- **训练显存 ≈ 0.6 GB + 0.029 GB × batch**（bf16 混合精度，含参数/梯度/AdamW 两份状态）。
  batch=32 约 1.3 GB；batch=64 约 2.5 GB。
- 加上验证阶段（fp32、无 AMP）与 cuDNN autotune 的临时占用，进程峰值约 3 GB。
  **8 GB 卡可以放心开 batch 64–96；4 GB 卡用 batch 16–32。**
- 端到端训练 100 k 图 1 epoch ≈ 7.5 分钟（229 img/s），12 epoch ≈ 1.5 小时（单卡）。
- 首次前向会花 ~18 s 做 cuDNN autotune / 内核编译，只在进程启动时发生一次。

## 已知限制

- 只支持单块车牌：没有 NMS，argmax 只取一个格子。多车牌场景需要改成 top-k + NMS。
- 训练集只有 `splits/train.txt`（ccpd_base）。想提升 blur/fn/weather 的鲁棒性，可以把
  `ccpd_weather` 或 test 子集并进训练（改 `build_loader` 的 split 参数即可）。
- 不含车牌颜色/类型（蓝牌/绿牌/黄牌）分类。
- CTC 贪心解码，没有语言模型或约束解码；7/8 位车牌的位次约束可以进一步降 CER。

## 参考

- Xu et al., *Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline*, ECCV 2018（CCPD + RPnet，本项目的主要参照：共享特征、角点检测 + 透视校正 + 单次前向）
  <https://github.com/detectRecog/CCPD>
- Shi et al., *An End-to-End Trainable Neural Network for Image-based Sequence Recognition*（CRNN + CTC）
- Zherzdev & Gruzdev, *LPRNet: License Plate Recognition via Deep Neural Networks*（轻量识别头设计思路）
- Tian et al., *FCOS: Fully Convolutional One-Stage Object Detection*（本文的格子内正样本分配方式）
