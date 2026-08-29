# Chobits-Chi-TTS

[![Made with Love](https://img.shields.io/badge/Made%20with-Love-ff69b4.svg)](https://madewithlove.org.in)
[![GitHub](https://img.shields.io/badge/GitHub-Chobits--Chi--TTS-181717?logo=github)](https://github.com/chenxin199305/Chobits-Chi-TTS)
[![Dataset: Chobits-Chi-Voice](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Chobits--Chi--Voice-yellow)](https://huggingface.co/datasets/chenxin199305/Chobits-Chi-Voice)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Chobits--Chi--TTS-yellow)](https://huggingface.co/chenxin199305/Chobits-Chi-TTS)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Language: Japanese](https://img.shields.io/badge/Language-Japanese-green.svg)]()

> ✅ 首轮模型已训练完成并人工试听通过（GPT-SoVITS v2Pro，e10 checkpoint），权重已发布到 [Hugging Face](https://huggingface.co/chenxin199305/Chobits-Chi-TTS)，见[模型文件](#模型文件)。

《人形电脑天使心》(Chobits) 中 **小叽 (Chi / ちぃ)** 角色的 TTS（语音合成）模型项目。

本项目以 [Chobits-Chi-Voice](https://github.com/chenxin199305/Chobits-Chi-Voice) 数据集为训练数据，基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 微调出小叽声线的文本转语音模型。

> ⚠️ 注意：原始动画音频的版权归其权利方所有。本项目仅供学习与研究使用，请勿用于商业用途。

## 数据

训练数据来自 [Chobits-Chi-Voice](https://github.com/chenxin199305/Chobits-Chi-Voice) 数据集（[Hugging Face](https://huggingface.co/datasets/chenxin199305/Chobits-Chi-Voice)）：

| 项目 | 数值 |
| --- | --- |
| 片段数量 | 487 |
| 总时长 | 约 21.4 分钟 |
| 片段时长 | 中位 2.24s，平均 2.63s（0.98s – 9.05s） |
| 采样率 | 22050 Hz，单声道，16-bit PCM |
| 语言 | 日语 |
| 标注格式 | `metadata.csv`：`文件名|文本` |

数据集的 `metadata.csv` 标注格式与 GPT-SoVITS 训练输入格式兼容，可直接用于微调。

## 训练路线

基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的少样本语音克隆流程（v2Pro，2026-08 已跑通首轮）：

1. ~~数据准备~~：✅ `tools/clean_dataset.py` 清洗 Chobits-Chi-Voice 数据集（修复 Whisper 误转写 18 条、剔除 4 条），输出 `data/gpt_sovits.list`（483 条 / 21.4 分钟）
2. ~~预处理~~：✅ 文本→音素、HuBERT SSL 特征、说话人嵌入（v2Pro）、语义 token（VQ），产物在 `GPT-SoVITS/logs/chi/`
3. ~~微调 SoVITS~~：✅ v2Pro 全量微调，batch 4 × 15 epochs → `GPT-SoVITS/SoVITS_weights_v2Pro/chi_e{5,10,15}_*.pth`
4. ~~微调 GPT~~：✅ batch 4 × 15 epochs → `GPT-SoVITS/GPT_weights_v2Pro/chi-e{5,10,15}.ckpt`
5. **推理评估**：✅ 首轮合成已产出 `outputs/inference/e{5,10,15}/output.wav`（32kHz）。人工试听选定 **e10**（`chi-e10.ckpt` + `chi_e10_s1210.pth`）为当前首选
6. **迭代优化**：根据试听结果调整 epochs / 参考音频 / 数据量，必要时换 v4 对比

复现命令见下方[快速开始](#快速开始从零复现)。

## 模型文件

当前首选权重（人工试听选定，v2Pro，e10）已发布到 Hugging Face：
**[huggingface.co/chenxin199305/Chobits-Chi-TTS](https://huggingface.co/chenxin199305/Chobits-Chi-TTS)**

```bash
# 下载模型 (约 290MB)
pip install huggingface_hub
hf download chenxin199305/Chobits-Chi-TTS --local-dir models/
```

| 文件 | 说明 |
| --- | --- |
| `chi-e10.ckpt` | GPT 语义模型（149MB） |
| `chi_e10_s1210.pth` | SoVITS 声学模型（129MB） |
| `ref_audio.wav` | 参考音频（ep07「秀樹は地位を拾ってくれた」，4.2s） |
| `ref_text.txt` | 参考音频对应文本（同 `examples/ref_text.txt`） |

下载后可直接推理（见快速开始第 5 步，将权重路径替换为 `models/` 内文件），无需训练。
本地训练产生的全部 checkpoint 在 `GPT-SoVITS/GPT_weights_v2Pro/` 与 `GPT-SoVITS/SoVITS_weights_v2Pro/`（e5/e10/e15），试听样本在 `outputs/inference/`。

## 仓库结构

```
Chobits-Chi-TTS/
├── README.md               # 本文件
├── LICENSE                 # CC BY-NC-SA 4.0
├── .gitignore
├── data/                   # 清洗后的训练数据 (由 tools/clean_dataset.py 生成)
│   ├── wavs/                  # 483 个音频片段 (不入库, 由脚本从 Chobits-Chi-Voice 复制)
│   ├── metadata.csv           # 文件名|文本
│   ├── gpt_sovits.list        # GPT-SoVITS 训练格式 (含本机绝对路径, 不入库)
│   └── cleaning_report.csv    # 清洗记录 (修复/剔除明细, 供人工复核)
├── tools/                  # 数据整理脚本
│   └── clean_dataset.py       # 数据集清洗 (修复 Whisper 误转写, 剔除脏条目)
├── training/               # 训练流水线
│   └── train_chi.py           # 预处理 + SoVITS/GPT 微调一键驱动 (幂等, 可续跑)
├── examples/               # 示例
│   ├── ref_text.txt           # 参考音频文本 (ep07_00783.96s)
│   └── target_text.txt        # 推理示例文本
├── models/                 # 导出的模型权重与参考音频 (不入库, 见"模型文件")
├── GPT-SoVITS/             # GPT-SoVITS 框架 (git clone, 不入库)
└── outputs/                # 试听音频等产物 (不入库)
```

## 快速开始（从零复现）

以下步骤在 Ubuntu 24.04 + RTX 4060 Laptop 8GB 上实测通过（2026-08）。国内网络已按镜像优化。

### 1. 获取训练数据

```bash
# 从 Hugging Face 下载 Chobits-Chi-Voice 数据集 (含 wavs/, metadata.csv, transcripts.csv)
git clone https://huggingface.co/datasets/chenxin199305/Chobits-Chi-Voice
```

### 2. 搭建环境

```bash
# conda 环境 (官方测试配置: Python 3.10 + PyTorch CUDA 12.8)
conda create -n GPTSoVits -c conda-forge --override-channels python=3.10 pip -y
conda activate GPTSoVits

# 克隆框架 (锁定到本项目验证过的 commit, 避免 main 分支漂移)
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS && git checkout 48b1a0169a28582a8984402f82cf438d3bfa6aca

# 一键安装 (依赖 + 预训练模型, 模型走 ModelScope 源; 非 TTY 环境需 TERM=xterm)
TERM=xterm bash install.sh --device CU128 --source ModelScope
```

install.sh 完成后，**按顺序**执行以下修复（2026-08 时点的上游版本兼容问题）：

```bash
# a) fastapi 0.141/starlette 1.6 与 gradio 4.x 不兼容, 回退
pip install 'fastapi[standard]==0.115.6' 'starlette==0.41.3' 'uvicorn==0.32.0'

# b) install.sh 装的 PyPI 版 torchaudio 2.11 按 CUDA 13 编译, 与 torch cu128 不兼容, 换装 cu128 版
pip install --force-reinstall --no-deps torchaudio --index-url https://download.pytorch.org/whl/cu128

# c) torchcodec 依赖: npp 动态库 + ffmpeg<=8
pip install nvidia-npp-cu12
conda install -c conda-forge --override-channels 'ffmpeg=7' -y

# d) nltk 3.10 需要 punkt_tab (install.sh 下载的 nltk_data 不含)
python -m nltk.downloader punkt_tab
```

> PyPI 下载缓慢时换清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；
> torch cuXXX 系列可用上海交大镜像
> `pip install torch torchcodec --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128`
> （阿里云与清华均已下线 pytorch-wheels 镜像）。

### 3. 数据清洗

```bash
# 回到本仓库根目录, 生成 data/ (修复 Whisper 误转写 18 条, 剔除 4 条)
python3 tools/clean_dataset.py --src /path/to/Chobits-Chi-Voice/dataset
```

### 4. 训练

```bash
# 预处理 + SoVITS + GPT 一键完成 (幂等, 中断后可续跑)
python training/train_chi.py
```

训练产物：`GPT-SoVITS/SoVITS_weights_v2Pro/chi_e*.pth`（声学模型）与 `GPT-SoVITS/GPT_weights_v2Pro/chi-e*.ckpt`（语义模型），每 5 个 epoch 保存一次。

### 5. 推理

```bash
cd GPT-SoVITS
version=v2Pro LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/npp/lib \
python GPT_SoVITS/inference_cli.py \
  --gpt_model GPT_weights_v2Pro/chi-e10.ckpt \
  --sovits_model SoVITS_weights_v2Pro/chi_e10_s1210.pth \
  --ref_audio ../data/wavs/ep07_00783.96s.wav \
  --ref_text ../examples/ref_text.txt --ref_language 日文 \
  --target_text ../examples/target_text.txt --target_language 日文 \
  --output_path ../outputs/inference/run1
```

`LD_LIBRARY_PATH` 前缀是 torchcodec 加载 `libnppicc.so.12` 所需（见 2-c）。
也可以 `python webui.py` 启动浏览器界面操作（localhost:9874）。

> 权重已在 Hugging Face 发布，不想训练可直接下载（见[模型文件](#模型文件)）；
> 参考音频可用 `data/wavs/` 中任意一条 3–8 秒的片段替代
> （`ref_text.txt` 内容需与音频一致，参考 `data/metadata.csv`）。

## 许可协议

本项目的派生内容遵循数据集的 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)（署名-非商业性使用-相同方式共享）协议：

- **署名 (BY)**：使用时须注明来源。
- **非商业性使用 (NC)**：不得用于商业目的。
- **相同方式共享 (SA)**：衍生作品须以相同协议发布。

## 免责声明

- 本项目仅用于学术研究与个人学习，不构成对原作品版权的任何主张。
- 使用本项目训练的模型所生成的内容，不得用于侵犯原作品及相关声优（田中理惠）权益的用途。
- 若权利方提出要求，本项目将被下架。

## 相关项目

- [Chobits-Chi-Voice](https://github.com/chenxin199305/Chobits-Chi-Voice) — 小叽语音数据集（本项目的数据来源）
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 底层语音合成框架
