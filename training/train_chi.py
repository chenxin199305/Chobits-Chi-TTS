"""小叽 TTS 训练流水线驱动脚本 (GPT-SoVITS v2Pro, 命令行版, 复刻 webui.py 的调用方式).

步骤:
  1. 1-get-text.py          文本 -> 音素 (ja 不需要 BERT 特征, 但脚本框架一致)
  2. 2-get-hubert-wav32k.py HuBERT SSL 特征 + wav 重采样到 32k
  3. 2-get-sv.py            说话人嵌入 (v2Pro 需要)
  4. 3-get-semantic.py      语义 token (VQ)
  5. s2_train.py            SoVITS 声学模型微调 (全量, v2Pro)
  6. s1_train.py            GPT 语义模型微调

用法 (在 GPT-SoVITS 目录下运行, 或任意目录, 脚本会自行定位):
  conda activate GPTSoVits
  python training/train_chi.py [--skip-preprocess] [--skip-s2] [--skip-s1]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Chobits-Chi-TTS/
GS_ROOT = os.path.join(REPO_ROOT, "GPT-SoVITS")

EXP_NAME = "chi"
VERSION = "v2Pro"
LIST_PATH = os.path.join(REPO_ROOT, "data", "gpt_sovits.list")
WAV_DIR = os.path.join(REPO_ROOT, "data", "wavs")
OPT_DIR = os.path.join(GS_ROOT, "logs", EXP_NAME)
TMP_DIR = os.path.join(GS_ROOT, "TEMP")

PRETRAINED = {
    "bert": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "hubert": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    "sv": "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
    "s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
    "s2D": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth",
    "s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "s2config": "GPT_SoVITS/configs/s2v2Pro.json",
}

# 8GB 显存 (4060 Laptop, 桌面约占 1.6G) + 483 条/21.4min 小数据集
S2_BATCH_SIZE = 4
S2_EPOCHS = 15
S2_SAVE_EVERY = 5
S1_BATCH_SIZE = 4
S1_EPOCHS = 15
S1_SAVE_EVERY = 5

BASE_ENV = {
    **os.environ,
    # webui.py 靠 users.pth 注入这些路径, 命令行直接用 PYTHONPATH 显式指定
    "PYTHONPATH": os.pathsep.join(
        [GS_ROOT, os.path.join(GS_ROOT, "GPT_SoVITS"), os.path.join(GS_ROOT, "GPT_SoVITS", "BigVGAN")]
        + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    ),
    # torchcodec 的 libtorchcodec_core*.so 依赖 pip 包 nvidia-npp-cu12 提供的
    # libnppicc.so.12, 但该目录不在默认搜索路径, 需显式加入
    "LD_LIBRARY_PATH": os.pathsep.join(
        [
            os.path.join(
                os.path.dirname(sys.executable),
                "..",
                "lib",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                "site-packages",
                "nvidia",
                "npp",
                "lib",
            )
        ]
        + ([os.environ["LD_LIBRARY_PATH"]] if os.environ.get("LD_LIBRARY_PATH") else [])
    ),
    "version": VERSION,
    "is_half": "True",
    "i_part": "0",
    "all_parts": "1",
    "_CUDA_VISIBLE_DEVICES": "0",
}


def run(script: str, extra_env: dict, desc: str) -> None:
    env = {**BASE_ENV, **extra_env}
    print(f"\n{'=' * 60}\n[{desc}]\n{'=' * 60}", flush=True)
    p = subprocess.run(
        [sys.executable, "-s", script],
        cwd=GS_ROOT,
        env=env,
    )
    if p.returncode != 0:
        raise SystemExit(f"[失败] {desc} 退出码 {p.returncode}")


def preprocess() -> None:
    common = {
        "inp_text": LIST_PATH,
        "inp_wav_dir": WAV_DIR,
        "exp_name": EXP_NAME,
        "opt_dir": OPT_DIR,
    }
    name2text = os.path.join(OPT_DIR, "2-name2text.txt")
    if os.path.exists(name2text):
        print("[跳过] 1/6 文本 -> 音素 (已存在)")
    else:
        run(
            "GPT_SoVITS/prepare_datasets/1-get-text.py",
            {**common, "bert_pretrained_dir": PRETRAINED["bert"]},
            "1/6 文本 -> 音素",
        )
        # 单进程结果合并 (webui 中多 GPU 分片后的合并逻辑)
        shutil.move(os.path.join(OPT_DIR, "2-name2text-0.txt"), name2text)

    hubert_dir = os.path.join(OPT_DIR, "4-cnhubert")
    if os.path.isdir(hubert_dir) and len(os.listdir(hubert_dir)) > 0:
        print("[跳过] 2/6 HuBERT SSL 特征 (已存在)")
    else:
        run(
            "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
            {**common, "cnhubert_base_dir": PRETRAINED["hubert"]},
            "2/6 HuBERT SSL 特征",
        )

    sv_dir = os.path.join(OPT_DIR, "7-sv_cn")
    if os.path.isdir(sv_dir) and len(os.listdir(sv_dir)) > 0:
        print("[跳过] 3/6 说话人嵌入 (已存在)")
    else:
        run(
            "GPT_SoVITS/prepare_datasets/2-get-sv.py",
            {**common, "cnhubert_base_dir": PRETRAINED["hubert"], "sv_path": PRETRAINED["sv"]},
            "3/6 说话人嵌入 (v2Pro)",
        )

    semantic_tsv = os.path.join(OPT_DIR, "6-name2semantic.tsv")
    if os.path.exists(semantic_tsv):
        print("[跳过] 4/6 语义 token (已存在)")
    else:
        run(
            "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
            {
                "inp_text": LIST_PATH,
                "exp_name": EXP_NAME,
                "opt_dir": OPT_DIR,
                "pretrained_s2G": PRETRAINED["s2G"],
                "s2config_path": PRETRAINED["s2config"],
            },
            "4/6 语义 token",
        )
        with open(os.path.join(OPT_DIR, "6-name2semantic-0.tsv"), encoding="utf-8") as f:
            body = f.read().strip("\n")
        with open(semantic_tsv, "w", encoding="utf-8") as f:
            f.write("item_name\tsemantic_audio\n" + body + "\n")
        os.remove(os.path.join(OPT_DIR, "6-name2semantic-0.tsv"))


def train_s2() -> None:
    with open(os.path.join(GS_ROOT, PRETRAINED["s2config"]), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["train"].update(
        {
            "batch_size": S2_BATCH_SIZE,
            "epochs": S2_EPOCHS,
            "text_low_lr_rate": 0.4,
            "pretrained_s2G": PRETRAINED["s2G"],
            "pretrained_s2D": PRETRAINED["s2D"],
            "if_save_latest": True,
            "if_save_every_weights": True,
            "save_every_epoch": S2_SAVE_EVERY,
            "gpu_numbers": "0",
            "grad_ckpt": False,
            "lora_rank": "32",
        }
    )
    cfg["model"]["version"] = VERSION
    cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = OPT_DIR
    cfg["save_weight_dir"] = "SoVITS_weights_v2Pro"
    cfg["name"] = EXP_NAME
    cfg["version"] = VERSION
    os.makedirs(os.path.join(OPT_DIR, "logs_s2_v2Pro"), exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    tmp_cfg = os.path.join(TMP_DIR, "tmp_s2.json")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    print(f"\n{'=' * 60}\n[5/6 SoVITS 训练]\n{'=' * 60}", flush=True)
    p = subprocess.run(
        [sys.executable, "-s", "GPT_SoVITS/s2_train.py", "--config", tmp_cfg],
        cwd=GS_ROOT,
        env=BASE_ENV,
    )
    if p.returncode != 0:
        raise SystemExit(f"[失败] SoVITS 训练退出码 {p.returncode}")


def train_s1() -> None:
    with open(os.path.join(GS_ROOT, "GPT_SoVITS/configs/s1longer-v2.yaml"), encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["train"].update(
        {
            "batch_size": S1_BATCH_SIZE,
            "epochs": S1_EPOCHS,
            "save_every_n_epoch": S1_SAVE_EVERY,
            "if_save_every_weights": True,
            "if_save_latest": True,
            "if_dpo": False,
            "half_weights_save_dir": "GPT_weights_v2Pro",
            "exp_name": EXP_NAME,
        }
    )
    cfg["pretrained_s1"] = PRETRAINED["s1"]
    cfg["train_semantic_path"] = os.path.join(OPT_DIR, "6-name2semantic.tsv")
    cfg["train_phoneme_path"] = os.path.join(OPT_DIR, "2-name2text.txt")
    cfg["output_dir"] = os.path.join(OPT_DIR, "logs_s1_v2Pro")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    tmp_cfg = os.path.join(TMP_DIR, "tmp_s1.yaml")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    env = {**BASE_ENV, "hz": "25hz"}
    print(f"\n{'=' * 60}\n[6/6 GPT 训练]\n{'=' * 60}", flush=True)
    p = subprocess.run(
        [sys.executable, "-s", "GPT_SoVITS/s1_train.py", "--config_file", tmp_cfg],
        cwd=GS_ROOT,
        env=env,
    )
    if p.returncode != 0:
        raise SystemExit(f"[失败] GPT 训练退出码 {p.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-s2", action="store_true")
    ap.add_argument("--skip-s1", action="store_true")
    args = ap.parse_args()

    for k, v in PRETRAINED.items():
        p = os.path.join(GS_ROOT, v)
        if not os.path.exists(p):
            raise SystemExit(f"预训练文件缺失: {p}")

    if not args.skip_preprocess:
        preprocess()
    if not args.skip_s2:
        train_s2()
    if not args.skip_s1:
        train_s1()
    print("\n[完成] 权重输出: GPT-SoVITS/SoVITS_weights_v2Pro/ 与 GPT-SoVITS/GPT_weights_v2Pro/")


if __name__ == "__main__":
    main()
