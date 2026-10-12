
# LeRobot + Sigmastar IPU SDK 全流程落地指南
本文档按照从零到一的操作顺序覆盖完整链路：

> 环境搭建 → 数据采集 → 模型训练 → IPU 模型转换 → 仿真验证

---

## 1. 项目简介

`leisaac_sgs` 是一套面向 Sigmastar IPU 板端部署的机器人学习全流程方案，打通以下环节：

- **仿真环境**：基于 Isaac Sim + Isaac Lab 的 LeIsaac 任务（SO-101 机械臂操作）
- **数据采集**：遥操作录制 + 公开数据集
- **模型训练**：ACT 在 LeIsaac 数据集上微调
- **IPU 推理**：ACT 模型编译为 `.img`，在 Comake Pi D3 板端推理
- **仿真验证**：policy server + 仿真环境联调评测

### 项目目录

```text
leisaac_sgs/
├── assets/                   # 3D 资产（机器人、场景）
│   ├── robots/
│   └── scenes/
├── lerobot_ipu/              # Sigmastar IPU 推理适配层
│   ├── __init__.py
│   ├── act.py
│   └── server.py
└── scripts/                  # 日常命令入口
    ├── convert/              # IsaacLab ↔ LeRobot 数据转换
    ├── environments/         # 环境自检、遥操作
    │   └── teleoperation/
    ├── evaluation/           # 评测与策略服务
    └── training/act/         # ACT 微调准备
```

---

## 2. 前置准备

### 硬件要求

| 设备 | 推荐配置 | 用途 |
|:---|:---|:---|
| 开发机 | Ubuntu 24.04 + NVIDIA 4090（驱动 580.167.08） | 仿真、数据采集、模型训练 |
| Comake Pi D3 | aarch64 Ubuntu，Sigmastar IPU | 板端推理部署 |

### 前提条件

- 已安装 `git`、`conda`（Miniconda 或 Anaconda）
- 开发机已安装 NVIDIA 驱动和 CUDA 工具链
- 板端已部署 Sigmastar SDK 及 `py_ipu` 运行环境

### 克隆仓库

```bash
git clone https://git.sigmastar.com.cn/sigmastar/leisaac_sgs.git
cd leisaac_sgs
```

---

## 3. 环境搭建

开发机使用一个 conda 环境即可同时覆盖仿真与 lerobot 数据处理/训练。

### 3.1 开发机环境

用于 Isaac Sim 仿真环境的注册、遥操作采集、标准评测、数据转换与模型训练。

```bash
conda create -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg=7.1.1 -c conda-forge
pip install lerobot[async]==0.4.2

conda create -n leisaac python=3.11
conda activate leisaac
pip install -U torch==2.7.0 torchvision==0.22.0
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
pip install flatdict==4.0.1 --no-build-isolation
pip install isaaclab[all]==2.3.0 --extra-index-url https://pypi.nvidia.com
pip install msgpack  pydantic websockets "protobuf>=6.33.5" grpcio grpcio-tools
git clone https://github.com/LightwheelAI/leisaac.git
cp  modify.patch leisaac/
cp -r assets/ leisaac/
cd leisaac
git apply modify.patch
pip install -e source/leisaac

```

### 3.2 版本对照

| 组件 | 开发机 | 板端 |
|:---|:---|:---|
| Python | 3.11 | 3.10 |
| CUDA | 13.0 | — |
| 关键包 | Isaac Sim 5.1.0, Isaac Lab 2.3.0, lerobot 0.4.2 | lerobot 0.4.2 |

### 3.3 环境自检

安装完成后，先跑一个低成本自检，确认仿真环境和任务注册正常：

```bash
conda run -n leisaac --no-capture-output python scripts/environments/list_envs.py
```

### 3.4 项目入口一览

| 阶段 | 入口脚本 | 用途 |
|:---|:---|:---|
| 环境检查 | `scripts/environments/list_envs.py` | 列出已注册的 LeIsaac 环境 |
| 数据采集 | `scripts/environments/teleoperation/teleop_se3_agent.py` | 遥操作录制仿真数据 |
| 数据转换 | `scripts/convert/isaaclab2lerobotv3.py` | HDF5 → LeRobot v3.0 |
| 仿真评测 | `scripts/evaluation/policy_inference.py` | 拉起仿真 + 连接策略服务 |
| IPU 推理 | `lerobot_ipu/server.py` | 板端 async 推理服务 |

### 3.5 板端环境（Comake Pi D3）

在板端同样需要安装 lerobot 环境：

```bash
pip install ffmpeg
pip install lerobot[async]==0.4.2
```

板端核心入口文件：

- `lerobot_ipu/server.py` — IPU async gRPC server 入口
- `lerobot_ipu/act.py` — ACT `.img` adapter（`py_ipu.IPUModel`）及共享工具

> 本仓库在板端的职责是提供 LeRobot async 协议与 IPU 推理工具链之间的桥接。不包含 Sigmastar SDK 安装脚本和 `Scripts/LLM_Converter/` 目录。

---

## 4. 数据采集

### 4.1 遥操作录制

使用 SO-101 leader 设备进行仿真遥操作并录制数据：

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-SO101-PickOrange-v0 \
  --teleop_device=so101leader \
  --port=/dev/ttyACM0 \
  --num_envs=1 \
  --device=cuda \
  --enable_cameras \
  --record \
  --dataset_file=./datasets/dataset.hdf5
```

| 参数 | 说明 |
|:---|:---|
| `--task` | LeIsaac 任务名，如 `LeIsaac-SO101-PickOrange-v0` |
| `--teleop_device` | 遥操作设备：`keyboard` / `gamepad` / `so101leader` |
| `--port` | 串口设备路径（so101leader 模式下使用） |
| `--num_envs` | 并行仿真环境数量 |
| `--device` | 计算设备：`cuda` / `cpu` |
| `--enable_cameras` | 启用相机渲染 |
| `--record` | 启用数据录制 |
| `--dataset_file` | 录制输出的 HDF5 文件路径 |

录制输出为 HDF5 格式，存放在 `datasets/` 目录下。如果使用真机 leader，先确认串口、远端 endpoint 和重标定参数。进入 IsaacLab 窗口后，按下键盘上的 `b` 键以开始远程操作。随后，您可以通过指定的远程操作设备（teleop_device）在仿真环境中控制机器人。完成操作后，若需重置环境，只需按下 `r` 键或 `n` 键。其中，`r` 键表示重置环境并将任务标记为失败，而 `n` 键表示重置环境并将任务标记为成功。

录制得到的 HDF5 文件需要转换为 LeRobot v3.0 格式才能用于训练：

```bash
python scripts/convert/isaaclab2lerobotv3.py \
  --task_name=LeIsaac-SO101-PickOrange-v0 \
  --repo_id=<ORG>/<DATASET> \
  --hdf5_root=./datasets \
  --hdf5_files=dataset.hdf5 \
  --push_to_hub=false
```

| 参数 | 说明 |
|:---|:---|
| `--task_name` | LeIsaac 任务名，如 `LeIsaac-SO101-PickOrange-v0` |
| `--repo_id` | HuggingFace 数据集 repo ID（`<ORG>/<DATASET>`） |
| `--hdf5_root` | HDF5 文件所在根目录 |
| `--hdf5_files` | 待转换的 HDF5 文件名 |
| `--push_to_hub` | 是否推送至 HuggingFace Hub：`true` / `false` |

### 4.2 公开数据集

也可以直接使用已转换好的公开 LeRobot 数据集，如LightwheelAI/leisaac-pick-orange(需要转换为lerobot datasets_v3格式)：

```
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
--repo-id LightwheelAI/leisaac-pick-orange \
--root ./datasets \
--push-to-hub False
```


---

## 5. 模型训练

### 5.1 启动训练

```bash
 HF_HUB_LOCAL_FILES_ONLY=1 lerobot-train \
  --policy.type=act \
  --dataset.repo_id=LightwheelAI/leisaac-pick-orange \
  --dataset.root="/home/sigmastar/sgs_robot/leisaac_sgs/datasets/raw/leisaac-pick-orange/LightwheelAI/leisaac-pick-orange" \
  --job_name=act \
  --policy.device=cuda \
  --output_dir="/home/sigmastar/sgs_robot/leisaac_sgs/outputs/leisaac-pick-orange" \
  --policy.push_to_hub=false
```

| 参数 | 说明 |
|:---|:---|
| `HF_HUB_LOCAL_FILES_ONLY` | 环境变量，设为 `1` 仅使用本地文件，避免联网 |
| `--policy.type` | 策略架构类型，ACT 模型设为 `act` |
| `--dataset.repo_id` | HuggingFace 数据集 repo ID |
| `--dataset.root` | 数据集本地路径 |
| `--job_name` | 训练任务名称 |
| `--policy.device` | 训练设备：`cuda` / `cpu` |
| `--output_dir` | checkpoint 输出目录 |
| `--policy.push_to_hub` | 是否推送模型至 HuggingFace Hub：`true` / `false` |

### 5.2 训练前检查清单

- [ ] 数据集已转为 v3.0 格式
- [ ] lerobot 版本与训练产出兼容（当前 0.4.2）

### 5.3 训练产出

训练完成后，checkpoint 目录（如 `/home/sigmastar/sgs_robot/leisaac_sgs/outputs/leisaac-pick-orange/checkpoints/last/pretrained_model`）应包含：

| 文件 | 说明 |
|:---|:---|
| `config.json` | ACT 配置，含 `chunk_size`、`n_action_steps` 等 |
| `policy_preprocessor.json` | 预处理管线，指向 normalizer safetensors |
| `policy_preprocessor_step_*_normalizer_processor.safetensors` | state/action 归一化统计量（mean/std） |
| 模型权重文件 | safetensors 格式 |

---

## 6. IPU 模型转换

将训练好的 PyTorch checkpoint 转换为 Sigmastar IPU 可执行的 `.img` 格式（参考Sigmastar IPU工具链），用于 Comake Pi D3 板端推理。或者下载已经转换好的模型（基于数据集 `LightwheelAI/leisaac-pick-orange` ），见[sgs_model_zoo](https://git.sigmastar.com.cn/sigmastar/sgs_model_zoo/examples/vision_action/act)。

### 6.1 模型文件布局

`.img` 文件需与 LeRobot checkpoint 放在同一目录：

```
/path/to/model/
├── act.img                                              # 编译后的 ACT 模型
├── config.json                                          # ACT 配置
├── policy_preprocessor.json                             # 预处理管线
└── policy_preprocessor_step_*_normalizer_processor.safetensors  # 归一化统计量
```

### 6.2 启动 IPU 推理服务

板端启动 async 推理 server：

```bash
  HF_HUB_LOCAL_FILES_ONLY=1 python -m lerobot_ipu.server --host 0.0.0.0 --port 8080 --obs-queue-timeout 2
```

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--host` | `0.0.0.0` | gRPC 绑定地址 |
| `--port` | `8080` | gRPC 绑定端口 |
| `--obs-queue-timeout` | `2.0` | 等待 observation 超时（秒） |
| `--show-log` | `False` | 启用 IPUModel 推理日志 |

---

## 7. 仿真验证

### 7.1 本地评测（开发机）

终端 1 — 启动 policy server：

```bash
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080
```

终端 2 — 运行仿真评测：

```bash
 HF_HUB_LOCAL_FILES_ONLY=1 python -u scripts/evaluation/policy_inference.py \
  --task=LeIsaac-SO101-PickOrange-v0 \
  --eval_rounds=10 \
  --policy_type=lerobot-act \
  --policy_host=127.0.0.1 \
  --policy_port=8080 \
  --policy_checkpoint_path=/home/sigmastar/sgs_robot/leisaac_sgs/outputs/leisaac-pick-orange/checkpoints/last/pretrained_model \
  --policy_language_instruction="Pick up the orange and place it on the plate" \
  --device=cuda \
  --enable_cameras
```

| 参数 | 说明 |
|:---|:---|
| `--task` | LeIsaac 任务名，如 `LeIsaac-SO101-PickOrange-v0` |
| `--eval_rounds` | 评测轮数 |
| `--policy_type` | 策略类型：本地评测用 `act`，板端联调用 `act` / `ipu_act` / `lerobot-act` |
| `--policy_host` | 策略服务地址（本地 `127.0.0.1`，板端填入实际 IP） |
| `--policy_port` | 策略服务端口 |
| `--policy_checkpoint_path` | checkpoint 路径或 `.img` 文件所在目录 |
| `--device` | 推理设备：`cuda` / `cpu` |
| `--enable_cameras` | 启用相机渲染 |

### 7.2 板端联调

板端启动 IPU server 后，在开发机的仿真端将 `policy_host` 指向板端 IP：

```bash
# 板端
python -m lerobot_ipu.server --host 0.0.0.0 --port 8080 --obs-queue-timeout 2

# 仿真端（policy_type 使用 act/ipu_act/lerobot-act，policy_host 指向板端 IP）
 HF_HUB_LOCAL_FILES_ONLY=1 python -u scripts/evaluation/policy_inference.py \
  --task=LeIsaac-SO101-PickOrange-v0 \
  --policy_type=lerobot-act \
  --policy_host=192.168.3.104 \
  --policy_port=8080 \
  --policy_checkpoint_path=/home/ubuntu/act/act.img \
  --enable_cameras
```
