# PhyAgentOS × leisaac-sgs 端到端集成包

把 [PhyAgentOS](https://github.com/PhyAgentOS/PhyAgentOS)（具身智能 OS）与 [leisaac-sgs](https://github.com/LightwheelAI/leisaac)（Sigmastar D3 IPU 上的 ACT 部署 + Isaac Sim 仿真）打通，在 D3 边缘板上完成一次「**自然语言指令 → 云端 LLM → 板端 IPU 策略推理 → Isaac Sim 机械臂执行**」的端到端闭环。

```
[用户]            [云端 LLM]              [D3 边缘板]                    [GPU 服务器]
 paos agent ─→ DeepSeek (OpenAI 兼容) ─→ PhyAgentOS + IPU ACT ─→ Isaac Sim + SO101
                                         skillruntime 调度          TargetWS 仿真执行
                                         gRPC 板内推理
```

- **版本**：基于 PhyAgentOS `v0.1.7.20260724-56174c8-4-gd59dd52`
- **依赖框架**：LeRobot `0.4.2`、Isaac Sim `5.1.0` + Isaac Lab `2.3.0`
- **目标硬件**：Sigmastar D3（IPU 推理）+ NVIDIA GPU（Isaac Sim 仿真）
- **机械臂**：SO101（6-DoF + gripper，joint position control）

---

## 1. 包内容

本目录给出 `PhyAgentOS-preview/PhyAgentOS/` 树下**与 SO101 + IPU ACT 链路相关的全部源文件**，路径与 PhyAgentOS 工程内的实际位置一一对应（直接覆盖即可）。

| 工程内路径 | 关键文件 | 作用 |
|---|---|---|
| `runtime/policy/ipu_act/` | `client.py` | `IPUACTPolicyClient`：gRPC client，自包含 protobuf descriptor + pickle shim，调用板端 IPU async policy server |
| `runtime/policy/` | `factory.py` | `parse_policy_endpoint` 支持 `grpc://`，`build_policy_client` 在 `kind == grpc` 时走 `_build_ipu_act_client`，从 `skillruntime_config["ipu_act"]` / `session_ctx` 取业务参数 |
| `runtime/skillruntime/policy/` | `ipu_act.py` | `IPUACTSkillRuntime`：SO101 + IPU ACT 专属 skill runtime，自带 `build_ipu_act_client`，驱动 observe → adapter → infer → action_chunk 主循环 |
| `runtime/adapters/so101/` | `act_policy_adapter.py` | `ACTPolicyAdapter`：SO101 policy adapter，含 **Isaac rad ↔ LeRobot motor degree** 坐标转换 |
| `runtime/adapters/so101/` | `target_adapter.py` | `SO101TargetAdapter`：SO101 target adapter，对接 Isaac Sim 观测格式 |
| `runtime/adapters/` | `factory.py` | 注册 `target_adapter://so101_isaac_adapter` 与 `policy_adapter://act_so101_adapter` |
| `runtime/targets/remote/so101_isaac/` | `server.py` | GPU 服务器侧 TargetWS server，承载 Isaac Sim 仿真（独立进程，不依赖 PhyAgentOS 包） |
| `runtime/targets/remote/so101_isaac/` | `proxy.py` | `SO101IsaacRemoteTargetProxy`：D3 侧 target proxy，注入 SO101 默认参数 + action 校验 |
| `templates/configs/runtime/contracts/` | `so101_isaac.runtime.yaml` | target runtime contract YAML（`so101_joint_pos_v1` action contract） |
| `templates/` | `TARGETS.md` / `SKILLRUNTIME.md` / `SESSIONS.md` | 已内置 `so101_isaac` target、`act_ipu_so101` skillruntime、session 示例条目 |


---

## 2. 架构与数据流

### 2.1 三端协同架构

| 端 | 角色 | 运行内容 |
|---|---|---|
| **云端** | 任务理解 | DeepSeek（或任意 OpenAI 兼容 LLM） |
| **边缘（D3）** | 策略推理 + 调度 | PhyAgentOS `paos agent` + `watchdog`（paos_venv, python3.11）+ IPU 策略服务器（lerobot venv, python3.10） |
| **远端（GPU）** | 执行验证 | Isaac Sim 5.1.0 + Isaac Lab 2.3.0 + SO101 + `so101_server.py`（leisaac conda env, python3.11） |

### 2.2 一次任务的端到端流程

```
1. 用户在 paos agent 输入自然语言
2. 云端 LLM 通过 OpenAI 兼容协议返回 write_file 工具调用
3. watchdog 轮询 SESSIONS.md，claim pending session
4. IPUACTSkillRuntime 通过 TargetWS 拉取观测（front 图 / wrist 图 / 6 维关节）
5. ACTPolicyAdapter 把 Isaac rad → LeRobot motor degree，组 obs dict
6. IPUACTPolicyClient 通过 gRPC 把观测送给 IPU 策略服务器
7. IPU 推理 ACT 模型 → 返回 [T, 6] action chunk（motor degree）
8. ACTPolicyAdapter 把 motor degree → Isaac rad
9. skill runtime 通过 TargetWS 下发 action chunk 到 Isaac Sim
10. 循环 4-9 直到 success / max_steps / timeout
```

### 2.3 关键协议

| 链路 | 协议 | 数据格式 |
|---|---|---|
| D3 → 云端 LLM | OpenAI 兼容 HTTP | JSON |
| D3 板内（skill runtime ↔ IPU 策略服务器）| LeRobot async gRPC | protobuf + pickle |
| D3 ↔ GPU 服务器 | TargetWS（WebSocket）| msgpack |

---

## 3. 环境准备

### 3.1 硬件要求

| 端 | 推荐 | 最低 |
|---|---|---|
| 云端 | DeepSeek API 或任意 OpenAI 兼容服务 | 任意可达的 OpenAI 兼容 API |
| 边缘 D3 | ≥ 4GB RAM、≥ 16GB 存储 | 4GB RAM、16GB 存储 |
| 远端 GPU | RTX 4090 24GB、≥ 32GB 内存 | RTX 3060 12GB、16GB 内存 |

### 3.2 操作系统与系统依赖

| 端 | 系统 | 关键依赖 |
|---|---|---|
| 云端 | 无要求 | 无 |
| 边缘 D3 | 出厂 Linux（内核 5.x） | gcc / cmake / pkg-config / curl / git |
| 远端 GPU | Ubuntu 22.04 LTS | NVIDIA Driver ≥ 535、CUDA 12.x、Isaac Sim 5.1.0、Isaac Lab 2.3.0 |

系统依赖安装示例（Ubuntu 22.04）：

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl wget \
    python3.11 python3.11-venv \
    python3.10 python3.10-venv
```

PhyAgentOS 代码拉取（基于 commit-d59dd52 版本）；
```bash
cd /root
git clone https://github.com/PhyAgentOS/PhyAgentOS.git PhyAgentOS-preview

cd PhyAgentOS-preview
git checkout d59dd52
```

PhyAgentOS 项目代码更新：
```bash
cd /root/PhyAgentOS-preview
git apply --index /root/full_diff.diff
```

### 3.3 D3 板端双 venv 设计

D3 板同时运行 PhyAgentOS 与 IPU 策略服务器，依赖差异较大（尤其 Python 主版本不同），采用**双 venv 隔离**：

```
/root/
├── paos_venv/                # PhyAgentOS（python3.11）
├── lerobot/                  # IPU 策略服务器 + LeRobot（python3.10）
├── PhyAgentOS-preview/       # PhyAgentOS 源码
├── leisaac-sgs/              # lerobot_ipu + calibrator_custom 工具链
├── lerobot-0.4.2/            # LeRobot 依赖源码
├── .PhyAgentOS/              # 运行时 workspace（SESSIONS.md 等）
└── full_diff.diff            # git diff 文件
```

创建命令：

```bash
# paos_venv（python3.11）
python3.11 -m venv /root/paos_venv
source /root/paos_venv/bin/activate
pip install --upgrade pip
cd /root/PhyAgentOS-preview && pip install -e .
deactivate

# lerobot venv（python3.10）
python3.10 -m venv /root/lerobot
source /root/lerobot/bin/activate
pip install --upgrade pip
cd /root/lerobot-0.4.2 && pip install -e .
cd /root/leisaac-sgs && pip install -e ./ipu_toolchain && pip install -e ./lerobot_ipu
deactivate
```

### 3.4 GPU 服务器 conda 环境

GPU 服务器侧 `so101_server.py` 不依赖 PhyAgentOS 包，但依赖 Isaac Sim + Isaac Lab，`so101_server.py`可放入 /leisaac-sgs 文件夹下：

```bash
conda create -n leisaac python=3.11 -y
conda activate leisaac
# 按 NVIDIA 官方文档装 Isaac Sim 5.1.0 + Isaac Lab 2.3.0
pip install msgpack websockets numpy opencv-python pillow
```

---

## 4. 启动三端

严格按以下顺序起。

### 4.1 进程 1：GPU 服务器 TargetWS server

```bash
ssh <user>@<gpu-server-ip>
conda activate leisaac
cd /path/to/so101_server.py

python so101_server.py \
  --host 0.0.0.0 --port 9005 \
  --gui
# 可选：--record-dir /tmp/so101_episodes --record-fps 30 录制回放
```

**成功标志**：终端输出 `TargetWS server listening on 0.0.0.0:9005`，Isaac Sim GUI 打开。

> `so101_server.py` 是独立脚本，`import` 时不会拉起 PhyAgentOS 包；运行它的 conda 环境只需 Isaac Sim + Isaac Lab + msgpack/websockets/numpy/cv2。

### 4.2 进程 2：D3 板 IPU 策略服务器

```bash
ssh root@<D3-ip>
source /root/lerobot/bin/activate
export LD_LIBRARY_PATH=/path/to/py_ipu.so/dir:$LD_LIBRARY_PATH

python -m lerobot_ipu.serve_ipu_async \
  --host 0.0.0.0 --port 8080 \
  --obs-queue-timeout 2
```

**成功标志**：终端输出 `IPU policy server listening on 0.0.0.0:8080`，无 `py_ipu` 报错。

### 4.3 进程 3：D3 板 paos agent（自动拉起 watchdog）

```bash
ssh root@<D3-ip>   # 另开一个 terminal
source /root/paos_venv/bin/activate
paos agent
```

**成功标志**：进入 paos agent 交互界面，后台日志显示 watchdog 已拉起并轮询 SESSIONS.md。

### 4.4 发送任务

在 paos agent 交互界面输入：

```
[paos] > 让机械臂去抓桌子上的橘子
```

预期事件链：

1. LLM 返回 `write_file`，SESSIONS.md 出现 `status: pending` session
2. watchdog claim → `status: running`
3. Isaac Sim 中机械臂开始动作
4. 完成后 SESSIONS.md `status: success`

### 4.5 判断是否跑通

满足以下两条即视为链路跑通：

1. Isaac Sim GUI 中机械臂产生**连续的、有方向性的动作**（不是单次抽动、不是卡死）
2. SESSIONS.md 中对应 session 的 `status` 最终迁移为 `success`

快速确认 session 状态：

```bash
grep -E "status:|session_id:" /root/.PhyAgentOS/workspace/SESSIONS.md | head -4
```

---

## 5. 关键模块说明

### 5.1 SESSIONS.md 文件协议

`SESSIONS.md` 是 LLM 与 watchdog 之间的**唯一通信媒介**，采用 YAML 格式，文件即状态机。

状态迁移：

```
pending → running → success / failed / timeout
                  → retry_pending → running  (attempted < max_retries)
```

调试时第一步永远是 `cat /root/.PhyAgentOS/workspace/SESSIONS.md` 看 session 停在哪个状态、`result` 字段记录了什么。

### 5.2 IPUACTSkillRuntime 调度主循环

`runtime/skillruntime/policy/ipu_act.py` 的 `IPUACTSkillRuntime.run_policy_loop` 是 SO101 + IPU ACT 链路的执行核心：

```
[1] target_handle.observe()                    拉取观测
[2] policy_adapter.to_policy_input()           Isaac rad → motor degree
[3] policy_client.infer()                      gRPC 调 IPU 策略服务器
[4] policy_adapter.from_policy_output()        motor degree → Isaac rad
[5] target_handle.action_chunk()               下发动作
[6] 循环 [1]-[5]，直到 success / done / max_steps
```

session_ctx 里注入的关键参数（来自 TARGETS.md + SKILLRUNTIME.md）：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `ipu_act.actions_per_chunk` | 单次推理 chunk 长度 | 16 |
| `ipu_act.policy_checkpoint_path` | 板上 ACT `.img` 路径 | `/root/leisaac-sgs/act/act_fixed.sim_sgsimg.img` |
| `ipu_act.timeout_s` | 单次 gRPC 推理超时 | 15.0 |
| `target.config.max_steps` | 单 session 最大步数 | 280 |
| `target.config.max_chunk_size` | target 端可接受 chunk 上限 | 100 |

### 5.3 IPU 策略服务器 + client 关键设计

**为什么 client 要做 pickle `__module__` 改写（shim）**：

服务端通过 pickle 序列化 `RemotePolicyConfig` / `TimedObservation` / `TimedAction` 等类。如果客户端的这些类定义在 `PhyAgentOS.*` 路径下，pickle 字节流里写死的就是 `PhyAgentOS.runtime.policy.ipu_act.client.TimedObservation`，服务端 `pickle.loads()` 时会试图导入 `PhyAgentOS` 包——但 IPU 板上没装 PhyAgentOS。

解决：`runtime/policy/ipu_act/client.py` 把这些 dataclass 的 `__module__` 改写成 `lerobot.async_inference.helpers`，并在 `sys.modules` 里注册同名 shim module，与服务端 `lerobot_ipu.act:install_async_helpers_shim()` 对称。pickle 字节流里就变成 `lerobot.async_inference.helpers.TimedObservation`，服务端能从自己 startup 时安装的 shim module 里反序列化，**无需服务端装 PhyAgentOS**。

同时，client **自包含 protobuf descriptor**：4 个消息类型（`Empty` / `Observation` / `Actions` / `PolicySetup`）从内嵌的 `FileDescriptorProto` 动态构造，**不依赖 `lerobot.transport.services_pb2`**。PhyAgentOS 侧只需 `grpcio` + `protobuf` 两个包。

### 5.4 坐标转换（Isaac rad ↔ LeRobot motor degree）

**关键正确性保证**：

- **Isaac Sim 关节值是弧度**（物理关节角度，rad）
- **ACT 模型训练时 state/action 用的是 LeRobot 「电机度数」**（每个关节映射到电机规格范围，约 [-100, 100]）

`runtime/adapters/so101/act_policy_adapter.py` 的 `ACTPolicyAdapter` 在两个方向各做一次转换：

```python
# to_policy_input：发送观测前 Isaac rad → LeRobot motor degree
proprio_data = _isaac_rad_to_lerobot_motor(proprio_data)

# from_policy_output：收到 action 后 LeRobot motor degree → Isaac rad
actions = _lerobot_motor_to_isaac_rad(actions)
```

**不做这两个转换，机械臂会乱动**——模型看到错乱坐标的 state，输出的 action 又被直接当成弧度下发。

转换表（来自 SO101 USD 关节极限 + LeRobot 电机规格）：

| 关节 | USD 极限（度） | 电机极限（度） |
|---|---|---|
| shoulder_pan | (-110, 110) | (-100, 100) |
| shoulder_lift | (-100, 100) | (-100, 100) |
| elbow_flex | (-100, 90) | (-100, 100) |
| wrist_flex | (-95, 95) | (-100, 100) |
| wrist_roll | (-160, 160) | (-100, 100) |
| gripper | (-10, 100) | (0, 100) |

### 5.5 注册 ID 对照

| 层 | 注册 ID / 类名 | 源文件 |
|---|---|---|
| target adapter | `target_adapter://so101_isaac_adapter` | `runtime/adapters/so101/target_adapter.py` → `SO101TargetAdapter` |
| policy adapter | `policy_adapter://act_so101_adapter` | `runtime/adapters/so101/act_policy_adapter.py` → `ACTPolicyAdapter` |
| target proxy | `SO101IsaacRemoteTargetProxy` | `runtime/targets/remote/so101_isaac/proxy.py` |
| skill runtime | `IPUACTSkillRuntime` | `runtime/skillruntime/policy/ipu_act.py` |
| policy client | `IPUACTPolicyClient`（endpoint 形如 `grpc://host:port`） | `runtime/policy/ipu_act/client.py` |

TARGETS.md / SKILLRUNTIME.md 里的关联字段：

```yaml
# TARGETS.md
- id: so101_isaac
  supported_skillruntimes: [act_ipu_so101]
  runtime:
    target_runtime: SO101IsaacRemoteTargetProxy
    target_endpoint: targetws://so101-isaac-host:9005
    target_adapter: target_adapter://so101_isaac_adapter

# SKILLRUNTIME.md
- id: act_ipu_so101
  runtime: IPUACTSkillRuntime
  policy:
    policy_client: ipu_act_grpc
    policy_adapter: policy_adapter://act_so101_adapter
  config:
    ipu_act:
      policy_type: act
      policy_checkpoint_path: /root/leisaac-sgs/act/act_fixed.sim_sgsimg.img
      actions_per_chunk: 16
```

### 5.6 二次开发扩展点

| 扩展方向 | 改动点 |
|---|---|
| 替换云端 LLM | 改 PhyAgentOS LLM 配置（`base_url`、`api_key`），新模型必须支持 `tool_calls` |
| 同类换 ACT checkpoint | 改 `SKILLRUNTIME.md` 的 `config.ipu_act.policy_checkpoint_path` |
| 换策略架构（ACT → SmolVLA） | 切换 adapter、改 `policy_type`、换模型目录 |
| 对接真实机械臂 | 实现 `so101_server.py` 里的 `reset` / `observe` / `action_chunk` 三个 TargetWS 方法，改 `TARGETS.md` 的 `target_endpoint` |

---

## 6. 常见问题排查

按链路环节分组，每条「症状 / 原因 / 解法」三层结构。

### 6.1 环境与网络

| 症状 | 原因 | 解法 |
|---|---|---|
| paos agent 调 LLM 报 5xx 或超时 | 网络出口未放行 LLM 域名白名单 | `curl -sS https://<llm>/v1/models` 看返回码；放行 https:443 |
| watchdog 报 `TargetWS connect failed` | GPU 服务器端口未开放或 IP 错 | `nc -zv <gpu-ip> 9005`，检查 `TARGETS.md` 里 `target_endpoint: targetws://...:9005` |
| session 卡在 pending，watchdog 不 claim | watchdog 未启动或读不到 SESSIONS.md | `ps aux \| grep paos`，确认 `/root/.PhyAgentOS/workspace/SESSIONS.md` 存在 |
| IP 漂移导致链路断开 | DHCP 续租 | 把 GPU 服务器改静态 IP |

### 6.2 Python / venv / py_ipu

| 症状 | 原因 | 解法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'py_ipu'` | `LD_LIBRARY_PATH` 未含 `py_ipu.so` | `export LD_LIBRARY_PATH=/path/to/py_ipu.so/dir:$LD_LIBRARY_PATH` |
| 启动报 python 版本不对 | 在错误 venv 启动 | paos → paos_venv（py3.11）；IPU server → lerobot venv（py3.10）；GPU server → leisaac conda（py3.11） |
| `pip install` 报 `No space left on device` | D3 板 `/tmp` 空间不足 | `export TMPDIR=/root/tmp` 后重装 |
| 双 venv 串包 | venv 创建用了 `--system-site-packages` | 重建 venv 不带该参数 |

### 6.3 SESSIONS.md / 链路

| 症状 | 原因 | 解法 |
|---|---|---|
| `SchemaValidationError` 直接 failed | YAML 格式错或必填字段缺失 | `python -c "import yaml; yaml.safe_load(open('SESSIONS.md'))"` 校验 |
| session 卡在 running 不结束 | `max_steps` 过大 / success 条件未触发 | 调小 `target.config.max_steps` |
| 机械臂动一下就停，报 `PolicyTimeoutError` | 单次推理超 `ipu_act.timeout_s` | 调大到 20-30，看 IPU 推理耗时日志 |
| LLM 返回 write_file 但 SESSIONS.md 没更新 | path 参数写成相对路径 | 看 paos agent 日志里的 `write_file` 调用参数 |
| **机械臂乱动** | 坐标系不匹配（rad vs motor degree） | 确认 `act_policy_adapter.py` 的 `_isaac_rad_to_lerobot_motor` / `_lerobot_motor_to_isaac_rad` 被调用 |
| `No module named 'PhyAgentOS'`（服务端报） | client pickle `__module__` 没改写 | 确认 `client.py` 把 dataclass `__module__` 改写成 `lerobot.async_inference.helpers` |
| `unsupported policy endpoint: grpc://...` | policy/factory.py 没处理 `grpc` scheme | 确认用的是本包提供的 `runtime/policy/factory.py`（含 `grpc` 分支） |
| `unsupported target adapter: so101_isaac_adapter` | adapters/factory.py 没注册 so101 | 确认用的是本包提供的 `runtime/adapters/factory.py` |

### 6.4 通用排查思路

```
[1] 看 paos agent 日志          LLM 调用 / 工具执行是否出错
[2] 看 watchdog 日志            claim / observe / infer / action 哪一步出错
[3] 看 IPU 策略服务器日志       推理报错或超时
[4] 看 GPU 服务器 server.py 日志   仿真执行报错
[5] cat SESSIONS.md            session 停在哪个状态、result 记录了什么
```

80% 的问题都能从上述五处日志直接定位。

---

## 7. 关键命令速查

```bash
# 网络连通性自检（D3 板上）
curl -sS -o /dev/null -w "%{http_code}\n" https://<llm-api>/v1/models
nc -zv <gpu-server-ip> 9005
nc -zv 127.0.0.1 8080

# 确认 venv / conda env
which python                                  # D3 板上必须是 paos_venv 或 lerobot venv 的 python
conda activate leisaac                        # GPU 服务器上

# 确认进程
ps aux | grep -E "paos|watchdog|serve_ipu"

# 看 session 状态
grep -E "status:|session_id:" /root/.PhyAgentOS/workspace/SESSIONS.md | head -4

# YAML 语法校验
python -c "import yaml; yaml.safe_load(open('/root/.PhyAgentOS/workspace/SESSIONS.md'))"
```

---

## 8. 依赖清单

Python 依赖详见 [requirements/](requirements/) 目录：

- `paos_venv.txt` — PhyAgentOS 侧（python3.11）
- `lerobot_venv.txt` — IPU 策略服务器侧（python3.10）
- `gpu_server.txt` — GPU 服务器侧（Isaac Sim 仿真）

各端核心依赖概览：

| 端 | 核心包 |
|---|---|
| paos_venv | grpcio、protobuf、msgpack、numpy、pydantic、pyyaml、websockets、openai |
| lerobot venv | torch（CPU 版）、numpy、grpcio、protobuf、pillow、calibrator_custom、py_ipu |
| GPU 服务器（leisaac conda） | isaacsim 5.1.0、isaac-lab 2.3.0、torch（CUDA 版）、msgpack、websockets、numpy、opencv-python、leisaac |

---

## 9. 参考文档

- PhyAgentOS 官方仓库：https://github.com/PhyAgentOS/PhyAgentOS

---

## 10. 版本与许可证

- **本包版本**：基于 PhyAgentOS `v0.1.7.20260724-56174c8-4-gd59dd52`
- **LeRobot 版本**：0.4.2
- **Isaac Sim 版本**：5.1.0 + Isaac Lab 2.3.0
- **License**：随 PhyAgentOS（MIT）与 leisaac-sgs 各自的协议
