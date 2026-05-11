# Latent_KalmanNet_TSP

本仓库是论文 **Latent-KalmanNet: Learned Kalman Filtering for Tracking From High-Dimensional Signals** 的一份 PyTorch 复现实验代码，主要用于从高维视觉观测中估计低维动态状态。论文的核心思想是：先用卷积 Encoder 将 28 x 28 图像观测映射到低维 latent 表示，再用 KalmanNet 在 latent 空间里进行数据辅助的 Kalman filtering；Encoder 和 KalmanNet 通过交替训练互相配合，使 latent 表示更适合后续跟踪。

## 论文与代码对应关系

论文中逐步构造的算法在代码里的对应关系如下：

| 论文模块 | 代码位置 | 说明 |
| --- | --- | --- |
| 高维观测生成 | `model_Lorenz.py`, `model_Pendulum.py`, `Extended_sysmdl_visual.py`, `PendulumGeneration_new.py` | Lorenz 和 Pendulum 的状态转移、图像观测生成 |
| Encoder / Encoder + Prior | `main_AE.py` | 卷积编码器、带 prior 的卷积编码器、单独训练和测试 |
| EKF baseline | `EKF_visual.py`, `EKFTest_visual.py` | 使用 Encoder 输出作为 latent observation 的 EKF |
| Latent-KalmanNet | `KalmanNet_nn_visual_new_architecture_gpu.py`, `Pipeline_KF_visual_gpu.py` | KalmanNet gain 网络、训练和推理 pipeline |
| 统一配置入口 | `configurations/config_file.yaml`, `config_script.py`, `config.py` | 数据集、场景、噪声、训练/加载开关 |
| 主实验脚本 | `main_visual.py` | 加载数据、加载/训练 Latent-KalmanNet、测试并绘图 |

## 环境安装

建议使用 Python 3.9 或 3.10。代码会自动检测 CUDA；没有 GPU 时也可以跑 CPU，但完整训练会慢很多。推荐使用 conda 管理环境：

```powershell
conda create --prefix .\.conda python=3.10 -y
conda activate .\.conda
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果需要安装指定 CUDA 版本的 PyTorch，请先在已激活的 conda 环境中按 PyTorch 官网命令安装 `torch`，再执行 `pip install -r requirements.txt`。例如 CPU 版本可直接使用上面的命令；GPU 版本应按本机 CUDA/驱动选择对应的 PyTorch 安装命令。

> 注意：`PendulumGeneration_new.py` 使用了 `Image.ANTIALIAS`，该符号在 Pillow 10 中被移除，所以 `requirements.txt` 将 Pillow 限制在 `<10.0`。

## 实验数据

论文主要复现两个高维视觉跟踪实验：

| 数据集 | 论文设置 | 代码设置 |
| --- | --- | --- |
| Pendulum | 部分可观测，状态为角度和角速度，单张图像只能恢复角度；观测为 28 x 28 灰度图 | `dataset_name: "Pendulum"`，`model_Pendulum.py` 中 `m=2`, `d=1`, `real_q2=0.001` |
| Lorenz | 三维混沌系统，全状态估计；观测为 28 x 28 Gaussian point-spread 图像，并加入 Salt-and-Pepper 噪声 | `dataset_name: "Lorenz"`，`model_Lorenz.py` 中 `m=3`, `d=3`, `real_q2=0.1`, `J=5`, `delta_t=0.02` |

默认数据规模在 `config_script.py` 中设置：

| 场景 | 训练集 | 验证集 | 测试集 | 训练长度 | 测试长度 |
| --- | --- | --- | --- | --- | --- |
| `Baseline` / `Decimation` | 1000 | 100 | 100 | 200(Lorenz) / 400(Pendulum) | 200(Lorenz) / 400(Pendulum) |
| `Test_Long_Trajectories`(Lorenz) | 2 | 2 | 100 | 200 | 2000 |

代码默认从以下路径读取数据：

```text
Simulations/<dataset_name>/states_q2_<real_q2>_<sinerio>.npz
Simulations/<dataset_name>/observations_q2_<real_q2>_<sinerio>.npz
```

仓库当前没有提交 `Simulations/` 数据目录，因此首次运行前需要生成或放入数据。

### 生成 Lorenz 数据

1. 打开 `configurations/config_file.yaml`，设置：

```yaml
dataset_name : "Lorenz"
sinerio : "Baseline"      # 或 "Decimation", "Test_Long_Trajectories"
data_gen_flag : True
real_r2 : 0.5             # 论文/代码常用: 0.5, 0.1, 0.01, 0
```

2. 运行：

```powershell
python main_visual.py
```

`config.py` 会在导入阶段调用 `DataGen(...)` 生成 `Simulations/Lorenz/*.npz`。生成完成后，把 `data_gen_flag` 改回 `False`，再进行训练或测试。

### 生成 Pendulum 数据

Pendulum 的生成脚本是：

```powershell
python PendulumGeneration_new.py
```

该脚本默认生成 `states_q2_0.001_r2_0.3.npz` 和 `observations_q2_0.001_r2_0.3.npz`。而主流程默认读取 `states_q2_0.001_Baseline.npz` 和 `observations_q2_0.001_Baseline.npz`，并在 `main_AE.py` 中通过 `sp_noise(input, real_r2)` 动态加入 Salt-and-Pepper 噪声。因此复现 Pendulum 时建议将生成脚本中的保存文件名改成主流程期望的 `*_Baseline.npz`，或将 `config.py` 中的 `path_for_states` / `path_for_observations` 改到实际文件名。

## 快速推理已有模型

仓库包含部分已训练权重：

```text
Encoder/
KNetLatent_models/
```

基本步骤：

1. 准备好 `Simulations/<dataset>/...npz` 数据。
2. 在 `configurations/config_file.yaml` 中设置：

```yaml
data_gen_flag : False
load_KNetLatent_trained : True
flag_Train : False
prior_flag : True
fix_encoder_flag : False
```

3. 运行：

```powershell
python main_visual.py
```

程序会输出 Latent-KalmanNet 和 Encoder 的测试 MSE(dB)，并在部分分支中保存论文风格的 `.eps` 图。

> 重要路径提醒：当前 `config.py` 拼出的 KNet 权重路径形如 `KNetLatent_models/Lorenz/KNetLatent_optimal_...pt`，但仓库中的 Lorenz 权重实际放在 `KNetLatent_models/Lorenz/J=1/`、`KNetLatent_models/Lorenz/J=5 baseline/` 或 `KNetLatent_models/Lorenz/Decimation/` 子目录下。若直接加载失败，请把目标 `.pt` 复制到 `config.py` 打印/报错的目标路径，或直接修改 `path_KNetLatent_trained` 指向对应子目录。

## 从头复现论文实验

论文复现不是单次命令完成的；需要按实验场景循环不同噪声、模型信息和轨迹长度。推荐顺序如下。

### 1. 预训练 Encoder

运行 `main_AE.py` 单独训练 Encoder 或 Encoder + Prior。脚本底部有可修改参数：

```python
dataset_name = "Lorenz"       # 或 "Pendulum"
sinerio = "Baseline"          # "Baseline", "Decimation"
flag_prior = True             # True: Encoder + Prior; False: 纯 Encoder
flag_train = True
flag_load_model = False
batch_size = 512
num_epochs = 30
prior_r2 = 1                  # Lorenz 常用 1/3/4; Pendulum 常用 8
```

然后运行：

```powershell
python main_AE.py
```

训练完成后权重保存到 `Encoder/<dataset>/<scenario>/...pt`。

### 2. 训练 Latent-KalmanNet

在 `configurations/config_file.yaml` 中设置：

```yaml
load_KNetLatent_trained : False
flag_Train : True
fix_encoder_flag : False      # 配置语义: False 训练 Encoder; True 固定 Encoder、训练 KGain
prior_flag : True
lr_kalman : 0.001
wd_kalman : 0.01
batch_size : 16
epoches : 300
```

然后运行：

```powershell
python main_visual.py
```

`Pipeline_KF_visual_gpu.py` 会在验证集 MSE 更优时保存 Latent-KalmanNet 权重到 `path_KNetLatent_trained`。

### 3. 复现 Lorenz Baseline, Fig. 10(a)

设置：

```yaml
dataset_name : "Lorenz"
sinerio : "Baseline"
real_r2 : 0.5     # 依次改为 0.5, 0.1, 0.01, 0
prior_flag : True
```

`model_Lorenz.py` 保持：

```python
J = 5
delta_t = 0.02
prior_r2 = 1      # real_r2=0.5 时代码里会使用 prior_r2=4
```

每个 `real_r2` 重复：

1. 生成/加载同一数据；
2. 训练或加载 Encoder + Prior；
3. 训练或加载 Latent-KalmanNet；
4. 运行 `python main_visual.py` 记录测试 MSE。

代码中内置的论文数值参考位于 `main_visual.py` 的 Lorenz 分支：

| 方法 | `p_r=0.5` | `p_r=0.1` | `p_r=0.01` | `p_r=0` |
| --- | ---: | ---: | ---: | ---: |
| Encoder | 5.8 | -0.5 | -3.7 | -5.6 |
| Encoder + Prior | 2.58 | -2.67 | -6.1 | -6.84 |
| Encoder + Prior + EKF | 0.51 | -3.0 | -6.31 | -7.16 |
| RKN | -1.0 | -4.2 | -6.9 | -7.8 |
| Latent-KalmanNet | -1.91 | -4.92 | -7.2 | -7.94 |

### 4. 复现 Lorenz 长轨迹, Fig. 10(b)

设置：

```yaml
dataset_name : "Lorenz"
sinerio : "Test_Long_Trajectories"
```

`config_script.py` 会使用 `T_train=200`, `T_test=2000`。论文/代码参考结果：

| 方法 | `p_r=0.5` | `p_r=0.1` | `p_r=0.01` | `p_r=0` |
| --- | ---: | ---: | ---: | ---: |
| Encoder | 5.4 | -0.58 | -3.9 | -6.0 |
| Encoder + Prior | 1.63 | -3.74 | -6.72 | -7.57 |
| Encoder + Prior + EKF | -0.61 | -4.34 | -7.11 | -7.91 |
| RKN | 5.7 | -0.2 | -3.5 | -5.8 |
| Latent-KalmanNet | -2.4 | -5.45 | -7.91 | -8.54 |

### 5. 复现 Lorenz 模型失配和 Decimation, Fig. 11

模型失配实验：在 `model_Lorenz.py` 中将 `J=5` 改为 `J=1`，但数据仍使用 `J=5` 生成。参考结果位于 `main_visual.py` 的 `Wrong F` 分支。

Decimation 实验：设置

```yaml
dataset_name : "Lorenz"
sinerio : "Decimation"
```

`Extended_sysmdl_visual.py` 会先生成高频轨迹再每 20 步抽样。参考 Latent-KalmanNet 结果为 `[3.3, -0.46, -3.4, -3.6]` dB。

### 6. 复现 Pendulum, Fig. 6

设置：

```yaml
dataset_name : "Pendulum"
sinerio : "Baseline"
prior_flag : True
real_r2 : 0.1     # 代码绘图分支会遍历 0.9, 0.4, 0.1, 0.01 等噪声
```

`model_Pendulum.py` 中关键设置：

```python
m = 2
d = 1
real_q2 = 0.001
prior_r2 = 8
T_test = 400
```

运行：

```powershell
python main_visual.py
```

`main_visual.py` 的 Pendulum 分支会绘制设计步骤对比，并内置论文风格的参考结果：

| 方法 | `10log10(1/r^2)=10` | `4` | `2` | `1` |
| --- | ---: | ---: | ---: | ---: |
| Encoder after alternating | 1.6 | -0.2 | -1.8 | -4.5 |
| Encoder | 0.0 | -1.2 | -3.1 | -4.9 |
| Encoder + Prior | -4.0 | -5.1 | -6.5 | -8.28 |
| Encoder + Prior + EKF | -4.8 | -6.1 | -7.3 | -9.3 |
| Latent-KalmanNet | -8.2 | -8.3 | -9.8 | -11.1 |

## 常用配置说明

| 配置项 | 含义 |
| --- | --- |
| `dataset_name` | `"Lorenz"` 或 `"Pendulum"` |
| `sinerio` | 实验场景，代码沿用原作者拼写：`"Baseline"`, `"Decimation"`, `"Test_Long_Trajectories"` |
| `data_gen_flag` | `True` 生成数据，`False` 读取已有数据 |
| `real_r2` | Salt-and-Pepper 观测噪声概率，Lorenz 常用 `0.5, 0.1, 0.01, 0` |
| `load_KNetLatent_trained` | 是否加载已训练 Latent-KalmanNet |
| `flag_Train` | 是否训练 Latent-KalmanNet |
| `fix_encoder_flag` | 配置语义为是否固定 Encoder；实际冻结逻辑请以 `Pipeline_KF_visual_gpu.py::NNTrain` 为准 |
| `prior_flag` | 是否使用带 prior 的 Encoder |
| `warm_start_flag` | 使用预编码 latent 序列时的开关，默认 `False` |

## 已知注意事项

1. 代码中 `sinerio` 是原作者拼写，配置文件中也需要保持这个键名。
2. `KalmanFilter_test_visual.py` 引用了未提交的 `Linear_KF_visual`，但主复现流程不使用该文件。
3. `Simulations/` 数据未随仓库提交，完整复现实验必须先生成或放入数据。
4. 已训练 KNet 权重的目录层级和默认拼接路径可能不一致，使用预训练模型前请确认 `path_KNetLatent_trained`。
5. `fix_encoder_flag` 的注释语义是固定 Encoder，但当前 `Pipeline_KF_visual_gpu.py::NNTrain` 中 `True` 分支会把整个模型参数设为可训练；若要严格复现“固定 Encoder/训练 KGain”，需要在该分支中显式将 `model_encoder.parameters()` 设为 `requires_grad=False`。
6. 论文结果受随机种子、PyTorch 版本、CPU/GPU 浮点差异影响，MSE(dB) 允许有小幅波动。代码默认在 `config.py` 中使用 seed `2`，`main_AE.py` 脚本底部默认 seed `0`。

## 引用

```bibtex
@article{buchnik2024latent,
  title={Latent-KalmanNet: Learned Kalman Filtering for Tracking From High-Dimensional Signals},
  author={Buchnik, Itay and Revach, Guy and Steger, Damiano and van Sloun, Ruud J. G. and Routtenberg, Tirza and Shlezinger, Nir},
  journal={IEEE Transactions on Signal Processing},
  volume={72},
  pages={352--367},
  year={2024}
}
```
