# MiniMind 当前环境与编译检查

本报告记录本次本地检查结果。代码基于提交 `0d3007c49483615d7b0df33d61d59a9027945617`，后续修改代码或依赖后需要重新验证。检查未修改训练源码或原有 requirements.txt。

Python 字节码编译、核心模块导入和小模型普通训练测试通过；默认训练入口尚不能直接开始训练，可选的 CUDA torch.compile 后端探测失败。

## 实际使用的环境

| 项目 | 检查结果 |
| --- | --- |
| 操作系统 | Windows 10，10.0.19045 |
| Conda 环境 | minimind |
| 检查所用解释器 | `D:\Miniconda3\envs\minimind\python.exe` |
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| PyTorch CUDA 运行时 | 12.4（cu124 构建） |
| GPU | NVIDIA T600 Laptop GPU |
| GPU 计算能力 | 7.5 |
| CUDA 是否可用 | 是 |
| 原生 BF16 支持 | 否 |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| datasets | 3.6.0 |
| pyarrow | 25.0.1 |
| numpy | 1.26.4 |
| swanlab | 0.7.11 |
| wandb | 0.18.3 |
| pip | 26.1.2 |
| Triton | 未安装；torch.compile CUDA 后端实际探测失败 |

当前工具终端没有激活上述 Conda 环境，直接执行 `python` 指向 `D:\python3.11\python.exe`。该解释器没有安装 torch、transformers、datasets，不能直接用于本项目训练。VS Code 应选择 minimind 环境的解释器；终端也需要使用该环境。

## requirements.txt 对照

27 项不重复的有效依赖均已安装，且版本全部符合声明。`python -m pip check` 返回 `No broken requirements found.`。此检查验证包声明的依赖关系，不代表所有可选功能均已运行验证。

| 依赖 | 声明版本与实际版本（全部一致） |
| --- | --- |
| datasets | 3.6.0 |
| datasketch | 1.6.4 |
| Flask | 3.0.3 |
| Flask_Cors | 4.0.0 |
| jieba | 0.42.1 |
| jsonlines | 4.0.0 |
| marshmallow | 3.22.0 |
| ngrok | 1.4.0 |
| nltk | 3.8 |
| numpy | 1.26.4 |
| openai | 1.59.6 |
| psutil | 5.9.8 |
| pydantic | 2.11.5 |
| rich | 13.7.1 |
| scikit_learn | 1.5.1 |
| sentence_transformers | 2.3.1 |
| simhash | 2.1.2 |
| tiktoken | 0.10.0 |
| transformers | 4.57.6 |
| jinja2 | 3.1.2 |
| trl | 0.13.0 |
| ujson | 5.1.0 |
| wandb | 0.18.3 |
| streamlit | 1.50.0 |
| einops | 0.8.1 |
| swanlab | 0.7.11 |
| modelscope | 1.37.0 |

依赖文件还存在两处整理事项：`jsonlines==4.0.0` 重复声明一次；torch 和 torchvision 的版本行被注释，不能依靠此文件明确固定其 CUDA 构建。

完整的 120 个已安装 Python 分发包版本见 [requirements.windows.snapshot.txt](../requirements.windows.snapshot.txt)。该文件按实际包元数据导出，不包含凭据或包下载地址；它是 Windows 环境版本快照，不是带哈希的锁文件，也不包含 NVIDIA 驱动、Conda 非 Python 包等系统组件。Linux 部署时应重新建立环境并选择与服务器匹配的 PyTorch CUDA 构建。

## 实际执行的代码检查

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| Python 字节码编译 | 通过，8/8 | 对 model、dataset、trainer 下全部 .py 文件执行 py_compile.compile(doraise=True)，输出写入临时目录并清理 |
| 核心模块导入 | 通过，4/4 | model.model_minimind、dataset.lm_dataset、trainer.trainer_utils、trainer.pretrain |
| CLI 帮助入口 | 通过 | 使用 minimind 解释器运行 trainer/pretrain.py --help，退出码 0 |
| 实际 tokenizer 加载 | 通过 | 从 trainer/ 加载，词表 6400，PAD/BOS/EOS ID 分别为 0/1/2 |
| CPU 普通模型训练步 | 通过 | 前向、有限 loss、全部参数有限梯度、AdamW 更新 |
| CPU MoE 训练步 | 通过 | 前向、有限 loss、全部参数有限梯度、AdamW 更新 |
| CUDA FP16 普通模型训练步 | 通过 | 前向、有限 loss、全部参数有限梯度、AdamW 更新 |
| CUDA FP16 MoE 训练步 | 通过 | 前向、有限 loss、全部参数有限梯度、AdamW 更新 |
| CUDA torch.compile 后端 | 失败 | 对简单加法函数使用默认 Inductor 后端，在第一次执行时报 BackendCompilerFailed：Cannot find a working triton installation |
| 默认训练启动 | 失败 | CPU 启动、输出指向临时目录，退出码 1，卡在默认 tokenizer 加载 |

训练步测试使用 hidden_size=64、2 层、词表 128、batch_size=2、序列长度 16 的随机输入。CPU loss 约为 4.8760（普通模型）与 4.8957（MoE）；CUDA FP16 loss 约为 4.8411 与 4.8794。CPU 和 CUDA 使用不同随机输入，这些 loss 不用于比较数值一致性。

以上验证普通执行模式的基本数值和优化器更新能力，不代表默认 768 维模型、真实数据集、多轮训练、断点恢复或 Linux 多卡训练已经通过。本次未使用 monkey patch 修复模型；当前源码中的 MoeFeedForward 类名已正确。

## 默认训练仍需处理的问题

1. Tokenizer 实际在 `trainer/tokenizer.json` 和 `trainer/tokenizer_config.json`，而 init_model 默认在项目 `model/` 中查找。实测抛出 `ValueError: Unrecognized model in .../Minimind/model`。应统一 tokenizer 目录或在训练入口显式传入路径。
2. 当前 pretrain.py 的数据、输出和 checkpoint 路径仍使用 `../dataset`、`../out`、`../checkpoints`；入口尚未提供 `--tokenizer_path` 与 `--checkpoint_dir` 参数，`--save_dir` 也没有传给 init_model。应重新统一当前版本的入口路径。
3. 检查时 dataset/ 中没有完成的默认 JSONL 数据，只有一个 `.crdownload` 临时下载文件。应等待下载完成，确认数据格式、完整性与路径后再训练。
4. 本机 T600 不支持原生 BF16。当前环境继续验证普通 GPU 训练时使用 `--dtype float16 --use_compile 0`；此组合已通过小模型测试。
5. 训练循环仍在末批保存之后才执行不足一个梯度累积周期的最后一次更新，因此最终保存文件漏掉该更新。累积中途保存也不保留待累积梯度。正式训练和续训前应修复保存边界。

## 本地解释器与检查命令

在项目根目录的 PowerShell 中可直接指定已验证的解释器：

```powershell
& D:/Miniconda3/envs/minimind/python.exe -m pip check
& D:/Miniconda3/envs/minimind/python.exe -B trainer/pretrain.py --help
```

完整依赖快照和本报告用于记录当前环境，不会自动安装依赖，也不会修复上述训练入口与保存问题。
