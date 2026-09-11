"""MiniMind 全参数监督微调（Full SFT）训练脚本。

本文统一使用以下维度符号：
    B：batch size，即一次送入模型的样本数
    S：sequence length，即 ``max_seq_len``
    H：hidden size，即模型隐藏层维度
    V：vocabulary size，即词表大小

数据在主要步骤中的形状为：
    单条样本            input_ids/labels: [S]
    DataLoader 组批后   input_ids/labels: [B, S]
    模型隐藏状态        hidden_states:     [B, S, H]
    模型输出            logits:            [B, S, V]
    错位后的训练数据    logits[:, :-1]:    [B, S-1, V]
                        labels[:, 1:]:      [B, S-1]

SFTDataset 先把所有 label 初始化为 -100，再仅填回 assistant 回答片段的 token；
交叉熵通过 ``ignore_index=-100`` 忽略其余位置，因此只监督 assistant 的回答。
"""

import os
import sys

# 允许既用 ``python trainer/train_full_sft.py`` 直接运行，也能正确导入项目根目录下的包。
# ``__file__`` 是当前脚本路径；dirname 取 trainer 目录，``..`` 再回到项目根目录。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 必须先导入 datasets：用于规避 Windows 下 pyarrow 与 torch 的 DLL 加载冲突（issue #771）。
import datasets  # noqa: F401
import argparse
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """训练一个 epoch。

    注意：本函数使用在主程序中创建的全局对象 ``args/model/optimizer/scaler`` 等。

    Args:
        epoch: 从 0 开始的 epoch 编号。
        loader: 产出 ``(input_ids, labels)`` 的 DataLoader，形状均为 [B, S]。
        iters: 当前 epoch 未跳批前的总 step 数，用于学习率、日志和保存判断。
        start_step: 断点前已经完成的 step 数；0 表示从 epoch 开头训练。
        wandb: 此项目实际传入 swanlab 模块；未启用时为 None。
    """
    start_time = time.time()
    last_step = start_step

    # ``enumerate(..., start=n)`` 只改变 step 的编号，并不会自动跳过数据。
    # 真正的跳批由主程序中的 SkipBatchSampler 完成。
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # DataLoader 组批后，两者形状均为 [B, S]，dtype 为 torch.long。
        # input_ids 是词表索引；labels 中 -100 表示该 token 不参与 loss。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # 使用“全局 step”计算余弦衰减学习率。optimizer.param_groups 是参数组字典列表；
        # 即使这里只有一个参数组，逐组赋值也兼容以后设置不同权重衰减/学习率的情况。
        lr = get_lr(
            epoch * iters + step,
            args.epochs * iters,
            args.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # GPU 上 autocast_ctx 会让适合的算子自动使用 bf16/fp16；CPU 上它是
        # nullcontext()，即“什么也不做”的上下文管理器，从而复用同一套 with 语法。
        with autocast_ctx:
            # input_ids/labels: [B, S]。
            # res.logits: [B, S, V]；模型内部把 logits[:, :-1] 和 labels[:, 1:]
            # 错开一位，训练“根据前面的 token 预测下一个 token”。
            # res.loss 和 res.aux_loss 都是标量（0 维张量）；非 MoE 模型的 aux_loss 为 0。
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss

            # 每个小 batch 的 loss 先除以累积次数，使 accumulation_steps 个小 batch
            # 的梯度之和近似等价于一个更大的 batch。单进程有效 batch 约为
            # B * accumulation_steps；DDP 全局还要再乘进程数 world_size。
            loss = loss / args.accumulation_steps

        # GradScaler 只在 float16 时真正缩放 loss，避免较小梯度下溢；bf16/CPU 时
        # scaler 被禁用，但这些 API 会以直通方式工作，因此无需另写训练分支。
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            # clip_grad_norm_ 必须看到真实梯度，所以先 unscale_，再做全局梯度范数裁剪。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # step() 更新参数；update() 根据本轮是否出现 inf/NaN 调整缩放因子。
            scaler.step(optimizer)
            scaler.update()

            # set_to_none=True 不把梯度张量填零，而是释放并设为 None，通常更省显存。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 日志恢复除以 accumulation_steps 之前的可读 loss。
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 已耗时 / 已完成 step = 每 step 平均耗时，再乘剩余 step；// 60 转为分钟。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, '
                f'epoch_time: {eta_min:.1f}min'
            )
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min,
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'

            # DDP 将原模型放在 .module 中；torch.compile 将原模型放在 ._orig_mod 中。
            # getattr(obj, name, default) 在属性不存在时返回 default，因此同一段代码兼容
            # 普通模型、DDP 模型和 compile 后的模型。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()

            # 字典推导式逐个把参数转为 fp16 并搬到 CPU，减小推理权重文件及 GPU 占用。
            # 这里保存一份权重到 save_dir；下面的 lm_checkpoint 还会在 checkpoints
            # 目录保存推理权重及包含优化器、step、scaler 等信息的可续训文件。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints',
                scaler=scaler,
            )
            model.train()
            del state_dict

        # 尽早解除大张量引用，让 Python/PyTorch 可以回收对象或复用显存。
        del input_ids, labels, res, loss

    # 如果 epoch 的 batch 数不能整除 accumulation_steps，循环内不会更新最后一组
    # 残余梯度；这里补做一次 optimizer step，避免丢掉这些样本贡献的梯度。
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型权重保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的文件名前缀")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="每个进程的 batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载子进程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度范数裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔（step）")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔（step）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度 H")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Transformer 层数")
    parser.add_argument(
        '--max_seq_len',
        default=768,
        type=int,
        help="训练的最大序列长度 S（中文通常 1 token 约对应 1.5~1.7 个字符）",
    )
    parser.add_argument(
        '--use_moe',
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用 MoE 架构（0=否，1=是）",
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help="随机种子（DDP 中初始化为 seed+rank，每轮重设为 seed+epoch）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/sft_t2t_mini.jsonl",
        help="训练数据路径",
    )
    parser.add_argument(
        '--from_weight',
        default='pretrain',
        type=str,
        help="基于哪份权重训练；设为 none 则从随机初始化开始",
    )
    parser.add_argument(
        '--from_resume',
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测断点并续训（0=否，1=是）",
    )
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 SwanLab 记录训练")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="SwanLab 项目名")
    parser.add_argument(
        "--use_compile",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用 torch.compile 加速（0=否，1=是）",
    )
    args = parser.parse_args()

    # ========== 1. 初始化运行环境和随机种子 ==========
    # torchrun 启动时会提供 RANK/LOCAL_RANK 等环境变量并初始化 NCCL 进程组；
    # 普通单卡运行时函数直接返回 0，且 dist.is_initialized() 为 False。
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    # 不同 rank 使用不同种子，避免各进程的数据增强随机结果完全相同。
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数并检查续训断点 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    # 条件表达式 ``A if condition else B``：仅 from_resume=1 时读取 resume 文件。
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置自动混合精度（AMP） ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # nullcontext 让 CPU 路径也能使用 ``with autocast_ctx:``，但不会改变数据精度。
    autocast_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. 配置实验记录 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 项目沿用 wandb 变量名，但这里实际导入的是 swanlab；只让 rank 0 写日志。
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = (
            f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-"
            f"LearningRate-{args.learning_rate}"
        )
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume,
        )

    # ========== 5. 创建模型、数据集和优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # DistributedSampler 主要让每个 rank 读取不同的数据子集；为保证各 rank 样本数
    # 相等，数据量不能整除 world_size 时，默认会补少量重复索引。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # fp16 的有效数值范围较小，需要 GradScaler；bf16 通常不需要，所以 enabled=False。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从 checkpoint 恢复训练状态 ==========
    start_epoch, start_step = 0, 0  # Python 的元组拆包：分别给两个变量赋值。
    if ckp_data:
        # 不仅恢复模型参数，也恢复 AdamW 的动量/方差和 GradScaler 状态；
        # 否则即使权重相同，续训轨迹也不能真正接上。
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)  # get 在键不存在时安全地返回默认值 0。

    # ========== 7. 编译模型并包装为分布式模型 ==========
    if args.use_compile == 1:
        # torch.compile 会捕获并优化计算图；首次迭代有编译开销，后续迭代通常更快。
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # 每个进程只绑定一张 GPU；backward 时 DDP 自动在各 rank 间同步梯度。
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # ``a and b()`` 利用短路求值：a 为 None 时不执行 b()。
        # set_epoch(epoch) 让 DistributedSampler 每轮产生不同但各 rank 协调一致的顺序。
        train_sampler and train_sampler.set_epoch(epoch)

        setup_seed(args.seed + epoch)
        # 单卡时 randperm 生成 [0, len(train_ds)) 的无放回随机排列，类型转为 Python list。
        # DDP 时实际使用 train_sampler，indices 只是备用分支，不会传给批采样器。
        indices = torch.randperm(len(train_ds)).tolist()

        # 只在恢复的第一个 epoch 跳过已完成 batch，后续 epoch 从第 0 个 batch 开始。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # ``train_sampler or indices``：DDP 时取 DistributedSampler，单卡时取 indices。
        # SkipBatchSampler 先按 batch_size 聚合样本索引，再丢弃前 skip 个 batch。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            # batch_sampler 直接产出“一批样本索引”，因此无需再传 batch_size/shuffle。
            # 最后一批未设置 drop_last，故实际形状可为 [B_last, S]，其中 B_last <= B。
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            # 锁页内存可加快 CPU -> CUDA 的拷贝；在纯 CPU 训练时通常没有收益。
            pin_memory=True,
        )

        if skip > 0:
            Logger(
                f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前 {start_step} 个 step，'
                f'从 step {start_step + 1} 开始'
            )
            # len(loader) 是跳批后的剩余 batch 数，加回 skip 才是本 epoch 的总 step 数。
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        # barrier 等待所有 rank 完成，随后销毁进程组并释放通信资源。
        dist.barrier()
        dist.destroy_process_group()
