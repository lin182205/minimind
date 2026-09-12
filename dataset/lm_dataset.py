"""语言模型训练数据集。

本文件包含两种数据格式：

1. PretrainDataset（预训练）
   每行 JSON 形如 ``{"text": "一段文本"}``，整段文本都参与监督。
2. SFTDataset（监督微调）
   每行 JSON 包含 ``conversations`` 列表；模型能看到整段对话，但通常只有
   assistant 消息对应的 token 参与 loss。

本文统一使用以下维度符号：
    S：固定序列长度，即 max_length
    B：DataLoader 的 batch size

``__getitem__`` 返回单条样本时：
    input_ids: [S]，torch.long
    labels:    [S]，torch.long

经过 DataLoader 自动堆叠后：
    input_ids: [B, S]
    labels:    [B, S]

input_ids 保存词表索引；labels 保存每个位置的监督目标。labels 中的 -100
不是词表 id，而是 PyTorch CrossEntropyLoss 默认忽略的标记（ignore_index）。
"""

import json
import os
import random
from typing import Any

import torch
from datasets import Features, Sequence, Value, load_dataset
from torch.utils.data import Dataset

# Hugging Face fast tokenizer 自己可能开多线程，而 DataLoader 又会启动多个子进程。
# 禁用 tokenizer 内部并行可避免嵌套并行带来的警告、线程争抢或 fork 后死锁风险。
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """训练前对消息列表做轻量数据增强。

    Args:
        conversations: 非空消息列表，每条消息至少包含 role 和 content。
        add_system_ratio: 原对话没有 system 消息时，随机添加 system prompt 的概率。

    Returns:
        原消息列表，或在头部拼接了一条 system 消息的新列表。

    这里使用 Python 的 ``random`` 模块；训练脚本中的 setup_seed 会控制其随机性。
    """
    # 工具调用数据的 system 消息还携带工具定义。随意插入 system prompt 可能改变
    # chat template 对工具定义的解析，因此只要任一消息含非空 tools 就完整保留。
    # any(...) 会逐个检查生成器表达式，一旦发现 True 就短路返回，不再继续遍历。
    if any(conv.get('tools') for conv in conversations):    #单条样本
        return conversations

    system_prompts = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]

    # 数据约定 conversations 非空，所以可以访问 conversations[0]。
    # random.random() 返回 [0.0, 1.0) 的浮点数；默认约 20% 的样本会进入分支。
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            # ``[新元素] + 原列表`` 会创建新列表，不会在原列表上原地插入。
            # random.choice 从候选 prompt 中等概率选择一个。
            return [
                {'role': 'system', 'content': random.choice(system_prompts)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """按概率删除 chat template 生成的空思考块。

    ``empty_think_ratio`` 表示空思考块的保留概率。默认值 0.2 时，条件
    ``random.random() > 0.2`` 约有 80% 的概率成立，因此约 80% 会被删除。
    ``str.replace`` 会删除字符串内所有完全匹配的空思考块。
    """
    empty_think = '<think>\n\n</think>\n\n'
    if empty_think in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace(empty_think, '')
    return prompt_content


class PretrainDataset(Dataset):
    """把纯文本 JSON/JSONL 数据转换为定长的因果语言模型样本。"""

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # load_dataset('json') 同时支持 JSON 和 JSONL。split='train' 表示直接取返回的
        # train Dataset，而不是保留 ``DatasetDict({'train': ...})`` 外层。
        # Hugging Face Dataset 底层使用 Arrow，可按索引读取，不需要先转成 Python list。
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        """供 DataLoader、Sampler 和 len(dataset) 查询样本总数。"""
        return len(self.samples)

    def __getitem__(self, index) -> Any:
        """读取并编码一条预训练样本，返回两个形状均为 [S] 的张量。"""
        # sample 是类似 {'text': '...'} 的字典。
        sample = self.samples[index]

        # 预留两个位置给 BOS/EOS，因此正文最多编码 S-2 个 token。
        # add_special_tokens=False 防止 tokenizer 自动再添加一套特殊 token。
        # tokenizer(...) 返回 BatchEncoding；.input_ids 在单条文本输入下是 list[int]，
        # 此处长度满足 0 <= len(tokens) <= S-2。
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Python 中 ``[pad_id] * n`` 会重复 n 次，再与 tokens 拼接。
        # 补齐后 len(input_ids) 恒为 S；torch.long 是 Embedding 所需的整数索引类型。
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)  # [S]

        # 因果语言模型在 model.forward 内部会用 input_ids[t] 预测 labels[t+1]。
        # clone() 创建独立张量，否则修改 labels 也会连带修改 input_ids。
        labels = input_ids.clone()  # [S]
        # 布尔索引的形状也是 [S]；为 True 的 padding 位置被批量写成 -100。
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # 如果模型需要显式 attention_mask，可使用下面的 [S] 0/1 张量。
        # 当前训练代码只传 input_ids 和 labels，且 padding 位已通过 -100 排除 loss。
        # attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return input_ids, labels


class SFTDataset(Dataset):
    """将多轮对话编码为定长 SFT 样本，只监督 assistant 回答。"""

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 显式声明 Arrow schema，避免不同 JSONL 行缺少可选字段时推断出不一致类型。
        # conversations 是变长消息列表；每条消息可以包含以下字符串字段：
        #   role: system/user/assistant/tool
        #   content: 正文
        #   reasoning_content: assistant 的思考内容
        #   tools: 工具定义的 JSON 字符串（通常放在 system 消息）
        #   tool_calls: 工具调用的 JSON 字符串
        # 缺失的可选字段由 Dataset 表示为 None。
        features = Features({
            'conversations': [{
                'role': Value('string'),
                'content': Value('string'),
                'reasoning_content': Value('string'),
                'tools': Value('string'),
                'tool_calls': Value('string'),
            }]
        })
        self.samples = load_dataset(
            'json',
            data_files=jsonl_path,
            split='train',
            features=features,
        )

        # 这两个变量虽然名为 bos_id/eos_id，实际类型都是 list[int]，可能含多个 token：
        #   bos_id 对应 "<|im_start|>assistant\n"，用于定位 assistant 回答开头；
        #   eos_id 对应 "<|im_end|>\n"，用于定位该回答结尾。
        # add_special_tokens=False 很重要，否则用于匹配的子序列可能被额外特殊符号污染。
        self.bos_id = tokenizer(
            f'{tokenizer.bos_token}assistant\n', 
            add_special_tokens=False,
        ).input_ids                 #tokenizer返回字典，从中取出input_ids
        self.eos_id = tokenizer(
            f'{tokenizer.eos_token}\n',
            add_special_tokens=False,
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """将结构化消息转换为符合 tokenizer 模板的完整对话字符串。"""
        messages = []
        tools = None

        for message in conversations:
            # dict(message) 做一层浅拷贝。下面要把 JSON 字符串解析成对象；使用副本可避免
            # 直接改写 Dataset 返回的原消息字典。
            message = dict(message)

            if message.get("role") == "system" and message.get("tools"):
                # 某些数据把工具列表存成 JSON 字符串，另一些数据可能已经是 Python 对象。
                # isinstance(..., str) 配合条件表达式兼容这两种表示。
                tools = (
                    json.loads(message["tools"])
                    if isinstance(message["tools"], str)
                    else message["tools"]
                )

            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                # json.loads 把形如 '[{"function": ...}]' 的字符串还原为 list/dict，
                # 以便 Jinja chat template 按字段访问工具名和参数。
                message["tool_calls"] = json.loads(message["tool_calls"])

            messages.append(message)

        # apply_chat_template 根据 tokenizer_config.json 中的 Jinja 模板插入 role、
        # <|im_start|>/<|im_end|>、思考块及工具调用格式。
        # tokenize=False：此处先返回 str，后续还要做字符串级数据增强；
        # add_generation_prompt=False：训练数据已有 assistant 回答，无需追加一个等待生成的头；
        # tools=tools：把解析后的工具定义单独交给模板渲染。
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    def generate_labels(self, input_ids):
        """根据 assistant 起止标记生成 SFT 标签。

        Args:
            input_ids: 已截断并补齐到 S 的 Python ``list[int]``。

        Returns:
            长度为 S 的 ``list[int]``。assistant 内容及其消息结束标记保留原 token
            id，其余位置为 -100。

        算法会处理多轮对话中的多个 assistant 区间。例如：
            input_ids: [system/user..., assistant头, 回答1, 结束, user...,
                        assistant头, 回答2, 结束, padding...]
            labels:    [-100......, -100......, 回答1, 结束, -100...,
                        -100......, 回答2, 结束, -100......]
        """
        # int 是不可变对象，因此 ``[-100] * S`` 可以安全创建一维列表。
        labels = [-100] * len(input_ids)  # [S]
        i = 0

        while i < len(input_ids):
            # 切片 ``input_ids[i:i+n]`` 产生候选子序列；列表 == 会逐元素比较，
            # 因而这里是在判断位置 i 是否出现完整的 assistant 头标记。
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # assistant 头本身仍保持 -100，从头标记之后才开始监督。
                start = i + len(self.bos_id)
                end = start

                # 向右寻找第一个完整的消息结束标记。
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                # range 的右端不包含在内，所以写 end + len(eos_id) 才会把整个结束标记
                # 也纳入 loss；min(...) 防止切片末端越过最大序列长度 S。
                for j in range(
                    start,
                    min(end + len(self.eos_id), self.max_length),
                ):
                    labels[j] = input_ids[j]

                # 条件表达式：找到 eos 时直接跳到其后，避免重复扫描本段回答；
                # 没找到时说明已扫描至序列末尾，直接结束外层循环。
                i = (
                    end + len(self.eos_id)
                    if end < len(input_ids)
                    else len(input_ids)
                )
            else:
                i += 1

        # 边界说明：若 assistant 回答被 max_length 截断且 eos_id 已丢失，当前算法会把
        # assistant 头后的所有剩余位置都填入 labels；若其后存在 padding，padding 也会
        # 被填回而不再是 -100。这是当前实现的既有行为，并非额外的 padding 掩码。
        return labels

    def __getitem__(self, index):
        """读取一段对话，返回形状均为 [S] 的 input_ids 和 labels。"""
        sample = self.samples[index]

        # conversations 是 list[dict]，长度是该样本的消息轮数，不是 token 长度。
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)  # str
        prompt = post_processing_chat(prompt)  # str

        # tokenizer(prompt).input_ids 是变长 list[int]，先右侧截断到最多 S 个 token。
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # ``+=`` 对列表做原地扩展；乘数为 S-len(input_ids)，补齐后长度严格等于 S。
        input_ids += [self.tokenizer.pad_token_id] * (
            self.max_length - len(input_ids)
        )
        labels = self.generate_labels(input_ids)

        # === 调试 next-token 对齐关系时可取消下面两行注释 ===
        # zip(input_ids[:-1], labels[1:]) 对应“当前位置输入 -> 下一位置标签”；
        # !r 显示转义后的字符串表示，:16s 表示至少占 16 个字符宽度。
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> "
        #           f"Y={self.tokenizer.decode([input_ids[i + 1]])!r:16s} label={y}")

        # 单样本返回 [S]；DataLoader 会沿新维度堆叠为 [B, S]。
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
