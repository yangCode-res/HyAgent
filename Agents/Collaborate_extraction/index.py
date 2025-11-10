# Agents/Entity_normalize/index.py

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from openai import OpenAI
from Memory.index import Memory, Subgraph
from Logger.index import get_global_logger
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from Core.Agent import Agent

logger = get_global_logger()

# 控制台颜色（如果重定向到文件，只是普通字符串，不影响）
ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[96m"    # LLM 相关
ANSI_GREEN = "\033[92m"   # 汇总信息


class EntityNormalizationAgent(Agent):
    """
    子图级实体归一化 Agent

    三步流程：

    1）规则归一化（同子图 + 同类型，确定性字符串匹配）
    2）BioBERT 相似度候选（同子图 + 同类型，基于 description/name）
    3）LLM 裁决合并（候选批次并行请求，合并操作串行执行）
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        biobert_dir: str = "/home/nas2/path/models/biobert-base-cased-v1.1",
        sim_threshold: float = 0.94,
    ):
        system_prompt = """

"""
        super().__init__(client, model_name, system_prompt)
        self.logger = get_global_logger()

    # ===================== 对外入口 =====================

    def process(self, memory: Memory) -> None:
        """
        对所有 subgraph 执行：
        1）规则归一化；
        2）用 BioBERT 生成候选对；
        3）并行 LLM 裁决；
        带总体进度条。
        """
        
        

    