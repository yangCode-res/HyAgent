from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


class ChatLLM:
    """
    OpenAI Chat API 简单封装。

    功能：
    - 读取环境变量 OPENAI_API_KEY、OPENAI_API_BASE_URL、OPENAI_MODEL
    - 维护会话历史（system/user/assistant）
    - 单轮与多轮对话
    - 设置/替换 system 提示
    - 重置会话（可选择保留 system）
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 8000,
    ) -> None:
        # try:
        #     env_path = find_dotenv(usecwd=True)
        #     if env_path:
        #         load_dotenv(env_path, override=False)
        # except Exception:
        #     pass
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_API_BASE_URL")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = os.environ.get("OPENAI_MODEL")
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._history: List[Dict[str, str]] = []
        if system:
            self.set_system(system)

    # ---------- 会话与消息 ----------
    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def set_system(self, content: str, *, replace: bool = True) -> None:
        """设置 system 提示；replace=True 时替换已有 system。"""
        if replace:
            self._history = [m for m in self._history if m.get("role") != "system"]
        self._history.insert(0, {"role": "system", "content": content})

    def reset(self, *, keep_system: bool = True) -> None:
        """重置会话；可选择保留 system 提示。"""
        if keep_system:
            system_msgs = [m for m in self._history if m.get("role") == "system"]
            self._history = system_msgs
        else:
            self._history = []

    # ---------- 调用接口 ----------
    def chat(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """多轮：在历史基础上追加 user，再请求并记录 assistant。"""
        temp = self._resolve_temperature(temperature)
        mtk = self._resolve_max_tokens(max_tokens)

        messages = self._history + [{"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=mtk,
        )
        content = resp.choices[0].message.content

        self._history.append({"role": "user", "content": prompt})
        self._history.append({"role": "assistant", "content": content})
        return content

    def single(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> str:
        """单轮：不污染现有历史，可临时传入 system。"""
        temp = self._resolve_temperature(temperature)
        mtk = self._resolve_max_tokens(max_tokens)

        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        else:
            msgs.extend([m for m in self._history if m.get("role") == "system"])
        msgs.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temp,
            max_tokens=mtk,
        )
        return resp.choices[0].message.content

    # ---------- 配置 ----------
    def set_model(self, model: str) -> None:
        self.model = model

    def set_default_params(self, *, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> None:
        if temperature is not None:
            self.temperature = float(temperature)
        if max_tokens is not None:
            self.max_tokens = int(max_tokens)

    # ---------- 辅助 ----------
    def _resolve_temperature(self, value: Optional[float]) -> float:
        return float(self.temperature if value is None else value)

    def _resolve_max_tokens(self, value: Optional[int]) -> int:
        return int(self.max_tokens if value is None else value)


