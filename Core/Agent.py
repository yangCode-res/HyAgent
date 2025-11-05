from __future__ import annotations

from dataclasses import dataclass, field, asdict
from multiprocessing import process
import time
from typing import Any, Dict, List, Optional
from urllib import response
from venv import logger
from Logger.index import get_global_logger
from openai import OpenAI
from Store.index import get_memory
@dataclass
class Agent:
    """
    通用父类 Agent。

    必备属性（对齐 meta_agent 中的规划字段）：
    - template_id: 模板 ID
    - name: Agent 名称
    - responsibility: 该 Agent 的职责说明
    - entity_focus: 关注的实体类型列表（在不同工程中可为字符串或枚举）
    - relation_focus: 关注的关系类型列表（在不同工程中可为字符串或枚举）
    - priority: 该 Agent 的优先级（数值越小，优先级越高）
    """

    template_id: str
    name: str
    responsibility: str
    entity_focus: List[Any] = field(default_factory=list)
    relation_focus: List[Any] = field(default_factory=list)
    priority: int = 1

    # 可选扩展字段（不强依赖于 Agent2 的实现，便于后续扩展）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def configure(
        self,
        *,
        template_id: Optional[str] = None,
        name: Optional[str] = None,
        responsibility: Optional[str] = None,
        entity_focus: Optional[List[Any]] = None,
        relation_focus: Optional[List[Any]] = None,
        priority: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """按需更新 Agent 的配置。"""
        if template_id is not None:
            self.template_id = template_id
        if name is not None:
            self.name = name
        if responsibility is not None:
            self.responsibility = responsibility
        if entity_focus is not None:
            self.entity_focus = list(entity_focus)
        if relation_focus is not None:
            self.relation_focus = list(relation_focus)
        if priority is not None:
            self.priority = int(priority)
        if metadata is not None:
            self.metadata.update(metadata)
    
    def __init__(self,client:OpenAI,model_name:str,system_prompt:str):
        """初始化 Agent 基类。
        Args:
            client (OpenAI): OpenAI 客户端实例。
            model_name (str): 使用的模型名称。
            system_prompt (str): Agent提示语。
            metadata (Dict[str, Any], optional): 额外的元数据。默认为空字典。
        """
        self.client=client
        self.model_name=model_name
        self.system_prompt=system_prompt
        self.metadata={
            "total_calls":0,
            "total_call_prompt_tokens":0,
            "total_call_completion_tokens":0,
            "total_call_processing_time":0.0
        }
        self.logger=get_global_logger()
        self.memory = get_memory()
    def call_llm(self,prompt:str,temperature:float=0.1,max_tokens:Optional[int]=None,system_prompt:Optional[str]=None):
        """调用语言模型接口。
        Args:
            prompt (str): 用户输入的提示语。
            temperature (float, optional): 生成文本的随机性。默认为0.1。
            max_tokens (Optional[int], optional): 生成文本的最大长度。默认为None。
            system_prompt (Optional[str], optional): 系统提示语，覆盖默认的system_prompt。默认为None。"""
        
        start_time=time.time()
        try:
            messages=[
                {"role":"system","content":system_prompt if system_prompt else self.system_prompt},
                {"role":"user","content":prompt}
            ]
            call_kwargs={
                "model":self.model_name,
                "messages":messages,
                "temperature":temperature
            }
            if max_tokens:
                call_kwargs["max_tokens"]=max_tokens
            response=self.client.chat.completions.create(**call_kwargs)

            content=response.choices[0].message.content.strip()
            prompt_tokens=response.usage.prompt_tokens
            completion_tokens=response.usage.completion_tokens
            processing_time=time.time()-start_time
            
            self.metadata["total_calls"]+=1
            self.metadata["total_call_prompt_tokens"]+=prompt_tokens
            self.metadata["total_call_completion_tokens"]+=completion_tokens
            self.metadata["total_call_processing_time"]+=processing_time

            return content
        except Exception as e:
            processing_time=time.time()-start_time
            logger.error(f"LLM 调用失败: {e}")
            raise e

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典，便于日志/序列化。"""
        return asdict(self)

    def parse_json(self,response:str)->List[Dict]: # type: ignore
        import json
        
        try:
            if "[" in response and "]" in response:
                json_str=response[response.find("["):response.rfind("]")+1]
                return json.loads(json_str)
            return json.loads(response)
        except Exception as e:
            logger=get_global_logger()
            logger.info(f"Failed to parse JSON response {e}")
            return []
        
    # 预留的运行接口，子类按需实现
    def run(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
        """执行 Agent 的主流程（需由具体子类实现）。"""
        raise NotImplementedError("Subclasses must implement run()")


