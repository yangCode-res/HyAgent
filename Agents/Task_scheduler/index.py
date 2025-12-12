import json

from openai import OpenAI

from Core.Agent import Agent
from TypeDefinitions.PipelineDefinitions.index import PipeLine

"""
任务调度 Agent，负责根据用户查询判断所需知识图类型，并调度相应的 Agent 模块流水线。
输入:用户查询字符串
输出:对应的 Agent 模块流水线（PipeLine 对象）
调用入口：agent.process(user_input:str)
"""

class TaskSchedulerAgent(Agent):
    system_prompt="""
    You are a Task Scheduler Agent designated to manage and schedule Agent modules efficiently on the project of medical hypothesis generation through knowledge graph construction and reasoning.
    Your responsibilities include:
    1. Receiving queries from users and try to understand their requirements and analyze thier complexity in requirements.
    2. Distinguish the complexity of the tasks and decide which kind of knowledge graph to generate and which specialized Agent module pipeline is best suited to handle the request.
    3. Scheduling and delegating tasks to the appropriate Agent modules based on their expertise and capabilities.
    There exists eight specialized Agent modules:
    1.Entity Extraction Agent
    2.Entity Normalization Agent
    3.Relationship Extraction Agent
    4.Collaboration Extraction Agent
    5.Causal Extraction Agent
    6.Temporal Extraction Agent
    7.Alignment Triple Agent
    8.Mechanism Extraction Agent
    and five types of knowledge graphs:
    1.Basic Knowledge Graph
    2.Causal Knowledge Graph(without mechanism)
    3.Temporal Knowledge Graph
    4.Causal Knowledge Graph(with mechanism)
    5.Comprehensive Knowledge Graph(with causal,mechanism and temporal information)
    Among these Agent modules, the Entity Extraction Agent, Entity Normalization Agent, Relationship Extraction Agent,Collaboration Extraction Agent and Alignment Triple Agent are fundamental and must be executed for every task to build the Basic Knowledge Graph.
    The Causal Extraction Agent, Mechanism Extraction Agent, and Temporal Extraction Agent are specialized modules that should be invoked based on the specific requirements of the user's query.
    All the optional modules should be placed after the collaboration extraction module and before the alignment triple agent in the execution pipeline.
    Your goal is to judge the user's intention from the query and judge which type of knowledge graph is needed.
    Example:
    Input:Could you recommend some potential factors related to Alzheimer's disease based on existing research but still lack of mining?
    Output:To address this query, a Comprehensive Knowledge Graph is required to capture the multifaceted,so you should judge the knowledge graph type to generate.
    So you return response in json format:
    {"type":"Comprehensive Knowledge Graph",
     "question_complexity":"high"
    }
    """

    """
    or output could be:
    {"type":"Comprehensive Knowledge Graph",
     "pipeline":["Entity Extraction Agent","Entity Normalization Agent","Relationship Extraction Agent","Collaboration Extraction Agent","Causal Extraction Agent","]
     "question_complexity":"high"
    }
    """
    def __init__(self,client:OpenAI,model_name:str):
        super().__init__(system_prompt=self.system_prompt,client=client,model_name=model_name)
    
    def process(self,user_input:str)->PipeLine:
        response=self.call_llm(user_input)
        try:
            response=json.loads(response)
            type=response.get("type","")
            if type not in ["Basic Knowledge Graph","Causal Knowledge Graph (without mechanism)","Temporal Knowledge Graph","Causal Knowledge Graph (with mechanism)","Comprehensive Knowledge Graph"]:
                raise ValueError("Invalid knowledge graph type")
            pipeline=PipeLine(graph_type=type,user_query=user_input,client=self.client,model_name=self.model_name)
        except Exception:
            raise ValueError("Failed to parse LLM response or invalid knowledge graph type")
        return pipeline
    '{\n    "type": "Causal Knowledge Graph (with mechanism)",\n    "question_complexity": "high"\n}'