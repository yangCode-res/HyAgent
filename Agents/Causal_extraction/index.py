import json
import os
from typing import List
from openai import OpenAI
from Core.Agent import Agent

class CausalExtractionAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str):
        self.system_prompt=""""""
        pass