from Core.Agent import Agent
from Logger.index import get_global_logger
from Store.index import get_memory
from TypeDefinitions.TimeDefinitions.TimeFormat import TimeFormat

"""
时序信息抽取 Agent。
基于已有的三元组和文本，抽取三元组对应的时序信息并补充到三元组中。
输入: 无（从内存中获取已有三元组和文本）
输出: 无（将补充了时序信息的三元组更新回内存）
调用入口：agent.process()
时序信息包括三类：
1.瞬时时间点（如“2020年1月1日”，“下午3点”，“下周一”）
2.区间时间段（如“1月至3月”，“下午2点到4点之间”，“上周”）
3.相对时间表达（如“昨天”，“两周后”，“三天前”）
抽取结果以结构化 JSON 格式返回。
"""

class TemporalExtractionAgent(Agent):
    def __init__(self,client,model_name="deepseek-chat",memory=None):
        self.memory=memory or get_memory()
        self.logger=get_global_logger()
        system_prompt="""
        You are a Temporal Information Extraction Expert Agent.
        Your task is to extract temporal information based on extracted triples,looking for their temporal information in their origin text.
        The temporal information could be classified into the following categories:
        1.instant time points (e.g., "January 1, 2020", "3 PM", "next Monday")
        2.interval time periods (e.g., "from January to March", "between 2 PM and 4 PM", "last week")
        3.relative time expressions (e.g., "yesterday", "in two weeks", "three days ago")
        The categories above should be returned in a structured JSON format.
        Example JSON format:
        {
            "type":"instant",
            "value":"2020-01-01T15:00:00Z",
            "origin_text":"The event is scheduled for January 1, 2020 at 3 PM."
        }
        {
            "type":"interval",
            "start_time":"2020-01-01",
            "end_time":"2020-03-31",
            "origin_text":"The project runs from January to March."
        }
        {
            "type":"relative",
            "offset":"-1",
            "granularity":"days",
            "origin_text":"The meeting was yesterday."
        }
        Ensure the extracted temporal information is accurate and complete.
        The precise temporal information should be in the form of ISO 8601 format where applicable.
        If there is no temporal information in the text, respond with an empty JSON object: {}.
        The response should only contain the JSON objects without any additional text.
        Each JSON object should correspond to one triple in sequence.
        example response:
        [
            {
            "type":"instant",
            "value":"2020-01-01T15:00:00Z",
            "origin_text":"The event is scheduled for January 1, 2020 at 3 PM."
        },###This JSON object corresponds to the first triple
         {
            "type":"relative",
            "offset":"-1",
            "granularity":"days",
            "origin_text":"The meeting was yesterday."
        }###This JSON object corresponds to the second triple
        ]
        NOTICE: Each triple must have a corresponding JSON object in the response, even if it's an empty object.
        The temporal information should be extracted based on triples, neither just entities or relations alone.
        Each triple could only have one temporal information. The priority order is interval>instant>relative.
        1.If a triple has interval time information, extract only that.
        2.If a triple has no interval time information but has instant time information, extract only that.
        3.If a triple has neither interval nor instant time information but has relative time information, extract only that.
        4.If a triple has no temporal information, respond with an empty JSON object: {}.
        """
        super().__init__(client,model_name,system_prompt)

    def process(self):
        subgraphs=self.memory.subgraphs
        for subgraph in subgraphs:
            triples=subgraph.relations.triples
            text=subgraph.meta.get("text","")
            temporal_info=self.extract_temporal_info(triples,text)
            for i,triple in enumerate(triples):
                if temporal_info[i].type=="unknown":
                    continue
                elif temporal_info[i].type=="instant":
                    triple.temporal_info.value=temporal_info[i].value
                elif temporal_info[i].type=="interval":
                    triple.temporal_info.start_time=temporal_info[i].start_time
                    triple.temporal_info.end_time=temporal_info[i].end_time
                elif temporal_info[i].type=="relative":
                    triple.temporal_info.offset=temporal_info[i].offset
                    triple.temporal_info.granularity=temporal_info[i].granularity
            subgraph.relations.reset()
            subgraph.relations.add_many(triples)
            memory.register_subgraph(subgraph)

        
    def extract_temporal_info(self,triples,text):
        triple_text="\n".join(f"id{i}: {triple.__str__()}" for i,triple in enumerate(triples))
        prompt=f"""
        Given the following extracted triples and their origin text, extract all relevant temporal information for all triples in the specified JSON format.

        Triples:
        {triple_text}

        Origin Text:
        {text}

        Please provide the extracted temporal information in JSON format as specified in the system prompt.
        """
        response=self.call_llm(prompt)
        try:
            response=self.parse_json(response)
            for i,item in enumerate(response):
                response[i]=TimeFormat.from_dict(item)
        except Exception as e:
            self.logger.error(f"Failed to parse temporal extraction response: {response}. Error: {e}")
            response=[]
        return response


if __name__=="__main__":
    head="COVID-19 pandemic"
    relation="initiated"
    tail="lockdowns measures"
    triple=KGTriple(head,relation,tail)