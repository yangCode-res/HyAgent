import json
import re
from pathlib import Path
from typing import Dict, List
from collections import Counter
from pipeline.index import Pipeline


class Benchmark:
    def __init__(self, limit: int = 5):
        self.limit = limit
        self.test_data = self.load_test_data(limit=self.limit)

    def load_test_data(self, limit: int = 5) -> list:
        data_path = Path(__file__).parent / "data" / "test_unseen.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[:limit]

    def runOneTestData(self, item: dict):
        overlap_scores = []

        pipeline = Pipeline(user_query=item.get('background', ''))
        pipeline.run()
        for i in pipeline.scores:
            overlap_scores.append(self.compute_overlap(i.get('hypothesis', ''), item.get('hypothesis', '')))
        
        # 获取各个指标的最大值
        max_scores = self.get_max_scores(overlap_scores)
        return pipeline.scores, max_scores

    def get_max_scores(self, overlap_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """
        获取 overlap_scores 中各个指标的最大值
        
        Args:
            overlap_scores: 多个 overlap 计算结果的列表
            
        Returns:
            Dict: 每个指标的最大值
        """
        if not overlap_scores:
            return {"jaccard": 0, "precision": 0, "recall": 0, "f1": 0, "bleu_1": 0, "rouge_l": 0}
        
        score_keys = ["jaccard", "precision", "recall", "f1", "bleu_1", "rouge_l"]
        max_scores = {}
        
        for key in score_keys:
            max_scores[key] = max(score[key] for score in overlap_scores)
        
        return max_scores

    def run(self, test_data: list):
        for item in test_data:
            scores = self.runOneTestData(item)
            print(scores)

    # ==================== 文本重叠度计算 ====================

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """简单分词：转小写，去除标点，按空格分割"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # 去除标点
        return text.split()

    def compute_overlap(self, generated: str, reference: str) -> Dict[str, float]:
        """
        计算生成文本与参考文本的词重叠度
        
        Args:
            generated: 生成的假设文本
            reference: 原始参考假设文本
            
        Returns:
            Dict: 包含多个重叠度指标
                - jaccard: Jaccard 相似度 (交集/并集)
                - precision: 精确率 (生成词中有多少在参考中)
                - recall: 召回率 (参考词中有多少被生成出来)
                - f1: F1 分数
                - bleu_1: BLEU-1 (unigram)
                - rouge_l: ROUGE-L (基于 LCS)
        """
        gen_tokens = self.tokenize(generated)
        ref_tokens = self.tokenize(reference)

        if not gen_tokens or not ref_tokens:
            return {"jaccard": 0, "precision": 0, "recall": 0, "f1": 0, "bleu_1": 0, "rouge_l": 0}

        gen_set = set(gen_tokens)
        ref_set = set(ref_tokens)

        # Jaccard 相似度
        intersection = gen_set & ref_set
        union = gen_set | ref_set
        jaccard = len(intersection) / len(union) if union else 0

        # 精确率和召回率
        precision = len(intersection) / len(gen_set) if gen_set else 0
        recall = len(intersection) / len(ref_set) if ref_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # BLEU-1 (unigram precision with clipping)
        bleu_1 = self._compute_bleu_1(gen_tokens, ref_tokens)

        # ROUGE-L (基于最长公共子序列)
        rouge_l = self._compute_rouge_l(gen_tokens, ref_tokens)

        return {
            "jaccard": round(jaccard, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "bleu_1": round(bleu_1, 4),
            "rouge_l": round(rouge_l, 4)
        }

    def _compute_bleu_1(self, gen_tokens: List[str], ref_tokens: List[str]) -> float:
        """计算 BLEU-1 分数 (unigram precision with clipping)"""
        gen_counter = Counter(gen_tokens)
        ref_counter = Counter(ref_tokens)

        # Clipped count: 每个词最多算参考中出现的次数
        clipped_count = 0
        for word, count in gen_counter.items():
            clipped_count += min(count, ref_counter.get(word, 0))

        return clipped_count / len(gen_tokens) if gen_tokens else 0

    def _compute_rouge_l(self, gen_tokens: List[str], ref_tokens: List[str]) -> float:
        """计算 ROUGE-L 分数 (基于最长公共子序列 LCS)"""
        lcs_len = self._lcs_length(gen_tokens, ref_tokens)

        if not gen_tokens or not ref_tokens:
            return 0

        precision = lcs_len / len(gen_tokens)
        recall = lcs_len / len(ref_tokens)

        if precision + recall == 0:
            return 0

        # F1-score
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """计算两个序列的最长公共子序列长度 (动态规划)"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]


# 测试代码
if __name__ == "__main__":
    benchmark = Benchmark(limit=1)
    
    # 测试重叠度计算
    generated = "Different subgroups of depressive symptoms exist among Korean police officers. Drinking behaviors may contribute to depression."
    reference = "Different subgroups of depressive symptoms exist among Korean police officers. Drinking behaviors may contribute to the at-risk subgroup of depressive symptoms."
    
    overlap_scores = benchmark.compute_overlap(generated, reference)
    print("Overlap Scores:")
    for key, value in overlap_scores.items():
        print(f"  {key}: {value}")
