import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List
import math
class SapBertTester:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _get_device(self):
        """
        自动检测最佳计算设备：
        1. 遍历所有 NVIDIA 显卡。
        2. 获取每张卡的剩余显存。
        3. 自动选择剩余显存最大的一张卡。
        """
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"🔍 Found {num_gpus} GPUs available.")
            
            best_gpu_id = 0
            max_free_memory = 0
            
            # 遍历检查每张卡的显存状态
            for i in range(num_gpus):
                try:
                    # mem_get_info 返回 (free, total) 单位是字节
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    free_gb = free_mem / (1024 ** 3)
                    total_gb = total_mem / (1024 ** 3)
                    
                    print(f"   - GPU {i}: Free {free_gb:.2f} GB / Total {total_gb:.2f} GB")
                    
                    # 记录剩余显存最多的卡
                    if free_mem > max_free_memory:
                        max_free_memory = free_mem
                        best_gpu_id = i
                except Exception as e:
                    print(f"   - GPU {i}: Check failed ({e})")

            # 如果所有卡显存都很小（比如都小于1GB），可能需要警报，这里默认选最大的
            device_str = f"cuda:{best_gpu_id}"
            print(f"✅ Auto-selected Device: {device_str} (Has {max_free_memory / (1024**3):.2f} GB free)")
            return torch.device(device_str)

        elif torch.backends.mps.is_available():
            print("✅ Device: MPS (Mac M1/M2/M3)")
            return torch.device("mps")
        else:
            print("⚠️ Device: CPU (Slow)")
            return torch.device("cpu")

    def _load_model(self):
        """加载模型和分词器"""
        print(f"⏳ Loading BioBERT from: {self.model_path} ...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            
            # 将模型移动到计算出的 best device
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully on {self.device}!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            exit(1)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        print(f"🔄 Encoding {len(texts)} texts...")
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        # 关键：确保数据也移动到了选定的 device (例如 cuda:5)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return vec

    def get_main_pooling(self,texts:List[str])->np.ndarray:
        if not texts:
            return np.array([])

        print(f"🔄 Encoding {len(texts)} texts...")
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        # 关键：确保数据也移动到了选定的 device (例如 cuda:5)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            attention_mask = inputs['attention_mask']  # (batch_size, seq_len)

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            vec = (sum_embeddings / sum_mask).cpu().numpy()
        return vec
    
    def compute_similarity(self, vec_a, vec_b):
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)
        vec_a= self.l2_normalize(vec_a)
        vec_b= self.l2_normalize(vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        dot_product = np.dot(vec_a, vec_b)
        similarity = dot_product / (norm_a * norm_b)
        return similarity
    @staticmethod
    def l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0 or math.isnan(norm):
            return vec
        return vec / norm
# ==========================================
#              测试入口
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = "/home/nas2/path/models/SapBERT-from-PubMedBERT-fulltext" 

    tester = SapBertTester(MODEL_PATH)

    word1 = "Tirzepatide"
    word2 = "[UNK]"
    word3 = "diarrhea"
    word4 = "Semaglutide"

    sentence=word1+"[SEP]"+"Tirzepatide is a novel medication that combines dual action on the GIP and GLP-1 receptors, designed to improve blood sugar control and promote weight loss in individuals with type 2 diabetes."
    sentence2=word4+"[SEP]"+"Semaglutide is a GLP-1 receptor agonist that helps regulate blood sugar levels and support weight loss in people with type 2 diabetes, while also reducing the risk of cardiovascular events."
    sentence_unrelated="The mitochondrion is a double-membrane-bound organelle found in most eukaryotic organisms. Mitochondria generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy."
    sentence3 = "diabetes"+"[SEP]"+"Diabetes is a chronic condition where the body either cannot produce enough insulin or cannot effectively use the insulin it produces, leading to high blood sugar levels."
    all_texts = [word1, word2,word3,word4]
    
    embeddings = tester.get_main_pooling(all_texts)

    vec1 = embeddings[0]
    vec2 = embeddings[1]
    vec3 = embeddings[2]
    vec4 = embeddings[3]
    vec1_pooling=tester.get_main_pooling([word1])[0]
    sentence_pooling=tester.get_main_pooling([sentence])[0]
    sentence2_pooling=tester.get_main_pooling([sentence2])[0]
    sentence3_pooling=tester.get_main_pooling([sentence3])[0]
    sentence_unrelated_pooling=tester.get_main_pooling([sentence_unrelated])[0]
    print("\n" + "="*40)
    print("🧪 Similarity Test Results")
    print("="*40)

    sim_sentence2=tester.compute_similarity(sentence_pooling, sentence2_pooling)
    sim_unrelated=tester.compute_similarity(sentence_pooling, sentence_unrelated_pooling)
    # Tirzepatide+描述 vs 认识的 diabetes（统一采用 mean pooling 以保持可比性）
    diabetes_pooling = tester.get_main_pooling([sentence3])[0]
    sim_sentence_diabetes = tester.compute_similarity(sentence_pooling, diabetes_pooling)

    print(f"5️⃣  Pair: Sentence1 vs Sentence2")
    print(f"   Similarity: {sim_sentence2:.4f}")
    print("="*40 + "\n")

    print(f"6️⃣  Pair: Sentence1 vs Unrelated Sentence")
    print(f"   Similarity: {sim_unrelated:.4f}")
    print("="*40 + "\n")

    print(f"7️⃣  Pair: Sentence1 (Tirzepatide+desc) vs Diabetes+description")
    print(f"   Similarity: {sim_sentence_diabetes:.4f}")
    print("="*40 + "\n")

    
