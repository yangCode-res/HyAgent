import json
from pathlib import Path

def load_test_data(limit: int = 5) -> list:
    data_path = Path(__file__).parent / "data" / "test_unseen.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:limit]


if __name__ == "__main__":
    # 加载前5条数据
    test_data = load_test_data(5)
    
    for i, item in enumerate(test_data):
        print(f"\n{'='*60}")
        print(f"Item {i + 1}:")
        print(f"Background: {item.get('background', '')[:100]}...")
        print(f"Hypothesis: {item.get('hypothesis', '')[:100]}...")

