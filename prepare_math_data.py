import json
import os
from datasets import load_dataset
from tqdm import tqdm

def save_to_jsonl(dataset, output_path):
    print(f"Processing {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset):
            # 原始MATH数据集包含: problem, solution, level, type
            # data_utils.py 需要这些字段
            entry = {
                "problem": item['problem'],
                "solution": item['solution'],
                "level": item['level'],
                "type": item['type']
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():
    # 设置数据保存路径 (根据你的参数 --data_path)
    data_dir = "/home/haichao/TA/CoTVRL/data"  # <--- 请确认为你的实际路径
    os.makedirs(data_dir, exist_ok=True)

    print("Downloading MATH dataset from HuggingFace...")
    # 加载 hendrycks/competition_math
    dataset = load_dataset("hendrycks/competition_math")

    # 保存训练集
    train_path = os.path.join(data_dir, "math_train.jsonl")
    save_to_jsonl(dataset['train'], train_path)

    # 保存测试集
    test_path = os.path.join(data_dir, "math_test.jsonl")
    save_to_jsonl(dataset['test'], test_path)

    print("\n✅ Done! Data saved to:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")

if __name__ == "__main__":
    main()
