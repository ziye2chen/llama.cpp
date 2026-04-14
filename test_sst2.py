'''
SST-2 测试：
cmake --build . --config Debug --target llama-server
模型api： ./bin/llama-server -m /root/autodl-tmp/Qwen3-4B-GGUF/Qwen3-4B-Q5_K_M.gguf -c 2048 --port 8080
模型 + lora api： ./bin/llama-server -m /root/autodl-tmp/Qwen3-4B-GGUF/Qwen3-4B-Q5_K_M.gguf --lora /root/autodl-tmp/llama.cpp/Qwen3_lora_adapter.gguf -c 2048 --port 8080
(--save-lora-only 保存lora模型)
python test_sst2.py
python test_sst2.py --dataset examples/zeroth-order-opt/sst2_test.json --num 100 --temperature 0.0
'''

import argparse
import json
import re
import time
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key-required")

DEFAULT_DATASET = "examples/zeroth-order-opt/sst2_train.json"
DEFAULT_NUM = 100
INSTRUCTION = (
    "You are a strict sentiment classification expert. "
    "Output ONLY 'positive' or 'negative'. "
    "Do not output any other text, explanation, or punctuation."
)

def normalize_label(label: str) -> str:
    x = (label or "").strip().lower()
    if x in {"positive", "pos", "1"}:
        return "positive"
    if x in {"negative", "neg", "0"}:
        return "negative"
    return "unknown"

def extract_pred_label(text: str) -> str:
    x = (text or "").strip().lower()
    x = re.sub(r"[^\w\s-]", " ", x)
    pos = bool(re.search(r"\bpositive\b", x))
    neg = bool(re.search(r"\bnegative\b", x))
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    if x.strip() in {"positive", "negative"}:
        return x.strip()
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SST-2 with raw completion + accuracy.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("-n", "--num", type=int, default=DEFAULT_NUM, help="<=0 means all samples")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    dataset_path = Path(__file__).resolve().parent / args.dataset

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.num > 0:
        data = data[: args.num]

    n_total = len(data)
    n_scored = 0
    n_correct = 0
    total_time = 0.0

    print(f"Loading SST-2 dataset from: {dataset_path}")
    print(f"Total examples to evaluate: {n_total}")

    for i, item in enumerate(data, 1):
        sentence = (item.get("passage") or "").strip()
        answers_spans = item.get("answers_spans") or {}
        gold_raw = answers_spans.get("spans") if isinstance(answers_spans, dict) else ""
        gold = normalize_label(str(gold_raw))
        if gold == "unknown":
            gold = normalize_label(str(item.get("label", "")))

        prompt = (
            f"{INSTRUCTION}\n\n"
            f"Review: {sentence}\n"
            "Sentiment: "
        )

        print(f"\n[样本 {i}/{n_total}] {sentence[:80]}...")
        start_time = time.time()

        response = client.completions.create(
            model="local-model",
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=2,
        )

        elapsed = time.time() - start_time
        total_time += elapsed

        pred_raw = (response.choices[0].text or "").strip()
        pred = extract_pred_label(pred_raw)
        scored = gold in {"positive", "negative"}
        correct = scored and (pred == gold)
        if scored:
            n_scored += 1
            if correct:
                n_correct += 1
        print(f"[模型输出] {pred} (raw: {pred_raw})")
        print(f"[结果] {'✅ 正确' if correct else '❌ 错误'} | 计分={scored} | [耗时] {elapsed:.2f}s")

    acc = n_correct / n_scored if n_scored > 0 else 0.0
    print("\n" + "=" * 44)
    print("SST-2 测试统计：")

    print(f"总样本数:       {n_total}")
    print(f"可计分样本数:   {n_scored}")
    print(f"正确样本数:     {n_correct}")
    print(f"Accuracy:       {acc * 100:.2f}%" if n_scored else "Accuracy:       N/A (no labeled samples)")
    print(f"总耗时:         {total_time:.2f}s")
    print(f"平均每样本耗时: {total_time / n_total:.2f}s" if n_total > 0 else "平均每样本耗时: N/A")
    print("=" * 44)
if __name__ == "__main__":
    main()