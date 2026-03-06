"""
在 DROP 数据集上评估模型：给定 passage 和 question，生成回答并计算词级别 F1。
Recall = 匹配上的词数 / 标准答案总词数
Precision = 匹配上的词数 / 模型生成文本总词数
F1 = 2 * P * R / (P + R)
"""
import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

# 连接到本地的 llama-server
client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key-required")

# 默认配置
# DEFAULT_DATASET = "examples/zeroth-order-opt/drop_validation.json"
DEFAULT_DATASET = "examples/zeroth-order-opt/drop_train.json"
QUESTIONS_NUMBER = 10  # 大于 0 时只取前 N 题；小于等于 0 时使用全部题目


def _normalize_word(w: str) -> str:
    """小写并做轻量归一化，便于匹配类似 '28-yard' 与 '28 yards'。"""
    w = w.lower().strip()
    # 去掉首尾所有非字母数字字符，避免 "**Chaz" 与 "Chaz" 无法匹配
    w = re.sub(r"^[^\w]+|[^\w]+$", "", w)
    if not w:
        return ""
    # 将常见的单位复数（如 yards -> yard, points -> point）简单还原为单数
    if len(w) > 3 and w.endswith("s"):
        w = w[:-1]
    return w.strip()


def _tokenize_to_words(text: str) -> set[str]:
    """将文本按空白与连字符等切分为词集合（归一化后）。"""
    if not text or not text.strip():
        return set()
    # 将连字符、斜杠等视为分隔符，避免 "28-yard" 与 "28 yards" 无法对齐
    normalized_text = re.sub(r"[-/]", " ", text.strip())
    words = re.split(r"\s+", normalized_text)
    return {_normalize_word(w) for w in words if _normalize_word(w)}


def compute_f1(gold: str, pred: str) -> tuple[float, float, float]:
    """
    词级别 F1。
    - Recall = 匹配上的词数 / 标准答案总词数
    - Precision = 匹配上的词数 / 模型生成文本总词数
    - 匹配：标准答案中的词若在模型输出中出现则计为匹配（集合交集）。
    """
    gold_words = _tokenize_to_words(gold)
    pred_words = _tokenize_to_words(pred)
    if not gold_words:
        recall = 1.0 if not pred.strip() else 0.0
        precision = 1.0 if not pred_words else 0.0
        f1 = 1.0 if (precision == 1.0 and recall == 1.0) else 0.0
        return precision, recall, f1
    if not pred_words:
        return 0.0, 0.0, 0.0
    matched = len(gold_words & pred_words)
    recall = matched / len(gold_words)
    precision = matched / len(pred_words)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on DROP with word-level F1.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Path to drop_train.json or drop_validation.json",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=QUESTIONS_NUMBER,
        help="Max number of questions (<=0 for all)",
    )
    args = parser.parse_args()

    dataset_path = Path(__file__).resolve().parent / args.dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.num > 0:
        dataset = dataset[: args.num]

    # 系统提示（每题独立，不保留历史）
    system_message = {"role": "system", "content": " "}

    total_time = 0.0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    n = len(dataset)

    print("开始 DROP 测试...")
    print("-" * 40)

    for i, item in enumerate(dataset, 1):
        passage = item.get("passage", "")
        question = item.get("question", "")
        answers_spans = item.get("answers_spans") or {}
        gold_raw = answers_spans.get("spans") if isinstance(answers_spans, dict) else None
        gold = (gold_raw if isinstance(gold_raw, str) else str(gold_raw or "")).strip()

        # 输入：passage + question
        user_content = f"{passage}\n\nQuestion: {question}"
        messages = [system_message, {"role": "user", "content": user_content}]

        print(f"\n[题目 {i}] {question[:80]}...")
        start_time = time.time()
        response = client.chat.completions.create(
            model="local-model",
            messages=messages,
            temperature=0,
        )
        elapsed = time.time() - start_time
        total_time += elapsed

        pred = (response.choices[0].message.content or "").strip()
        precision, recall, f1 = compute_f1(gold, pred)

        total_precision += precision
        total_recall += recall
        total_f1 += f1

        print(f"[标准答案] {gold}")
        print(f"[模型输出] {pred[:200]}{'...' if len(pred) > 200 else ''}")
        print(f"[P/R/F1] {precision:.4f} / {recall:.4f} / {f1:.4f}  [耗时] {elapsed:.2f}s")

    print("\n" + "=" * 40)
    print("DROP 测试统计：")
    print(f"题目数:     {n}")
    print(f"平均 Precision: {total_precision / n:.4f}" if n else "N/A")
    print(f"平均 Recall:    {total_recall / n:.4f}" if n else "N/A")
    print(f"平均 F1:       {total_f1 / n:.4f}" if n else "N/A")
    print(f"总耗时:     {total_time:.2f} 秒")
    print(f"平均每题:   {total_time / n:.2f} 秒" if n else "N/A")
    print("=" * 40)


if __name__ == "__main__":
    main()
