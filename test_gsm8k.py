import time
import json
import re
from openai import OpenAI

# 连接到本地的 llama-server
client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key-required")

# 配置：大于 0 时只取前 N 题；小于等于 0 时使用全部题目
questions_number = 10
dataset_path = "examples/zeroth-order-opt/gsm8k_test.jsonl"

dataset = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        question = item.get("question")
        answer = item.get("answer")
        if question and answer:
            dataset.append({"question": question, "answer": answer})

if questions_number > 0:
    dataset = dataset[:questions_number]


def _extract_gold_answer(text: str) -> str | None:
    if not text:
        return None
    marker = "####"
    if marker in text:
        return text.split(marker)[-1].strip()
    return text.strip()


def _extract_pred_answer(text: str) -> str | None:
    if not text:
        return None
    # 优先从 \boxed{} 里提取数字
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed_match:
        inner = boxed_match.group(1).replace(",", "")
        nums = re.findall(r"-?\d+(?:\.\d+)?", inner)
        if nums:
            return nums[-1]
    # 没有 \boxed{} 或里面无数时：从全文取最后一个出现的数字
    cleaned = text.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not nums:
        return None
    return nums[-1]


def _normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    t = text.strip()
    if t.startswith("$"):
        t = t[1:]
    t = t.replace(",", "")
    if t.endswith("."):
        t = t[:-1]
    return t.strip()

# 初始化系统提示词（每题独立提问，不保留历史）
# system_message = {"role": "system", "content": "你是一个严谨的数学助手，请直接给出解答过程和最终答案。"}
system_message = {"role": "system", "content": " "}

total_time = 0.0
total_questions = len(dataset)
correct_count = 0

print("开始测试...")
print("-" * 30)

for i, item in enumerate(dataset, 1):
    q = item["question"]
    gold_answer_raw = item["answer"]
    print(f"\n[问题 {i}] {q}")
    # 每题单独构造消息，不带前序题目和回答
    messages = [
        system_message,
        {"role": "user", "content": q}
    ]
    
    # 记录开始时间
    start_time = time.time()
    
    # 发送请求
    response = client.chat.completions.create(
        model="local-model", # 这里填什么都可以，server 会自动使用当前加载的 GGUF 模型
        messages=messages,
        temperature=0 # 数学题建议降低 temperature 以保证输出的确定性
    )
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_time += elapsed_time
    
    answer = response.choices[0].message.content or ""
    print(f"[回答]\n{answer.strip()}")
    print(f"[耗时] {elapsed_time:.2f} 秒")

    # 计算当前题是否正确：
    #   - 标准答案按 gsm8k 约定，从 '####' 后取最终答案
    #   - 模型答案按你要求，从回答中最后出现的数字作为最终答案
    gold_final = _normalize_answer(_extract_gold_answer(gold_answer_raw))
    pred_final = _normalize_answer(_extract_pred_answer(answer))
    if gold_final is not None and pred_final is not None and gold_final == pred_final:
        correct_count += 1
        print("[判定] 正确")
    else:
        print(f"[判定] 错误（标准答案: {gold_final}）")
    
print("\n" + "=" * 30)
print("测试完成统计：")
print(f"回答总题数: {total_questions}")
print(f"答对题数:   {correct_count}")
accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0.0
print(f"正确率:     {accuracy:.2f}%")
print(f"总计耗时:   {total_time:.2f} 秒")
avg_time = (total_time / total_questions) if total_questions > 0 else 0.0
print(f"平均每题耗时: {avg_time:.2f} 秒")
print("=" * 30)