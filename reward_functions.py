from typing import Optional
from difflib import SequenceMatcher
import re


# Format Enforcement: Ensures that the generation follows a specific format using <think> </think> <answer> </answer> tags for reasoning.
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format.

    Args:
        completions: List of conversations, where each conversation is a list of messages.
                     We need to extract the assistant's response text.
    """
    pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    rewards = []

    for completion in completions:
        # Extract text from conversation format
        if isinstance(completion, list) and len(completion) > 0:
            # Get the last message (assistant's response)
            content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        else:
            content = str(completion)

        # Check if the content matches the expected format
        match = re.search(pattern, content, re.DOTALL)

        # print(f"is match: {bool(match)}")
        rewards.append(1.0 if match else 0.0)

    print(f"format_reward: {rewards}, {len(rewards)}")

    return rewards


def inventory_mode_reasoning_score(generated_think_text, ground_truth_think_text) -> float:
    score = 0.0
    sequenceMatcher = SequenceMatcher(None, generated_think_text, ground_truth_think_text)
    score = sequenceMatcher.ratio()
    return score

def query_mode_reasoning_score(generated_think_text, ground_truth_think_text) -> float:
    score = 0.0
    if len(generated_think_text) > 15 and len(generated_think_text) < 160:
        score += 1.0
        
        sequenceMatcher = SequenceMatcher(None, generated_think_text, ground_truth_think_text)
        ratio = sequenceMatcher.ratio()
        score += ratio

    
    score = score / 2.0
    return score

def reasoning_reward(completions: list[list[dict[str, str]]], problem: list[str], solution: list[str], **kwargs) -> list[Optional[float]]:
    rewards = []

    for idx, (completion, problem, sol) in enumerate(zip(completions, problem, solution)):
        score = 0.0
        # print(f"problem: {problem}")
        # print(f"sol: {sol}")

        # Extract text from conversation format
        if isinstance(completion, list) and len(completion) > 0:
            # Get the last message (assistant's response)
            content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        else:
            content = str(completion)


        pattern = r"<think>\s*(.*?)\s*</think>"

        generated_think_match = re.search(pattern, content, re.DOTALL)
        ground_truth_think_match = re.search(pattern, sol, re.DOTALL)
        if generated_think_match and ground_truth_think_match:
            generated_think_text = generated_think_match.group(1).strip()
            ground_truth_think_text = ground_truth_think_match.group(1).strip()
            if problem == 'n':
                score = inventory_mode_reasoning_score(generated_think_text, ground_truth_think_text)
            else:
                score = query_mode_reasoning_score(generated_think_text, ground_truth_think_text)
        rewards.append(score)        

    print(f"reasoning_reward: {rewards}, {len(rewards)}")
    return rewards
    

class Item:
    def __init__(self, name: str, color: str, count: int):
        self.name = name
        self.color = color
        self.count = count

# Solution Accuracy: Verifies whether the solution to the problem is correct, comparing it to the solution column in the dataset.
def inventory_mode_accuracy_score(generated_answer_text, ground_truth_answer_text) -> float:
    score = 0.0
    sentence_pattern = r'^([^(),]+)\(([^()]+)\):(\d+)(,([^(),]+)\(([^()]+)\):(\d+))*$'

    match = re.search(sentence_pattern, generated_answer_text, re.DOTALL)
    

    if match:
        # 格式正確獎勵
        score += 0.2
        print(f"1 score: {score}")
        # Extract item-quantity pairs from generated answer
        # Pattern: item(color):count
        item_pattern = r'([^(),]+)\(([^()]+)\):(\d+)'

        generated_items = []
        for match in re.finditer(item_pattern, generated_answer_text):
            item_name = match.group(1).strip()
            color = match.group(2).strip()
            count = int(match.group(3))

            generated_items.append(Item(item_name, color, count))


        # Extract item-quantity pairs from ground truth
        ground_truth_items = []
        for match in re.finditer(item_pattern, ground_truth_answer_text):
            item_name = match.group(1).strip()
            color = match.group(2).strip()
            count = int(match.group(3))
            ground_truth_items.append(Item(item_name, color, count))


        total_ground_truth_items = len(ground_truth_items)
        total_generated_items = len(generated_items)

        if total_ground_truth_items == total_generated_items:

             # 格式分數 加上 item 總數匹配分數 佔比 0.5

            score += 0.3  # 完全項目種類數量正確獎勵

            print(f"2 score: {score}")


            total_item_score = 0.0

            for generated_item in generated_items:
                generated_item_name = generated_item.name
                max_item_score = 0.0 # 紀錄 generated_item 能夠在 ground_truth_item 中得到的最高分數
                for ground_truth_item in ground_truth_items:
                    item_score = 0.0
                    ground_truth_item_name = ground_truth_item.name
                    sequenceMatcher = SequenceMatcher(None, generated_item_name, ground_truth_item_name)
                    ratio = sequenceMatcher.ratio()
                    if ratio > 0.7:
                        item_score += 0.3
                        # 檢查 generated_item count 是否匹配 ground_truth_item count
                        # 根據數量誤差給分
                        error = abs(generated_item.count - ground_truth_item.count)
                        if error == 0:
                            item_score += 0.6
                        elif error == 1:
                            item_score += 0.2
                        elif error == 2:
                            item_score += 0.1

                        # 額外檢查 generated_item color 是否匹配 ground_truth_item color
                        if generated_item.color == ground_truth_item.color:
                            item_score += 0.1

                        # 如果這次的 item_score 比目前的 max_item_score 高，更新 max_item_score
                        # 最後會匹配到最高分數的 ground_truth_item
                        if item_score > max_item_score:
                            max_item_score = item_score


                
                total_item_score += max_item_score
            
            # 這部分是計算最終的 item 分數，並加到總分中 佔比 0.5
            final_item_score = total_item_score / total_ground_truth_items / 2.0
            score += final_item_score
    

    return score

def query_mode_accuracy_score(generated_answer_text, ground_truth_answer_text) -> float:
    score = 0.0
    # Convert the extracted answers to integers for comparison
    try:
        generated_answer_int = int(re.findall(r'\d+', generated_answer_text)[0])
        ground_truth_answer_int = int(re.findall(r'\d+', ground_truth_answer_text)[0])

        error = abs(generated_answer_int - ground_truth_answer_int)

        if error == 0:
            score = 1.0  # 完全正確
        elif error == 1:
            score = 0.5  # 差1個，給一半分數
        elif error == 2:
            score = 0.2  # 差2個，給一點分數
        elif error == 3:
            score = 0.1  # 差3個，給很少分數
        else:
            score = 0.0  # 差太多
    except (IndexError, ValueError):
        # Fallback to string comparison if conversion fails
        score = float(generated_answer_text.strip() == ground_truth_answer_text.strip())
        print("No numeric answer found, fallback to string comparison. Generated: {}, Ground Truth: {}".format(generated_answer_text, ground_truth_answer_text))

    return score



def accuracy_reward(completions: list[list[dict[str, str]]], problem: list[str], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text.

    Args:
        completions: List of conversations, where each conversation is a list of messages.
        solution: List of ground truth solutions.
    """
    rewards = []

    for idx, (completion, problem, sol) in enumerate(zip(completions, problem, solution)):
        score = 0.0
        # Extract text from conversation format
        if isinstance(completion, list) and len(completion) > 0:
            # Get the last message (assistant's response)
            content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        else:
            content = str(completion)


        pattern = r"<answer>\s*(.*?)\s*</answer>"
        generated_answer_match = re.search(pattern, content, re.DOTALL)
        ground_truth_answer_match = re.search(pattern, sol, re.DOTALL)
        if generated_answer_match and ground_truth_answer_match:
            generated_answer_text = generated_answer_match.group(1).strip()
            ground_truth_answer_text = ground_truth_answer_match.group(1).strip()

            print(f"Evaluating accuracy reward: {('='*50)}")
            print(f"problem: {problem}")
            print(f"generated_answer_text: {generated_answer_text}")
            print(f"ground_truth_answer_text: {ground_truth_answer_text}")
            print(f"{('='*50)}")

            if problem == 'n':
                score = inventory_mode_accuracy_score(generated_answer_text, ground_truth_answer_text)
                
            else:
                score = query_mode_accuracy_score(generated_answer_text, ground_truth_answer_text)

        rewards.append(score)

    print(f"accuracy_reward : {rewards}, {len(rewards)}")
    return rewards
