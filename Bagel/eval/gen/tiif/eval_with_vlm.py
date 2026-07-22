import argparse
import os
import json
import glob
import openai
import time
import re
from tqdm import tqdm
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


raw_prompt = '''
You are tasked with conducting a careful examination of the provided image. Based on the content of the image, please answer the following yes or no questions:

Questions:
##YNQuestions##

Note that:
1. Each answer should be on a separate line, starting with "yes" or "no", followed by the reason.
2. The order of answers must correspond exactly to the order of the questions.
3. Each question must have only one answer.
4. Directly return the answers to each question, without any additional content.
5. Each answer must be on its own line!
6. Make sure the number of output answers equal to the number of questions!
'''

raw_prompt_1 = '''
You are tasked with conducting a careful examination of the image. Based on the content of the image, please answer the following yes or no questions:

Questions:
##YNQuestions##

Note that:
Each answer should be on a separate line, starting with "yes" or "no", followed by the reason.
The order of answers must correspond exactly to the order of the questions.
Each question must have only one answer. Output one answer if there is only one question.
Directly return the answers to each question, without any additional content.
Each answer must be on its own line!
Make sure the number of output answers equal to the number of questions!
'''

raw_prompt_2 = '''
You are tasked with carefully examining the provided image and answering the following yes or no questions:

Questions:
##YNQuestions##

Instructions:

1. Answer each question on a separate line, starting with "yes" or "no", followed by a brief reason.
2. Maintain the exact order of the questions in your answers.
3. Provide only one answer per question.
4. Return only the answers—no additional commentary.
5. Each answer must be on its own line.
6. Ensure the number of answers matches the number of questions.
'''

def load_jsonl_lines(jsonl_file):
    """读取 jsonl 文件，每行 parse 成 json 对象，返回列表"""
    lines = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lines.append(obj)
            except Exception as e:
                print(f"[Warning] Parse line error in {jsonl_file}: {e}")
    return lines


def generate_with_prompt(
    prompt,
    image_path,
    client,
    model='gpt-4o',
    temperature=1.0,
    max_tokens=None,
):
    import base64
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    messages = [
        {"role": "system", "content": "You are a professional image critic."},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                }
            ]
        }
    ]

    request = dict(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    completion = client.chat.completions.create(**request)
    
    return completion.choices[0].message.content

def format_questions_prompt(raw_prompt, questions):
    question_texts = [item.strip() for item in questions]
    formatted_questions = "\n".join(question_texts)
    prompt_template = random.choice([raw_prompt, raw_prompt_1, raw_prompt_2])
    formatted_prompt = prompt_template.replace("##YNQuestions##", formatted_questions)
    return formatted_prompt

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def find_image_by_idx(img_dir, idx):
    pattern = os.path.join(img_dir, f"{idx}.*")
    files = [f for f in glob.glob(pattern) if f.lower().endswith(('.png', '.jpg', '.jpeg', 'webp'))]
    if files:
        return files[0]   # 返回路径字符串
    else:
        raise FileNotFoundError(f"No image found for index {idx} in {img_dir}")

def load_manifest_image_map(manifest_file):
    image_map = {}
    for row in load_jsonl_lines(manifest_file):
        if row.get("benchmark") != "tiif":
            continue
        image_path = row["output_path"]
        path_parts = os.path.normpath(image_path).split(os.sep)
        try:
            tiif_index = path_parts.index("tiif")
            attribute = path_parts[tiif_index + 2]
            description = path_parts[tiif_index + 4]
        except (ValueError, IndexError) as error:
            raise ValueError(f"Unexpected TIIF manifest path: {image_path}") from error
        key = (attribute, description, row["prompt"])
        if key in image_map:
            raise ValueError(f"Duplicate TIIF manifest key: {key}")
        image_map[key] = image_path
    if not image_map:
        raise ValueError(f"No TIIF entries found in manifest: {manifest_file}")
    return image_map


def collect_tasks_from_manifest(
    jsonl_dir,
    generation_jsonl_dir,
    manifest_file,
    eval_model,
    output_dir,
):
    image_map = load_manifest_image_map(manifest_file)
    tasks = []
    for jsonl_file in glob.glob(os.path.join(jsonl_dir, "*.jsonl")):
        eval_lines = load_jsonl_lines(jsonl_file)
        generation_name = os.path.basename(jsonl_file).replace(
            "_eval_prompts.jsonl", "_prompts.jsonl"
        )
        generation_file = os.path.join(generation_jsonl_dir, generation_name)
        generation_lines = load_jsonl_lines(generation_file)
        if len(eval_lines) != len(generation_lines):
            raise ValueError(
                f"TIIF prompt count mismatch: {jsonl_file} has {len(eval_lines)}, "
                f"{generation_file} has {len(generation_lines)}"
            )

        for eval_line, generation_line in zip(eval_lines, generation_lines):
            attribute = eval_line["type"]
            if generation_line["type"] != attribute:
                raise ValueError(
                    f"TIIF attribute mismatch between {jsonl_file} and {generation_file}"
                )
            for description in ("long_description", "short_description"):
                key = (attribute, description, generation_line[description])
                if key not in image_map:
                    raise FileNotFoundError(f"No manifest image found for TIIF key: {key}")
                image_path = image_map[key]
                image_index = os.path.splitext(os.path.basename(image_path))[0]
                out_dir = os.path.join(
                    output_dir,
                    eval_model,
                    attribute,
                    "long" if description.startswith("long") else "short",
                )
                ensure_dir(out_dir)
                out_path = os.path.join(out_dir, f"{image_index}.json")
                if os.path.exists(out_path):
                    continue
                tasks.append({
                    "attribute": attribute,
                    "desc": description,
                    "jsonl_file": jsonl_file,
                    "line_idx": int(image_index),
                    "jsonl_line": eval_line,
                    "img_path": image_path,
                    "out_path": out_path,
                })
    return tasks


def collect_tasks(
    jsonl_dir,
    image_dir,
    eval_model,
    output_dir,
    sample_idx_file=None,
    postfix="",
    generation_jsonl_dir=None,
    manifest_file=None,
):
    if bool(generation_jsonl_dir) != bool(manifest_file):
        raise ValueError("generation_jsonl_dir and manifest_file must be provided together")
    if manifest_file:
        if sample_idx_file is not None or postfix:
            raise ValueError("Manifest-based TIIF lookup does not support sampling or postfix")
        return collect_tasks_from_manifest(
            jsonl_dir,
            generation_jsonl_dir,
            manifest_file,
            eval_model,
            output_dir,
        )

    tasks = []
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))
    if sample_idx_file is not None:
        with open(sample_idx_file, 'r') as f:
            sample_idx = json.load(f)

    lines = []
    for jsonl_file in jsonl_files:
        file_lines = load_jsonl_lines(jsonl_file)
        for line in file_lines:
            line['jsonl_file'] = jsonl_file
        lines += file_lines

    line_indices = list(range(len(lines)))

    # for jsonl_file in jsonl_files:
        # attribute = os.path.splitext(os.path.basename(jsonl_file))[0]
        # lines = load_jsonl_lines(jsonl_file)
        # attr_type = lines[0]['type']
    if sample_idx_file is not None:
        line_indices = sample_idx[attr_type]
        lines = [lines[idx] for idx in line_indices]
    else:
        line_indices = list(range(len(lines)))

    for desc in ['long_description', 'short_description']:

        for idx, line in zip(line_indices, lines):
            attr_type = line['type']
            img_dir = os.path.join(image_dir, attr_type+postfix, eval_model, desc)
            out_dir = os.path.join(output_dir, eval_model, attr_type, 'long' if desc.startswith('long') else 'short')
            ensure_dir(out_dir)

            try:
                img_path = find_image_by_idx(img_dir, idx)
                out_path = os.path.join(out_dir, f"{idx}.json")
                if os.path.exists(out_path):
                    continue
            except Exception as e:
                print(e)
                continue
            tasks.append({
                "attribute": attr_type,
                "desc": desc,
                "jsonl_file": line['jsonl_file'],
                "line_idx": idx,
                "jsonl_line": line,
                "img_path": img_path,
                "out_path": out_path
            })
    return tasks


class OutputFormatError(Exception):
    pass
def extract_yes_no(model_output, questions, allow_extra_answers=False):
    lines = [line.strip() for line in model_output.strip().split('\n') if line.strip()]
    preds = []
    for idx, line in enumerate(lines):
        m = re.match(r'^(yes|no)\b', line.strip(), flags=re.IGNORECASE)
        if m:
            preds.append(m.group(1).lower())
        else:
            continue
    if allow_extra_answers and len(preds) >= len(questions):
        return preds[:len(questions)]
    if len(preds) != len(questions):
        raise OutputFormatError(f"Preds count {len(preds)} != questions count {len(questions)}")
    return preds



def process_task(
    task,
    client,
    model,
    raw_prompt,
    temperature,
    max_tokens_per_question,
    allow_extra_answers,
):
    try:
        if os.path.exists(task["out_path"]):
            print(f"Existing eval results. Skip {task['out_path']}")
            return None

        item = task["jsonl_line"]
        questions = item.get("yn_question_list", [])
        gt_answers = item.get("yn_answer_list", [])
        prompt = format_questions_prompt(raw_prompt, questions)
        max_tokens = None
        if max_tokens_per_question > 0:
            max_tokens = max(128, len(questions) * max_tokens_per_question)
        model_output = generate_with_prompt(
            prompt,
            task["img_path"],
            client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print(model_output)
        model_pred = extract_yes_no(model_output, questions, allow_extra_answers)
        result = {
            "attribute": task["attribute"],
            "desc": task["desc"],
            "jsonl_file": os.path.basename(task["jsonl_file"]),
            "line_idx": task["line_idx"],
            "questions": questions,
            "gt_answers": gt_answers,
            "model_pred": model_pred,
            "model_output": model_output
        }
        with open(task["out_path"], "w", encoding="utf-8") as fout:
            json.dump(result, fout, ensure_ascii=False, indent=2)
        print(f"Saved: {task['out_path']}")
        return None
    except Exception as e:
        print(f"[Error] {task['img_path']} : {e}")
        return task


def main(args):
    client = openai.OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    tasks = collect_tasks(
        args.jsonl_dir,
        args.image_dir,
        args.eval_model,
        args.output_dir,
        args.sample_idx_file,
        args.postfix,
        args.generation_jsonl_dir,
        args.manifest_file,
    )
    print(f"Total tasks to process: {len(tasks)}")

    retry_tasks = []
    attempt = 0
    while tasks and (args.max_retries <= 0 or attempt < args.max_retries):
        attempt += 1
        retry_tasks.clear()
        for task in tqdm(tasks):
            task = process_task(
                task,
                client,
                args.model,
                raw_prompt,
                args.temperature,
                args.max_tokens_per_question,
                args.allow_extra_answers,
            )
            if task is not None:
                retry_tasks.append(task)
        if retry_tasks:
            print(f"Retrying {len(retry_tasks)} failed tasks...")
            time.sleep(5)
        tasks = retry_tasks.copy()
    if tasks:
        raise RuntimeError(f"Failed to score {len(tasks)} tasks after {attempt} attempts")


def main_parallel(args):
    client = openai.OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    tasks = collect_tasks(
        args.jsonl_dir,
        args.image_dir,
        args.eval_model,
        args.output_dir,
        args.sample_idx_file,
        args.postfix,
        args.generation_jsonl_dir,
        args.manifest_file,
    )
    print(f"Total tasks to process: {len(tasks)}")

    retry_tasks = []
    attempt = 0
    while tasks and (args.max_retries <= 0 or attempt < args.max_retries):
        attempt += 1
        retry_tasks.clear()
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    process_task,
                    task,
                    client,
                    args.model,
                    raw_prompt,
                    args.temperature,
                    args.max_tokens_per_question,
                    args.allow_extra_answers,
                ): task
                for task in tasks
            }
            for future in tqdm(as_completed(future_to_task), total=len(tasks)):
                result = future.result()
                if result is not None:
                    retry_tasks.append(result)

        if retry_tasks:
            print(f"Retrying {len(retry_tasks)} failed tasks...")
            time.sleep(2)
        tasks = retry_tasks.copy()
    if tasks:
        raise RuntimeError(f"Failed to score {len(tasks)} tasks after {attempt} attempts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_dir", type=str, required=True, help="Directory containing jsonl files")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--eval_model", type=str, required=True, help="name of the eval model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output json files")
    parser.add_argument("--api_key", type=str, default="sk-xxx", help="OpenAI API key")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="OpenAI API base url")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--sample_idx_file", type=str, default=None, help="File containing sample indices")
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--generation_jsonl_dir", type=str, default=None)
    parser.add_argument("--manifest_file", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--max_retries",
        type=int,
        default=0,
        help="Maximum attempts per failed task; 0 preserves the original unlimited retry behavior",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_tokens_per_question",
        type=int,
        default=0,
        help="Bound completion length; 0 leaves the API default unchanged",
    )
    parser.add_argument("--allow_extra_answers", action="store_true")

    args = parser.parse_args()
    random.seed(args.seed)
    # main(args)
    main_parallel(args)
