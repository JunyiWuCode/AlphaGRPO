# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python
"""Prompt Decomposition Pipeline

Decomposes text-to-image prompts into semantic and quality evaluation questions
using DSG (Dependency Scene Graph) and LLM-based validation.

Usage:
    python decompose.py --input prompts.txt --output output.jsonl [--workers 32]

Environment variables:
    OPENAI_API_URL             - API base URL
    ARK_API_KEY                - API key (required)
    MODEL_NAME                 - Model name (default: gpt-4.1-2025-04-14)
    AZURE_OPENAI_API_VERSION   - API version for Azure (default: 2024-03-01-preview)
"""

import argparse
import json
import os
import re
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import openai
import tqdm
from dsg.openai_utils import openai_completion
from dsg.query_utils import generate_dsg
from dsg.parse_utils import parse_question_output


# ============================================================
# Semantic Categories & Helper
# ============================================================

def type_list_convert_to_markdown(categories):
    md_lines = []
    for item in categories:
        line = f"- **{item['type']}**: {item['description']}"
        md_lines.append(line)
    return "\n".join(md_lines)


# 1. Semantic Categories (Content Alignment)
semantic_categories = [
    {
        "type": "Style",
        "description": "Artistic medium (sketch, oil, photo), visual genre (anime, cyberpunk), or image format."
    },
    {
        "type": "Environment",
        "description": "Background setting, weather, time of day, lighting atmosphere, or location context."
    },
    {
        "type": "Viewpoint",
        "description": "Camera parameters: angle (top-down), shot size (close-up), lens type (fisheye), or framing."
    },
    {
        "type": "Existence",
        "description": "Presence or visibility of specific subjects/objects (Binary Yes/No), excluding quantity."
    },
    {
        "type": "Count",
        "description": "Numerical quantity or multiplicity of specific objects (e.g., 'three', 'a pair', 'single')."
    },
    {
        "type": "Attribute",
        "description": "Static visual properties of objects: color, material, shape, texture, size, or attire."
    },
    {
        "type": "Action",
        "description": "Dynamic movements (running), physical activities, body poses (sitting), or active states."
    },
    {
        "type": "Spatial",
        "description": "Relative positioning (left/right, behind), depth relations, or interactions like holding/wearing."
    },
    {
        "type": "Text",
        "description": "Presence, spelling, or visibility of specific written words, characters, signage, or logos."
    },
    {
        "type": "Negative",
        "description": "Explicit absence of elements or verification that something is NOT present."
    }
]


# ============================================================
# System Prompts
# ============================================================

tag_semantic_system_prompt = '''You are an expert Semantic Classifier for an image generation evaluation pipeline.
Your task is to analyze an **Evaluation Question** relative to an **Image Prompt** and categorize the question into exactly one semantic aspect.

**Category Definitions:**
''' + type_list_convert_to_markdown(semantic_categories) + \
'''
**Instructions:**
1. Analyze what specific visual element or relationship the Evaluation Question is checking.
2. Select the **most appropriate category** from the list above.
3. The "aspect_tag" MUST be an exact string match to one of the category names provided (e.g., "Style", "Count", "Action").

**Input Format:**
{
    "prompt": "The image prompt."
    "question": "The evaluation question to tag.",
}

**Output Format:**
Return a single valid JSON object containing no markdown formatting:
{
    "reason": "Brief analysis of why the question fits this category based on the image prompt.",
    "aspect_tag": "The selected category name"
}'''


verified_system_prompt = '''You are an expert Quality Assurance Validator for an image generation pipeline.
Your task is to verify whether a given **Evaluation Question** is strictly necessary and valid based on the **Image Prompt**.

**CRITICAL RULES FOR VERIFICATION:**

1.  **Ambiguity Handling (The "Or" Rule):**
    * If a prompt is a noun that can also imply an action (e.g., "Jump Rope", "Basketball", "Piano"), the image generator is allowed to depict EITHER just the object OR the action.
    * Therefore, a question asking strictly for the action (e.g., "Is someone playing the piano?") is **INVALID** because the prompt did not strictly require a person.
    * *Correct logic:* The question should be "Is a jump rope visible?"

2.  **Strict Necessity:**
    * Only approve questions about elements that MUST be present for the prompt to be considered "followed".
    * If the element is optional, the question is **INVALID**.

3.  **No Hallucinations:**
    * The question cannot ask for specific colors, styles, or backgrounds not mentioned in the prompt.

**Input Format:**
{
    "prompt": "The image prompt to validate against."
    "question": "The evaluation question to verify.",
}

**Output Format:**
Return a JSON object:
{
  "reason": "Explain why based on ambiguity or necessity rules."
  "is_valid": boolean,
}'''


quality_question_generation_system_prompt = '''# Role
You are an expert Visual Quality Assurance (VQA) Generator. Your goal is to expand a "Semantic Scene Graph" into a "Quality Evaluation Tree".

# Task
Given a user's `prompt` and a list of `semantic_questions`, generate detailed **Visual Quality Questions** and map them to their corresponding semantic node.

# 1. Critical Rule: Contextual Fidelity
You must distinguish between **Explicit Constraints** (in the prompt) and **General Quality** (universal laws).
- **No Hallucinated Constraints:** Do not enforce attributes not mentioned in the prompt. For example, if the prompt is "a car", do not ask "Is the car red?". Instead, ask "Does the car have a realistic paint texture?".
- **Universal Quality:** Always check for general visual fidelity, such as structural integrity, realistic textures (if photorealism is implied), and lack of artifacts.

# 2. Critical Rule: Prompt Priority
The User Prompt is the absolute truth. If the prompt specifies something physically impossible or abstract, that specific abnormality is the "standard of quality".
- If the prompt says "a floating stone", do not ask "Is the stone on the ground?". Instead, ask "Is the stone clearly floating?".

# 3. Phrasing Rule (Positive Only)
- All questions must be phrased so that the answer **"Yes"** indicates **High Quality** or **Good Alignment**.
    - Bad: "Is the image blurry?" (Yes = Bad)
    - Good: "Is the image sharp and free from blur?" (Yes = Good)
- Don't include two aspects in one question. For example, the clock's structure and text readability should be placed in two separate questions to make it clearly distinguishable.

# 4. Allowed Aspects
Classify each question into exactly one of these categories:
- **Geometry**: Structural integrity, perspective logic, and shape correctness for inorganic objects (buildings, cars).
- **Anatomy**: Biological correctness of humans/animals: limb proportions, hands, faces, and skeletal logic.
- **Texture**: Realism of surface materials, fine details, resolution, noise levels, and material fidelity.
- **Coherence**: Object integrity: no unintended melting, fusion, detachment, or illogical blending.
- **Lighting**: Consistency of illumination, shadow direction, light source logic, and reflections.
- **Physics**: Physical plausibility: gravity (ground contact/floating), motion blur, and fluid dynamics.
- **Legibility**: Readability of text: spelling accuracy, clear glyphs, and lack of gibberish.
- **Aesthetics**: Overall visual appeal, adherence to art style, and freedom from digital artifacts/glitches.

# Output Format
Output a strictly valid JSON dictionary where:
- The keys are the exact strings from the input `semantic_questions`.
- The values are lists of objects containing `question` and `aspect`.
- If a semantic question is purely relational (e.g., "Is X left of Y?") and does not require a specific visual quality check, you could omit it.

# Example
Input:
{
  "prompt": "a photo of a larger Traffic light on the above and a smaller Invertebrate on the below",
  "semantic_questions": [
    "Is there a traffic light?",
    "Is there an invertebrate?",
    "Is the traffic light larger?",
    "Is the invertebrate smaller?",
    "Is the traffic light above the invertebrate?",
    "Is the invertebrate below the traffic light?"
  ]
}

Output:
[
    {
      "question": "Is the traffic light structure geometrically correct with clearly defined circular signal lamps?",
      "aspect": "Geometry"
    },
    {
      "question": "Does the traffic light surface have a realistic metallic or plastic material finish?",
      "aspect": "Texture"
    },
    {
      "question": "Does the invertebrate possess anatomically plausible features (such as segments or limbs)?" ,
      "aspect": "Anatomy"
    },
    {
      "question": "Is the invertebrate distinct and separated from the background without blending artifacts?",
      "aspect": "Coherence"
    },
    {
      "question": "Are both objects lit consistently, suggesting they exist in the same physical space?",
      "aspect": "Lighting"
    },
    {
      "question": "Is the resolution of the larger traffic light sufficient to maintain sharp details at its scale?",
      "aspect": "Texture"
    }
]'''


# ============================================================
# API Configuration & Chat
# ============================================================

base_url = os.environ.get("OPENAI_API_URL")
ak = os.environ.get("ARK_API_KEY")
model_name = os.environ.get("MODEL_NAME", "gpt-4.1-2025-04-14")
model_tag = 'gpt-4.1' if 'gpt-4.1' in model_name else model_name
api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-03-01-preview")


def chat(prompt, system_prompt=None, is_output_json=True,
         temperature=0.6, top_p=0.9, max_tokens=4094, **kwargs):
    """Send a chat completion request with automatic retry."""
    for _ in range(10):
        try:
            if 'gpt' in model_tag:
                client = openai.AzureOpenAI(
                    azure_endpoint=base_url,
                    api_version=api_version,
                    api_key=ak,
                )
            else:
                client = openai.OpenAI(
                    base_url=base_url,
                    api_key=ak,
                )

            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_headers={"X-TT-LOGID": ""},
                timeout=1800000,
                **kwargs
            )
            response = completion.choices[0].message.content
            if is_output_json:
                response = re.sub(r"^```json|```$", "", response, flags=re.MULTILINE).strip()
                response = json.loads(response)
            return response
        except Exception as e:
            print(e)
            time.sleep(1)
    return None


# ============================================================
# Core Pipeline
# ============================================================

def decompose_prompt(prompt, verbose=False):
    """
    Decompose a single prompt into semantic and quality evaluation questions.

    Returns dict with: prompt, semantic_questions, quality_questions
    Or None on failure after retries.
    """
    for _ in range(5):
        try:
            # Step 1: Generate DSG semantic question candidates
            if verbose:
                print(f"Step 1: Generating DSG & Initial Semantic Candidates for prompt: {prompt}")
            id2prompts = {0: {'input': prompt}}

            id2tuple, id2question, id2dep = generate_dsg(
                id2prompts,
                generate_fn=partial(openai_completion, model=model_name),
                N_parallel_workers=1,
                verbose=verbose
            )

            raw_questions_dict = parse_question_output(id2question[0]['output'])
            candidate_questions = list(raw_questions_dict.values())

            # Step 2: Validate & tag each semantic question (parallel)
            if verbose:
                print(f"Step 2: Validating {len(candidate_questions)} semantic questions...")
            pending_tasks = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                for question in candidate_questions:
                    input_str = json.dumps({
                        'prompt': prompt,
                        'question': question
                    }, ensure_ascii=False)

                    future_val = executor.submit(
                        partial(chat, temperature=0.3),
                        prompt=input_str,
                        system_prompt=verified_system_prompt,
                        is_output_json=True
                    )
                    future_tag = executor.submit(
                        partial(chat, temperature=0.3),
                        prompt=input_str,
                        system_prompt=tag_semantic_system_prompt,
                        is_output_json=True
                    )
                    pending_tasks.append((question, future_val, future_tag))

            semantic_questions = []
            valid_question_texts = []
            for question, f_val, f_tag in pending_tasks:
                validation_response = f_val.result()
                tag_response = f_tag.result()

                is_valid = validation_response['is_valid']
                aspect_tag = tag_response['aspect_tag']

                semantic_questions.append({
                    'question_type': 'semantic',
                    'question': question,
                    'is_valid': is_valid,
                    'tag': aspect_tag
                })

                if is_valid:
                    valid_question_texts.append(question)

            if not all(sq['is_valid'] for sq in semantic_questions):
                raise RuntimeError("Has invalid semantic questions")

            # Step 3: Generate quality questions
            if verbose:
                print(f"Step 3: Generating Quality Questions for prompt: {prompt}")
            quality_gen_input = {
                "prompt": prompt,
                "semantic_questions": valid_question_texts
            }

            quality_response = chat(
                prompt=json.dumps(quality_gen_input, ensure_ascii=False),
                system_prompt=quality_question_generation_system_prompt,
                is_output_json=True, max_tokens=4096,
            )

            quality_questions = []
            for qs in quality_response:
                quality_questions.append({
                    'question_type': 'quality',
                    'question': qs['question'],
                    'is_valid': True,
                    'tag': qs['aspect']
                })

            return {
                "prompt": prompt,
                "semantic_questions": semantic_questions,
                "quality_questions": quality_questions,
            }
        except Exception as e:
            if verbose:
                print(f"Retry due to: {e}")
            else:
                print(e)
            continue

    return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Decompose T2I prompts into semantic + quality evaluation questions"
    )
    parser.add_argument("--input", required=True, help="Input txt file (one prompt per line)")
    parser.add_argument("--output", required=True, help="Output jsonl file")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.input) as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.input}")

    # Resume support: skip already-done lines
    done_indices = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                item = json.loads(line)
                if "original_index" in item:
                    done_indices.add(item["original_index"])
        print(f"Resuming: {len(done_indices)} already completed")

    todo = [(i, p) for i, p in enumerate(prompts) if i not in done_indices]
    if not todo:
        print("All prompts already processed.")
        return

    print(f"Processing {len(todo)} prompts with {args.workers} workers...")

    fn = partial(decompose_prompt, verbose=args.verbose)
    indices, todo_prompts = zip(*todo)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in tqdm.tqdm(
            executor.map(fn, todo_prompts),
            total=len(todo_prompts),
            desc="Decomposing",
        ):
            results.append(result)

    count = 0
    with open(args.output, "a") as f:
        for idx, result in zip(indices, results):
            if result is not None:
                result["original_index"] = idx
                f.write(json.dumps(result) + "\n")
                count += 1

    print(f"Wrote {count}/{len(todo)} results to {args.output}")


if __name__ == "__main__":
    main()