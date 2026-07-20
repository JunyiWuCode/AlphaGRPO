import argparse
import os
import os.path as osp
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="DPG-Bench evaluation.")
    parser.add_argument(
        "--image-root-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default='./eval/gen/dpg/dpg_bench.csv',
    )
    parser.add_argument(
        "--res-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pic-num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--vqa-model",
        type=str,
        default='mplug',
    )

    args = parser.parse_args()
    return args


class MPlugVQAPreprocessor:
    def __init__(self, model_dir, tokenizer_max_length=25):
        from modelscope.models.multi_modal.mplug import CONFIG_NAME, MPlugConfig
        from torchvision import transforms
        from transformers import BertTokenizer

        config = MPlugConfig.from_yaml_file(osp.join(model_dir, CONFIG_NAME))
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.tokenizer_max_length = tokenizer_max_length
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (config.image_res, config.image_res),
                interpolation=Image.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def __call__(self, image, question):
        image = self.image_transform(image.convert('RGB')).unsqueeze(0)
        question = self.tokenizer(
            question.lower(),
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors='pt',
        )
        return {'image': image, 'question': question}


class MPLUG:
    def __init__(self, ckpt='damo/mplug_visual-question-answering_coco_large_en', device='gpu'):
        import torch
        from modelscope.hub.snapshot_download import snapshot_download
        from modelscope.models.multi_modal.mplug_for_all_tasks import MPlugForAllTasks
        from modelscope.utils.constant import Tasks

        self.torch = torch
        self.device = torch.device(device)
        model_dir = snapshot_download(ckpt)
        self.preprocessor = MPlugVQAPreprocessor(model_dir)
        self.model = MPlugForAllTasks(
            model_dir, task=Tasks.visual_question_answering
        ).to(self.device)
        self.model.eval()

    def vqa(self, image, question):
        inputs = self.preprocessor(image, question)
        inputs = {
            key: value.to(self.device) if hasattr(value, 'to') else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            result = self.model(inputs)
        return result['text']

def prepare_dpg_data(args):
    previous_id = ''
    current_id = ''
    question_dict = dict()
    category_count = defaultdict(int)
    # 'item_id', 'text', 'keywords', 'proposition_id', 'dependency', 'category_broad', 'category_detailed', 'tuple', 'question_natural_language'
    data = pd.read_csv(args.csv)
    for i, line in data.iterrows():
        current_id = line.item_id
        qid = int(line.proposition_id)
        dependency_list_str = str(line.dependency).split(',')
        dependency_list_int = []
        for d in dependency_list_str:
            d_int = int(d.strip())
            dependency_list_int.append(d_int)

        if current_id == previous_id:
            question_dict[current_id]['qid2tuple'][qid] = line.tuple
            question_dict[current_id]['qid2dependency'][qid] = dependency_list_int
            question_dict[current_id]['qid2question'][qid] = line.question_natural_language
        else:
            question_dict[current_id] = dict(
                qid2tuple={qid: line.tuple},
                qid2dependency={qid: dependency_list_int},
                qid2question={qid: line.question_natural_language})
        
        category = line.question_natural_language.split('(')[0].strip()
        category_count[category] += 1
        
        previous_id = current_id

    return question_dict

def crop_image(input_image, crop_tuple=None):
    if crop_tuple is None:
        return input_image

    cropped_image = input_image.crop((crop_tuple[0], crop_tuple[1], crop_tuple[2], crop_tuple[3]))

    return cropped_image

# def compute_dpg_one_sample(args, question_dict, image_path, vqa_model, resolution):
#     generated_image = Image.open(image_path)
#     crop_tuples_list = [
#         (0,0,resolution,resolution),
#         (resolution, 0, resolution*2, resolution),
#         (0, resolution, resolution, resolution*2),
#         (resolution, resolution, resolution*2, resolution*2),
#     ]

#     crop_tuples = crop_tuples_list[:args.pic_num]
#     key = osp.basename(image_path).split('.')[0]
#     value = question_dict.get(key, None)
#     qid2tuple = value['qid2tuple']
#     qid2question = value['qid2question']
#     qid2dependency = value['qid2dependency']

#     qid2answer = dict()
#     qid2scores = dict()
#     qid2validity = dict()

#     scores = []
#     for crop_tuple in crop_tuples:
#         cropped_image = crop_image(generated_image, crop_tuple)
#         for id, question in qid2question.items():
#             answer = vqa_model.vqa(cropped_image, question)
#             qid2answer[id] = answer
#             qid2scores[id] = float(answer == 'yes')
#             with open(args.res_path.replace('.txt', '_detail.txt'), 'a') as f:
#                 f.write(image_path + ', ' + str(crop_tuple) + ', ' + question + ', ' + answer + '\n')
#         qid2scores_orig = qid2scores.copy()

#         for id, parent_ids in qid2dependency.items():
#             # zero-out scores if parent questions are answered 'no'
#             any_parent_answered_no = False
#             for parent_id in parent_ids:
#                 if parent_id == 0:
#                     continue
#                 if qid2scores[parent_id] == 0:
#                     any_parent_answered_no = True
#                     break
#             if any_parent_answered_no:
#                 qid2scores[id] = 0
#                 qid2validity[id] = False
#             else:
#                 qid2validity[id] = True

#         score = sum(qid2scores.values()) / len(qid2scores)
#         scores.append(score)
#     average_score = sum(scores) / len(scores)
#     with open(args.res_path, 'a') as f:
#         f.write(image_path + ', ' + ', '.join(str(i) for i in scores) + ', ' + str(average_score) + '\n')
  
#     return average_score, qid2tuple, qid2scores_orig


def compute_dpg_one_sample(args, question_dict, image_path, vqa_model, resolution):
    """
    修改点：不再写入文件，而是返回 formatted strings (result_line, detail_lines_list)
    """
    generated_image = Image.open(image_path)
    crop_tuples_list = [
        (0,0,resolution,resolution),
        (resolution, 0, resolution*2, resolution),
        (0, resolution, resolution, resolution*2),
        (resolution, resolution, resolution*2, resolution*2),
    ]

    crop_tuples = crop_tuples_list[:args.pic_num]
    key = osp.basename(image_path).split('.')[0]
    value = question_dict.get(key, None)
    
    # 简单的错误保护
    if value is None:
        return None, None, None, None, None

    qid2tuple = value['qid2tuple']
    qid2question = value['qid2question']
    qid2dependency = value['qid2dependency']

    qid2scores = dict()
    qid2validity = dict() # 虽然在这个函数内没用到返回值，但逻辑保留
    
    # 用列表缓存 detail 信息
    sample_detail_lines = []
    
    scores = []
    for crop_tuple in crop_tuples:
        cropped_image = crop_image(generated_image, crop_tuple)
        
        # 临时 dict 用于当前 crop 的计算
        current_crop_qid2scores = {} 
        
        for id, question in qid2question.items():
            answer = vqa_model.vqa(cropped_image, question)
            score_val = float(answer == 'yes')
            current_crop_qid2scores[id] = score_val
            
            # 【核心修改】：缓存 detail 字符串
            detail_str = f"{image_path}, {str(crop_tuple)}, {question}, {answer}\n"
            sample_detail_lines.append(detail_str)
            
        qid2scores.update(current_crop_qid2scores) # 更新到总字典（注意：原逻辑似乎是覆盖式的，这里保持原逻辑）
        qid2scores_orig = current_crop_qid2scores.copy() # 用于外部统计

        # Dependency check logic
        for id, parent_ids in qid2dependency.items():
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == 0: continue
                if qid2scores[parent_id] == 0:
                    any_parent_answered_no = True
                    break
            if any_parent_answered_no:
                qid2scores[id] = 0
                qid2validity[id] = False
            else:
                qid2validity[id] = True

        score = sum(qid2scores.values()) / len(qid2scores)
        scores.append(score)
        
    average_score = sum(scores) / len(scores)
    
    # 【核心修改】：生成 result 字符串
    sample_res_line = f"{image_path}, {', '.join(str(i) for i in scores)}, {str(average_score)}\n"
  
    return average_score, qid2tuple, qid2scores_orig, sample_res_line, sample_detail_lines

def main():
    from accelerate import Accelerator
    from accelerate.utils import gather_object

    args = parse_args()

    accelerator = Accelerator()

    question_dict = prepare_dpg_data(args)

    timestamp = time.time()
    time_array = time.localtime(timestamp)
    time_style = time.strftime("%Y%m%d-%H%M%S", time_array)
    if args.res_path is None:
        args.res_path = osp.join(args.image_root_path, f'dpg-bench_{time_style}_results.txt')
    else:
        os.makedirs(osp.dirname(args.res_path), exist_ok=True)
    
    detail_res_path = args.res_path.replace('.txt', '_detail.txt')
    if accelerator.is_main_process:
        with open(args.res_path, 'w') as f:
            pass
        with open(detail_res_path, 'w') as f:
            pass

    device = str(accelerator.device)
    if args.vqa_model == 'mplug':
        vqa_model = MPLUG(device=device)
    else:
        raise NotImplementedError

    filename_list = os.listdir(args.image_root_path)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    filename_list = [
        f for f in filename_list 
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    num_each_rank = len(filename_list) / accelerator.num_processes
    local_rank = accelerator.process_index
    local_filename_list = filename_list[round(local_rank * num_each_rank) : round((local_rank + 1) * num_each_rank)]

    local_category2scores = defaultdict(list)
    local_res_lines = []     # 存储 result.txt 的行
    local_detail_lines = []  # 存储 detail.txt 的行
    local_scores = []
    local_category2scores = defaultdict(list)
    model_id = osp.basename(args.image_root_path)
    print(f'Start to conduct evaluation of {model_id}')
    for fn in tqdm(local_filename_list):
        image_path = osp.join(args.image_root_path, fn)
        try:
            # compute score of one sample
            # score, qid2tuple, qid2scores = compute_dpg_one_sample(
            #     args=args, question_dict=question_dict, image_path=image_path, vqa_model=vqa_model, resolution=args.resolution)
            # local_scores.append(score)
            
            # # summarize scores by categoris
            # for qid in qid2tuple.keys():
            #     category = qid2tuple[qid].split('(')[0].strip()
            #     qid_score = qid2scores[qid]
            #     local_category2scores[category].append(qid_score)

            # compute score
            ret = compute_dpg_one_sample(
                args=args, question_dict=question_dict, image_path=image_path, 
                vqa_model=vqa_model, resolution=args.resolution
            )
            
            if ret[0] is None: continue
            
            score, qid2tuple, qid2scores, res_line, detail_lines = ret
            
            local_scores.append(score)
            
            # 【新增】：存入本地列表
            local_res_lines.append(res_line)
            local_detail_lines.extend(detail_lines) # detail_lines 是个 list，用 extend
            
            # summarize scores by categories
            for qid in qid2tuple.keys():
                category = qid2tuple[qid].split('(')[0].strip()
                qid_score = qid2scores[qid]
                local_category2scores[category].append(qid_score)

        except Exception as e:
            print('Failed filename:', fn, e)
            continue
    
    accelerator.wait_for_everyone()
    global_dpg_scores = gather_object(local_scores)
    mean_dpg_score = np.mean(global_dpg_scores) if global_dpg_scores else 0.0

    global_res_lines = gather_object(local_res_lines)
    global_detail_lines = gather_object(local_detail_lines)

    local_cat_keys = list(local_category2scores.keys())
    global_cat_keys = gather_object(local_cat_keys)
    # Every rank must iterate collectives in the same category order.
    global_categories = sorted(set(global_cat_keys))
    
    global_category2scores = defaultdict(list)

    for category in global_categories:
        loc_scores = local_category2scores.get(category, [])
        gathered_scores = gather_object(loc_scores)
        global_category2scores[category].extend(gathered_scores)

    global_category2scores_l1 = defaultdict(list)
    for category in global_categories:
        l1_category = category.split('-')[0].strip()
        global_category2scores_l1[l1_category].extend(global_category2scores[category])

    time.sleep(3)
    if accelerator.is_main_process:
        print("Gathering complete. Preparing to write...")

        # --- 步骤 1: 先在内存中计算好 Summary 字符串 ---
        output = f'Model: {model_id}\n'
        
        output += 'L1 category scores:\n'
        for l1_category in sorted(global_category2scores_l1.keys()):
            scores = global_category2scores_l1[l1_category]
            output += f'\t{l1_category}: {np.mean(scores) * 100:.2f} (n={len(scores)})\n'
        
        output += 'L2 category scores:\n'
        for category in sorted(global_categories):
            scores = global_category2scores[category]
            output += f'\t{category}: {np.mean(scores) * 100:.2f}\n'

        output += f'Image path: {args.image_root_path}\n'
        output += f'Save results to: {args.res_path}\n'
        output += f'DPG-Bench score: {mean_dpg_score * 100:.4f}'

        # --- 步骤 2: 打开结果文件 (Mode='w')，一次性写入所有内容 ---
        # HDFS 友好模式：Open -> Write All -> Close
        with open(args.res_path, 'w') as f:
            # 2.1 先写原本的 CSV 数据行
            for line in global_res_lines:
                f.write(line)
            
            # 2.2 紧接着写入 Summary (中间加个换行分隔)
            f.write('\n' + output + '\n')
        
        # --- 步骤 3: 写入 Detail 文件 (Mode='w') ---
        with open(detail_res_path, 'w') as f:
            for line in global_detail_lines:
                f.write(line)
        
        print(output)
        print(f"Results successfully saved to {args.res_path}")

if __name__ == "__main__":
    main()
