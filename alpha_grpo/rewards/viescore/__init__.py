import sys
sys.path.insert(0, 'viescore')

from viescore.utils import (
    mllm_output_to_dict
)
import math
import viescore.vie_prompts as vie_prompts
from viescore.mllm_tools.openai import GPT4o, GPT4v, Doubao, Qwen3VL

class VIEScore:
    def __init__(self, backbone="gpt4o", task="t2i", key_path='keys/secret.env') -> None:
        self.task = task 
        self.backbone_name = backbone

        if self.task not in ["t2i", "tie", "t2v"]:
            raise ValueError("task must be either 't2i' or 'tie'")

        if self.backbone_name == "gpt4o":
            self.model = GPT4o(key_path)
        elif self.backbone_name == "gpt4v":
            self.model = GPT4v(key_path)
        elif self.backbone_name == "doubao":
            self.model = Doubao(key_path)
        elif self.backbone_name == "qwen3vl":
            self.model = Qwen3VL(key_path)
        elif self.backbone_name == "gemini":
            from viescore.mllm_tools.gemini import Gemini
            self.model = Gemini()
        elif self.backbone_name == "idefics2":
            from viescore.mllm_tools.idefics2_eval import Idefics2
            self.model = Idefics2()
        elif self.backbone_name == "mantis":
            from viescore.mllm_tools.mantis_idefics2_eval import Mantis
            self.model = Mantis()
        elif self.backbone_name == "minicpmv":
            from viescore.mllm_tools.minicpmv_eval import MiniCPMV
            self.model = MiniCPMV()
        else:
            raise NotImplementedError("backbone not supported")
        self.context = vie_prompts._context_no_delimit
        if self.task == "t2i":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_one_image_gen_rule, vie_prompts._prompts_0shot_t2i_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_rule_PQ])
        elif self.task == "t2i_compare":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_compare_image_gen_rule, vie_prompts._prompts_0shot_t2i_compare_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_rule_t2i_compare_PQ])
        elif self.task == "tie":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_two_image_edit_rule, vie_prompts._prompts_0shot_tie_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_rule_PQ])
        elif self.task == "t2v":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_one_video_gen_rule, vie_prompts._prompts_0shot_t2v_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_t2v_rule_PQ])

    def evaluate(self, image_prompts, text_prompt, extract_overall_score_only=False, extract_all_score=True, echo_output=False):
        if not isinstance(image_prompts, list):
            image_prompts = [image_prompts]
        if self.backbone_name in ['gpt4o', 'gpt4v', 'doubao']:
            self.model.use_encode = False if isinstance(image_prompts[0], str) else True
            #print("Using encode:", self.model.use_encode)
        if self.task == "t2i":
            _SC_prompt = self.SC_prompt.replace("<prompt>", text_prompt)
        elif self.task == "tie":
            _SC_prompt = self.SC_prompt.replace("<instruction>", text_prompt)
        elif self.task == "t2v":
            _SC_prompt = self.SC_prompt.replace("<prompt>", text_prompt)
        SC_prompt_final = self.model.prepare_prompt(image_prompts, _SC_prompt)
        if self.task == "tie":
            PQ_prompt_final = self.model.prepare_prompt(image_prompts[-1], self.PQ_prompt)
        else:
            PQ_prompt_final = self.model.prepare_prompt(image_prompts, self.PQ_prompt)

        results_dict = {}

        SC_dict = False
        PQ_dict = False
        tries = 0
        max_tries = 1
        while SC_dict is False or PQ_dict is False:
            tries += 1
            guess_if_cannot_parse = True if tries > max_tries else False
            for _ in range(2):
                try:
                    result_SC = self.model.get_parsed_output(SC_prompt_final)
                    SC_dict = mllm_output_to_dict(result_SC, give_up_parsing=guess_if_cannot_parse)
                    [float(i) for i in SC_dict['score']]  # make sure the score is List[float or int]
                    SC_dict['reasoning']
                    break
                except Exception as e:
                    print(e)
                    continue

            for _ in range(2):
                try:
                    result_PQ = self.model.get_parsed_output(PQ_prompt_final)
                    PQ_dict = mllm_output_to_dict(result_PQ, give_up_parsing=guess_if_cannot_parse)
                    [float(i) for i in PQ_dict['score']]  # make sure the score is List[float or int]
                    PQ_dict['reasoning']
                    break
                except Exception as e:
                    print(e)
                    continue

        if SC_dict == "rate_limit_exceeded" or PQ_dict == "rate_limit_exceeded":
            print("rate_limit_exceeded") 
            raise ValueError("rate_limit_exceeded")
        results_dict['SC'] = SC_dict
        results_dict['PQ'] = PQ_dict
        if echo_output:
            print("results_dict", results_dict)
        if extract_all_score:
            SC_score = min(results_dict['SC']['score'])
            PQ_score = min(results_dict['PQ']['score'])
            O_score = math.sqrt(SC_score * PQ_score)
            return [SC_score, PQ_score, O_score]
        if extract_overall_score_only:
            SC_scores = results_dict['SC']['score']
            PQ_scores = results_dict['PQ']['score']
            O_score = math.sqrt(min(SC_scores) * min(PQ_scores))
            return O_score
        return results_dict

if __name__ == "__main__":
    model = VIEScore(backbone="gemini", task="t2i")
    from datasets import load_dataset
    dataset = load_dataset("TIGER-Lab/GenAI-Arena-Bench", "image_generation")
    dataset = dataset["test"]
    print("Now running the VIEScore model")
    for idx in range(5):
        left_image = dataset['left_image'][idx]
        right_image = dataset['right_image'][idx]
        prompt = dataset['prompt'][idx]
        print(model.evaluate(left_image, prompt, extract_all_score=True))
        print(model.evaluate(right_image, prompt, extract_all_score=True))

