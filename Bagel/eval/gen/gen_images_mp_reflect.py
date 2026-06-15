# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
from safetensors.torch import load_file
import copy

import torch
import torch.distributed as dist
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

from PIL import Image
from modeling.bagel.qwen2_navit import NaiveCache
from inferencer import InterleaveInferencer
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

reflection_and_regenerate_prompt_with_caption = """The user requires to generate image of `{}`. You tried to generate an image based on the user's request, but you failed to create the correct image. 
Reflect on what went wrong and write down the editing instruction that will help you do better and refine the image to meet user's request based on your own reflection."""


def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def apply_scale(width, height, scale):
    def _make_divisible(value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))
    
    new_width = round(width * scale)
    new_height = round(height * scale)
    new_width = _make_divisible(new_width, 16)
    new_height = _make_divisible(new_height, 16)
    return new_width, new_height


@torch.no_grad()
def reflect_image(
    images, prompt, num_timesteps=50, 
    cfg_text_scale=4.0, cfg_img_scale=2.0,
    cfg_interval=[0, 1.0], cfg_renorm_min=0., 
    cfg_type="serial_text_img", cfg_renorm_type="text_channel", 
    timestep_shift=3.0, max_image_size=1024, min_image_size=512, img_size=None,
    max_length=2048, simple_think=False, device=None
):
    # FIXME: acutally not very suitable for video input
    for image in images:
        # add VAE
        generation_input, newlens, new_rope = gen_model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image],
            transforms=vae_transform, 
            new_token_ids=new_token_ids,
            #timestep=0.0,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vae(vae_model, past_key_values, **generation_input)

        # add ViT
        generation_input, newlens, new_rope = gen_model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image],
            transforms=vit_transform, 
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vit(past_key_values, **generation_input)
        
    ########## think
    tmp_past_key_values = copy.deepcopy(past_key_values)
    tmp_newlens = copy.deepcopy(newlens)
    tmp_new_rope = copy.deepcopy(new_rope)
    tmp_generation_input, tmp_newlens, tmp_new_rope = gen_model.prepare_prompts(
        curr_kvlens=tmp_newlens,
        curr_rope=tmp_new_rope, 
        prompts=[reflection_and_regenerate_prompt_with_caption.format(prompt)],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        tmp_past_key_values = gen_model.forward_cache_update_text(tmp_past_key_values, **tmp_generation_input)  
    
    tmp_generation_input = gen_model.prepare_start_tokens(tmp_newlens, tmp_new_rope, new_token_ids)
    tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_text(
            past_key_values=tmp_past_key_values,
            max_length=max_length,
            do_sample=True,
            temperature=0.3,
            end_token_id=new_token_ids['eos_token_id'],
            **tmp_generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        think_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]  
        
    print("="*30, "original think", "="*30)
    print(think_output) 
    if simple_think:
        think_output_list = think_output.split("</think>")
        if think_output_list[1] != "":
            think_output = think_output_list[1].strip()
        print("="*30, "processed think", "="*30)
        print(think_output) 
    ########## think
    
    ##########  cfg_text
    cfg_text_past_key_values = copy.deepcopy(past_key_values)
    generation_input_cfg_text = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)], 
    )
    generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, device)
    
    ##########  cfg_img
    cfg_img_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0]
    cfg_img_new_rope = [0]
    
    # system prompt
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[SYSTEM_PROMPT],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    # add think_output
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[think_output],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    generation_input_cfg_img = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        image_sizes=[(h, w)], 
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    
    ##########  origin
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    
    # add think_output
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[think_output],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    
    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,  
        image_sizes=[(h, w)], 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type=cfg_type,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

    latent = unpacked_latent[0]
    latent = latent.reshape(1, h//16, w//16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, h//8, w//8)
    tmpimage = vae_model.decode(latent)
    tmpimage = ((tmpimage * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    tmpimage = Image.fromarray(tmpimage)
    
    return tmpimage, think_output


@torch.no_grad()
def reflect_image_v2(
    images, prompt, num_timesteps=50, 
    cfg_text_scale=4.0, cfg_img_scale=2.0,
    cfg_interval=[0, 1.0], cfg_renorm_min=0., 
    cfg_type="serial_text_img", cfg_renorm_type="text_channel", 
    timestep_shift=3.0, 
    max_length=2048, device=None
):
    input_lists=[[init_img, reflection_and_regenerate_prompt_with_caption.format(prompt)] for init_img in images]

    images_refined, think_texts = inferencer.batch_generate_images(
        input_lists=input_lists,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        num_timesteps=num_timesteps,
        cfg_interval=cfg_interval,
        timestep_shift=timestep_shift,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        deterministic=True,  # we use ode for evaluation.
        think=True,
        text_temperature=0.3,
    )
    return images_refined[0], think_texts[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Bagel model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--source_output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--metadata_file", type=str, required=True, help="JSONL file containing lines of metadata for each prompt.")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg_text_scale", type=float, default=4)
    parser.add_argument("--cfg_img_scale", type=float, default=2)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--only_wrong', action='store_true', help='Only generate refined-images for the wrong t2i images.')
    args = parser.parse_args()
    
    seed = 42
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    
    output_dir = args.output_dir
    source_output_dir = args.source_output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images are saved in {output_dir}")

    if not os.path.exists(source_output_dir):
        print(f"Source output directory {source_output_dir} does not exist.")

    # remove the last subdir and get the results.jsonl file
    source_results = [json.loads(line) for line in open(os.path.join(os.path.dirname(source_output_dir), "results.jsonl"), "r")]
    img2result = {'/'.join(line['filename'].split("/")[-3:]): line for line in source_results}


    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 378, 14)

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False, assign=True)
    if rank == 0:
        print(msg)
    del model_state_dict
    model.language_model.resize_token_embeddings(len(tokenizer))

    # vae_model.to('cuda', dtype=inference_dtype).eval()
    # model.to('cuda', dtype=inference_dtype).eval()

    lora_path = os.environ.get('BAGEL_LORA_PATH', None)
    if lora_path is not None:
        from peft import  PeftModel
        model = PeftModel.from_pretrained(model, lora_path, device_map={"": int(os.getenv('LOCAL_RANK', 0))})
        model.set_adapter("default")
        model.merge_and_unload()
        print("load lora from ", lora_path)

    model = model.to(device).eval()
    vae_model = vae_model.to(device).eval()
    gen_model = model

    inferencer = InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)

    cfg_text_scale = args.cfg_text_scale
    cfg_img_scale = args.cfg_img_scale
    cfg_interval = [0.4, 1.0]
    timestep_shift = 3.0
    num_timesteps = 50
    cfg_renorm_min = 0.0

    with open(args.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp]
    total_metadatas = len(metadatas)
    
    idx_list = list(range(total_metadatas))[rank::world_size]
    print(f"GPU {rank}: Processing {len(idx_list)} prompts")

    # for idx in range(start, end):
    for i, idx in enumerate(idx_list):
        metadata = metadatas[idx]
        outpath = os.path.join(output_dir, f"{idx:0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']
        print(f"GPU {rank} processing prompt {i}/{len(idx_list)}: '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp)
        
        flag = True
        for image_idx in range(args.num_images):
            if not os.path.exists(os.path.join(sample_path, f"{image_idx:05}.png")):
                flag = False
                break

        if flag:

            print(f"GPU {rank} skipping generation for prompt: {prompt}")
            continue
        
        sample_count = 0
        image_list = []
        think_text_list = []

        for image_idx in range(args.num_images):
            source_sample_path = os.path.join(source_output_dir, f"{idx:0>5}", "samples", f"{image_idx:05}.png")
            save_sample_path= os.path.join(sample_path, f"{image_idx:05}.png")

            source_image = Image.open(source_sample_path).convert('RGB')
            
            key = '/'.join(source_sample_path.split('/')[-3:])

            if not args.only_wrong or (args.only_wrong and not img2result[key]['correct']):
                tmp_image, think_output = reflect_image_v2(
                    images=[source_image],
                    prompt=prompt,
                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    cfg_renorm_min=cfg_renorm_min,
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    max_length=2048, 
                    device=device,
                )
                think_text_list.append(think_output)
                image_list.append(tmp_image)

                sample = tmp_image.crop(tmp_image.getbbox())
                sample.save(save_sample_path)
                with open(save_sample_path.replace('.png', '.txt'), "w") as f:
                    f.write(think_output)
            else:
                print(f"GPU {rank} skipping the correct generation for prompt: {prompt} image_idx: {image_idx}")
                source_image.save(save_sample_path)


        # sample_count = 0
        # for image_idx, sample in enumerate(image_list):
        #     # sample = sample.crop(sample.getbbox())
        #     sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
        #     with open(os.path.join(outpath, f"{sample_count:05}.txt"), "w") as f:
        #         f.write(think_text_list[image_idx])
        #     sample_count += 1
        
    print(f"GPU {rank} has completed all tasks")
    dist.barrier()