# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
from PIL import Image, ImageFile, PngImagePlugin
import json

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        image_num = len(row["image_list"])
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True, 
        )

        if end_idx - start_idx > 1 and random.random() < 0.5: # concat multiple insturction
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(row["instruction_list"][idx-1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(row["instruction_list"][idx-1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=True, 
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=False, 
                        need_vit=False,
                    )
        return data



class T2IWithReflectionThinkingIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    # Text -> Image -> Text -> Think Text -> Image

    reflection_system_prompt = '''You have generated an image based on the user's request. Your task is to critically evaluate this image.
    You should first perform the reflection process in the mind and then decide to refine the image or not. The reflection process is enclosed within <think> </think> tags under the json format:

    **JSON Format:**
    {
    "Decision": "'PASS' or 'FAIL'",
    "Reflection": "[Your reflection]",
    "Refinement plan": "[Your refinement plan if FAIL, otherwise 'no change']"
    }
    [Action]

    **Action:**
    - If `Decision` is `PASS`, output the text: 'The generated image meets the user's request.'
    - If `Decision` is `FAIL`, generate the refined image according to the refinement plan.'''


    def parse_row(self, row):
        data = self._init_data()

        # text to image pair.
        data = self._add_text(data, row['caption'].rstrip(), need_loss=False)
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["init_image"]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True,
        )
        
        # reflection mode.
        data = self._add_text(data, self.reflection_system_prompt, need_loss=False)
        
        think_text = row['think_text']
        decision = json.loads(think_text)['Decision']

        if not think_text.startswith('<think>'):
            think_text = "<think>\n" + think_text.strip() + "\n</think>"
        data = self._add_text(data, think_text, need_loss=True)
        
        if decision == 'PASS':
            data = self._add_text(data, row['output_text'], need_loss=True)
        elif decision == 'FAIL':
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(row['refined_image']))),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )
        else:
            raise RuntimeError(f'Unknown result: {row["result"]}')
        return data


class ReflectionThinkingOneConvIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    # System Prompt: Text + Image -> Think Text -> Image
    GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
    The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

    reflection_and_regenerate_prompt_with_caption = """The user requires to generate image of `{}`. You tried to generate an image based on the user's request, but you failed to create the correct image. 
    Reflect on what went wrong and write down the editing instruction that will help you do better and refine the image to meet user's request based on your own reflection."""

    def parse_row(self, row):
        data = self._init_data()

        resolution = 512 if random.random() < 0.5 else 1024

        # text to image pair.
        data = self._add_text(data, self.GEN_THINK_SYSTEM_PROMPT, need_loss=False)
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["init_image"])).resize((resolution, resolution))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True,
        )
        data = self._add_text(data, self.reflection_and_regenerate_prompt_with_caption.format(row['caption'].rstrip()), need_loss=False)
        
        # reflection mode.
        think_text = row['think_text'].strip()
        if not think_text.startswith('<think>'):
            think_text = "<think>\n" + think_text + "\n</think>"
        data = self._add_text(data, think_text, need_loss=True)
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row['refined_image'])).resize((resolution, resolution))),
            need_loss=True, 
            need_vae=False, 
            need_vit=False,
        )
        return data


class ReflectionThinkingOneConvOnlyTextIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    # System Prompt: Text + Image -> Think Text -> Image
    GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
    The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

    reflection_and_regenerate_prompt_with_caption = """The user requires to generate image of `{}`. You tried to generate an image based on the user's request, but you failed to create the correct image. 
    Reflect on what went wrong and write down the editing instruction that will help you do better and refine the image to meet user's request based on your own reflection."""

    def parse_row(self, row):
        data = self._init_data()

        resolution = 512 if random.random() < 0.5 else 1024

        # text to image pair.
        data = self._add_text(data, self.GEN_THINK_SYSTEM_PROMPT, need_loss=False)
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["init_image"])).resize((resolution, resolution))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True,
        )
        data = self._add_text(data, self.reflection_and_regenerate_prompt_with_caption.format(row['caption'].rstrip()), need_loss=False)
        
        # reflection mode.
        think_text = row['think_text'].strip()
        if not think_text.startswith('<think>'):
            think_text = "<think>\n" + think_text + "\n</think>"
        data = self._add_text(data, think_text, need_loss=True)
        return data


class T2IWithReflectionThinkingV2IterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    # Text -> Image -> Text -> Think Text -> Image

    reflection_system_promptv2 = '''You have generated an image based on the user's request.
    You should first perform the reflection process to critically evaluate this image in the mind and then follow the reflection to refine the image. The reflection process is enclosed within <think> </think> tags under the json format:

    **JSON Format:**
    {
    "Decision": "'PASS' or 'FAIL'",
    "Reflection": "[Your reflection]",
    "Refinement plan": "[Your refinement plan if FAIL, otherwise 'no change']"
    }
    [Generate the refined image according to the refinement plan.]'''


    def parse_row(self, row):
        data = self._init_data()

        # text to image pair.
        data = self._add_text(data, row['caption'].rstrip(), need_loss=False)
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["init_image"]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True,
        )
        
        # reflection mode.
        data = self._add_text(data, self.reflection_system_promptv2, need_loss=False)
        
        think_text = row['think_text']
        decision = json.loads(think_text)['Decision']

        if not think_text.startswith('<think>'):
            think_text = "<think>\n" + think_text.strip() + "\n</think>"
        data = self._add_text(data, think_text, need_loss=True)
        
        if decision == 'PASS':
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(row['init_image']))),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )
        elif decision == 'FAIL':
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(row['refined_image']))),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )
        else:
            raise RuntimeError(f'Unknown result: {row["result"]}')
        return data
