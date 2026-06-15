# This file is generated automatically through parse_prompt.py

_context_no_delimit = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{
"score" : [...],
"reasoning" : "..."
}"""

_context = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (the delimiter is necessary. Keep your reasoning concise and short.):
||V^=^V||
{
"score" : 
"reasoning" : 
}
||V^=^V||"""

_context_no_format = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials."""

_prompts_1shot_multi_subject_image_gen_rule = """RULES of each set of inputs:

Two images will be provided: 
This first image is a concatenation of two sub-images, each sub-image contain one token subject.
The second image being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_1shot_mie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

First lets look at the first set of input (1st and 2nd images) as an example. 
Editing instruction: What if the man had a hat?
Output:
||V^=^V||
{
"score" : [5, 10],
"reasoning" :  "The hat exists but does not suit well. The hat also looks distorted. But it is a good edit because only a hat is added and the background is persevered."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Editing instruction: <instruction>
"""

_prompts_1shot_msdig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first sub-image. 
(0 indicates that the subject in the second image does not look like the token subject in the first sub-image at all. 10 indicates the subject in the second image look exactly alike the token subject in the first sub-image.)
A third score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the second sub-image. 
(0 indicates that the subject in the second image does not look like the token subject in the second sub-image at all. 10 indicates the subject in the second image look exactly alike the token subject in the second sub-image.)
Put the score in a list such that output score = [score1, score2, score3], where 'score1' evaluates the prompt and 'score2' evaluates the resemblance for the first sub-image, and 'score3' evaluates the resemblance for the second sub-image.

First lets look at the first set of input (1st and 2nd images) as an example. 
Text Prompt: A digital illustration of a cat beside a wooden pot
Output:
||V^=^V||
{
"score" : [5, 5, 10],
"reasoning" :  "The cat is not beside the wooden pot. The pot looks partially resemble to the subject pot. The cat looks highly resemble to the subject cat."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Text Prompt: <prompt>"""

_prompts_1shot_t2i_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the AI generated image does not follow the prompt at all. 10 indicates the AI generated image follows the prompt perfectly.)

Put the score in a list such that output score = [score].

First lets look at the first set of input (1st image) as an example. 
Text Prompt: A pink and a white frisbee are on the ground.
Output:
||V^=^V||
{
"score" : [5],
"reasoning" :  "White frisbee not present in the image."
}
||V^=^V||

Now evaluate the second set of input (2nd image).
Text Prompt: <prompt>
"""

_prompts_1shot_tie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

First lets look at the first set of input (1st and 2nd images) as an example. 
Editing instruction: What if the man had a hat?
Output:
||V^=^V||
{
"score" : [5, 10],
"reasoning" :  "The hat exists but does not suit well. The hat also looks distorted. But it is a good edit because only a hat is added and the background is persevered."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Editing instruction: <instruction>
"""

_prompts_1shot_sdie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the second image. 
(0 indicates that the subject in the third image does not look like the token subject at all. 10 indicates the subject in the third image look exactly alike the token subject.)
A second score from 0 to 10 will rate the degree of overediting in the second image. 
(0 indicates that the scene in the edited image is completely different from the first image. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the resemblance and 'score2' evaluates the degree of overediting.

First lets look at the first set of input (1st, 2nd and 3rd images) as an example. 
Subject: <subject>
Output:
||V^=^V||
{
"score" : [5, 10],
"reasoning" :  "The monster toy looks partially resemble to the token subject. The edit is minimal."
}
||V^=^V||

Now evaluate the second set of input (4th, 5th, and 6th images).
Subject: <subject>
"""

_prompts_1shot_one_image_gen_rule = """RULES of each set of inputs:

One image will be provided; The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_1shot_sdig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first image. 
(0 indicates that the subject in the second image does not look like the token subject at all. 10 indicates the subject in the second image look exactly alike the token subject.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the prompt and 'score2' evaluates the resemblance.

First lets look at the first set of input (1st and 2nd images) as an example. 
Text Prompt: a red cartoon figure eating a banana
Output:
||V^=^V||
{
"score" : [10, 5],
"reasoning" :  "The red cartoon figure is eating a banana. The red cartoon figure looks partially resemble to the subject."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Text Prompt: <prompt>
"""

_prompts_1shot_rule_PQ = """RULES of each set of inputs:

One image will be provided; The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on image naturalness. 
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts. 
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]


First lets look at the first set of input (1st image) as an example. 
Output:
||V^=^V||
{
"score" : [5, 5],
"reasoning" :  "The image gives an unnatural feeling on hands of the girl. There is also minor distortion on the eyes of the girl."
}
||V^=^V||

Now evaluate the second set of input (2nd image).

"""

_prompts_1shot_subject_image_gen_rule = """RULES of each set of inputs:

Two images will be provided: The first being a token subject image and the second being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_1shot_cig_rule_SC = """
From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the generated image is following the guidance image. 
(0 indicates that the second image is not following the guidance at all. 10 indicates that second image is following the guidance image.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the prompt and 'score2' evaluates the guidance.

First lets look at the first set of input (1st and 2nd images) as an example. 
Text Prompt: the bridge is red, Golden Gate Bridge in San Francisco, USA
Output:
||V^=^V||
{
"score" : [5, 5],
"reasoning" :  "The bridge is red. But half of the bridge is gone."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Text Prompt: <prompt>
"""

_prompts_1shot_two_image_edit_rule = """RULES of each set of inputs:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.
"""

_prompts_1shot_subject_image_edit_rule = """RULES of each set of inputs:

Three images will be provided: 
The first image is a input image to be edited.
The second image is a token subject image.
The third image is an AI-edited image from the first image. it should contain a subject that looks alike the subject in second image.
The objective is to evaluate how successfully the image has been edited.
"""

_prompts_1shot_control_image_gen_rule = """RULES of each set of inputs:

Two images will be provided: The first being a processed image (e.g. Canny edges, openpose, grayscale etc.) and the second being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_two_image_edit_rule = """RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.
"""

_prompts_0shot_one_video_gen_rule = """RULES:

The images are extracted from a AI-generated video according to the text prompt.
The objective is to evaluate how successfully the video has been generated.
"""

_prompts_0shot_t2v_rule_PQ = """RULES:

The image frames are AI-generated.
The objective is to evaluate how successfully the image frames has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on the image frames naturalness. 
(
    0 indicates that the scene in the image frames does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image frames looks natural.
)
A second score from 0 to 10 will rate the image frames artifacts. 
(
    0 indicates that the image frames contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image frames has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""

_prompts_0shot_msdig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first sub-image. 
(0 indicates that the subject in the second image does not look like the token subject in the first sub-image at all. 10 indicates the subject in the second image look exactly alike the token subject in the first sub-image.)
A third score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the second sub-image. 
(0 indicates that the subject in the second image does not look like the token subject in the second sub-image at all. 10 indicates the subject in the second image look exactly alike the token subject in the second sub-image.)
Put the score in a list such that output score = [score1, score2, score3], where 'score1' evaluates the prompt and 'score2' evaluates the resemblance for the first sub-image, and 'score3' evaluates the resemblance for the second sub-image.

Text Prompt: <prompt>
"""

_prompts_0shot_sdie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the second image. 
(0 indicates that the subject in the third image does not look like the token subject at all. 10 indicates the subject in the third image look exactly alike the token subject.)
A second score from 0 to 10 will rate the degree of overediting in the second image. 
(0 indicates that the scene in the edited image is completely different from the first image. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the resemblance and 'score2' evaluates the degree of overediting.

Subject: <subject>"""

_prompts_0shot_subject_image_edit_rule = """RULES:

Three images will be provided: 
The first image is a input image to be edited.
The second image is a token subject image.
The third image is an AI-edited image from the first image. it should contain a subject that looks alike the subject in second image.
The objective is to evaluate how successfully the image has been edited.
"""

_prompts_0shot_mie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: <instruction>
"""

_prompts_0shot_sdig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first image. 
(0 indicates that the subject in the second image does not look like the token subject at all. 10 indicates the subject in the second image look exactly alike the token subject.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the prompt and 'score2' evaluates the resemblance.

Text Prompt: <prompt>
"""

_prompts_0shot_tie_rule_SC = """
From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: <instruction>
"""

_prompts_0shot_t2i_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the AI generated image does not follow the prompt at all. 10 indicates the AI generated image follows the prompt perfectly.)

Put the score in a list such that output score = [score].

Text Prompt: <prompt>
"""


_prompts_0shot_t2i_compare_rule_SC = """From scale 0 to 10: 
Use the Text Prompt as ground truth and evaluate how well the Candidate image follows the prompt relative to the Baseline image.

Scale:
  - 0 = Candidate follows the prompt much worse than the Baseline (or not at all).
  - 5 = Candidate follows the prompt about as well as the Baseline.
  - 10 = Candidate follows the prompt much better than the Baseline, capturing details and intent more completely.

Evaluation criteria (compare Candidate vs. Baseline w.r.t. the prompt):
  - Core elements: objects, counting, attributes, colors, materials, spatial relations, actions, poses, expressions, etc.
  - Rendered text (if required): correctness, spelling, legibility, placement, etc.
  - Overall intent: style, composition, lighting, atmosphere, camera/viewpoint, etc., as specified.

Guidelines:
  - Use the Text Prompt as the only ground truth; do not copy Baseline errors.
  - If both deviate, score by which is closer to the prompt.
  - Prioritize explicit constraints over ambiguous or implied ones.
  - Ignore differences not required by the prompt (e.g., palette/background if unspecified).
  - If both are equally aligned with the prompt, return 5.

Output format: Return only the list exactly as: output score = [score].

Text Prompt: <prompt>
"""


_prompts_0shot_cig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the generated image is following the guidance image. 
(0 indicates that the second image is not following the guidance at all. 10 indicates that second image is following the guidance image.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the prompt and 'score2' evaluates the guidance.

Text Prompt: <prompt>"""

_prompts_0shot_control_image_gen_rule = """RULES:

Two images will be provided: The first being a processed image (e.g. Canny edges, openpose, grayscale etc.) and the second being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_rule_PQ = """RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on image naturalness. 
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts. 
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""

_prompts_0shot_rule_t2i_compare_PQ = """RULES:
Two AI-generated images are provided: the first is the Baseline image and the second is the Candidate image.
The objective is to evaluate how successfully the Candidate image has been generated relative to the Baseline.
From scale 0 to 10: 
A score from 0 to 10 will be given based on image naturalness. 
(
    0 indicates that the scene in the Candidate image looks much less natural than the Baseline (e.g., wrong sense of distance, wrong shadow, or wrong lighting). 
    5 indicates that the Candidate image looks as natural as the Baseline image.
    10 indicates that the Candidate image looks much more natural than the Baseline.
)
A second score from 0 to 10 will rate the image artifacts. 
(
    0 indicates that the Candidate image contains far more distortion, watermark, scratches, blurred faces, unusual body parts, or subjects not harmonized than the Baseline. 
    5 indicates that the Candidate image has the same level of artifacts as the Baseline image.
    10 indicates the Candidate image has far fewer artifacts than the Baseline (ideally none).
)
Put the score in a list such that output score = [naturalness, artifacts]
"""


_prompts_0shot_t2v_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the image frames does not follow the prompt at all. 10 indicates the image frames follows the prompt perfectly.)

Put the score in a list such that output score = [score].

Text Prompt: <prompt>
"""

_prompts_0shot_multi_subject_image_gen_rule = """RULES:

Two images will be provided: 
This first image is a concatenation of two sub-images, each sub-image contain one token subject.
The second image being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_subject_image_gen_rule = """RULES:

Two images will be provided: The first being a token subject image and the second being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_one_image_gen_rule = """RULES:

The image is an AI-generated image according to the text prompt.
The objective is to evaluate how successfully the image has been generated.
"""


_prompts_compare_image_gen_rule = """RULES:

Two image will be provided: Both are the AI-generated images according to the text prompt. But we indicate the first image as the baseline image.
The objective is to evaluate how well the second image follows the prompt compared to the baseline image.
"""


_prompts_0shot_image_refinement_rule = """RULES:

Two images will be provided: The first being a reference image and the second being an AI-generated image refined from the referenced image to meet the provided image caption.
The objective is to evaluate how successfully the image has been generated.
"""

