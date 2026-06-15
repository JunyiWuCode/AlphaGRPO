import os

# read the results of the evaluation.

job='ocr_bagel_reflect_clamp_4nodes_v2/'
root = f'./output{job}/eval_vlm_output'


for subdir in os.listdir(root):
    print()
    print(subdir)
    for task in os.listdir(os.path.join(root, subdir)):
        print(task)
        if os.path.exists(os.path.join(root, subdir, task, 'results.txt')):
            print(open(os.path.join(root, subdir, task, 'results.txt')).read())