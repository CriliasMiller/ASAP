import json, random
from tqdm import tqdm

text_prompt = json.load(open('../GLM_QQWEN/glm_prompt.json'))
text_prompt_mistral = json.load(open('../Prompt_Generate/prompt_train.json'))

print(f'{len(text_prompt)} {len(text_prompt_mistral)}')
new_dataset = {}
for key in tqdm(text_prompt):
    if random.random() < 0.5:
        new_dataset[key] = text_prompt[key]

file_path = 'prompt_engine.json'
with open(file_path, 'w') as f:
    json.dump(new_dataset, f, ensure_ascii=False, indent=4)
print(len(new_dataset))

