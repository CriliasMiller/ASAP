import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

def parse():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"]) 
    return local_rank, world_size

def setup_distributed(local_rank, world_size):
    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)

def load_model(model_path, local_rank):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()
    model = DDP(model.to(f'cuda:{local_rank}'), device_ids=[local_rank], output_device=local_rank)
    return model

class CaptionIterableDataset(Dataset):
    def __init__(self, json_file, root_dir, tokenizer):
        self.json_file = json_file
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        with open(self.json_file, 'r') as f:
            self.json_contents = json.load(f)

    def __len__(self):
        return len(self.json_contents)
    
    def __getitem__(self, idx):
        prompt = 'Refer to the following text to describe the specific information of the corresponding image: '
        caption = self.json_contents[idx]['text']
        img_dir = self.json_contents[idx]['image']
        caption = f'{prompt}"{caption}"'
        return caption, img_dir

def write_to_json(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n') 

def main():
    local_rank, world_size = parse()
    setup_distributed(local_rank, world_size)

    model_path = '' # pick your own model
    model = load_model(model_path, local_rank)
    print(f"Model loaded on rank {local_rank}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    json_file = 'unprocessed_train.json'
    root_dir = '../data/rshaojimmy/'
    test = 'Caption_Prompt/test.json'

    dataset = CaptionIterableDataset(json_file, root_dir, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, sampler=sampler)

    gen_kwargs = {"max_length": 120, "do_sample": True, "top_k": 1}
    temp_file = f'temp/llm_{local_rank}.json'
    with open(temp_file, 'w') as f:
        f.write('[')
    for batch in tqdm(dataloader):
        query, img_dir = batch
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        inputs = inputs.to(f'cuda:{local_rank}')
        result = {}
        with torch.no_grad():
            outputs = model.module.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result[img_dir[0]] = answer
            with open(temp_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + ',\n')
    with open(temp_file, 'a') as f:
        f.write(']')
if __name__ == '__main__':
    main()
