import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pre-trained-model-path', type=str, default='/home/user/Meta-Llama-3.2-3B-Instruct')
args = parser.parse_args()
pre_trained_model_path = args.pre_trained_model_path

print(f"Loading Model {pre_trained_model_path}")
pre_trained_model = AutoModelForCausalLM.from_pretrained(pre_trained_model_path)

dataset = load_dataset("tatsu-lab/alpaca", split="train").select(range(1000))

def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        response = examples["output"][i]

        text = f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}
        
        ### Response:
        Hello Ajith, {response}
        '''
        output_text.append(text)

    return output_text

training_args = SFTConfig(
    max_length=1024,
    output_dir="/home/user/FineTuneAjith/Meta-Llama-3.2-3B-Instruct-Ajith",
)
trainer = SFTTrainer(
    pre_trained_model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_prompts_func, 
)
trainer.train()
