# FineTune-and-Deploy-LLaMA

### 1. Prerequisites : 
- Install Docker & Nvidia Docker. Follow [Link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) </br>
- Make sure all 8xNvidia H100 GPUs are visible. </br>
- Check GPUs status with Command : `nvidia-smi`


### 2. Setup TRT-LLM Docker Container : 

###### Replace this with your Work Space Path. Minimum Disk Space Required : 400GB

```
export HOSTSPACE="/mnt/Scratch_space/ajith"  
```
### Run the Docker Container : 

```
sudo docker pull huggingface/transformers-pytorch-gpu:latest
sudo docker run --runtime=nvidia --name=Ajith_Transformers_Latest_8xGPU --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
--cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH --security-opt seccomp=unconfined --gpus=all -it \ 
-v /mnt/Scratch_space/ajith:/home/user -w /home/user huggingface/transformers-pytorch-gpu:latest bash
```

### Inside the Container : 

```
git clone --recursive https://github.com/huggingface/transformers.git 
pip install --upgrade transformers
```
```
apt-get update -y && apt-get install vim wget -y
```

### Download the LLaMA 3.2 3B Model 
```
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --exclude "original/*" --local-dir Meta-Llama-3.2-3B-Instruct
```

### Run Fine-Tuning Sample Code : 
```
python3 run_finetune.py --pre-trained-model-path /home/user/Meta-Llama-3.2-3B
```

### Fine-Tuning Result : 
```
[codecarbon INFO @ 06:14:14] Energy consumed for all GPUs : 0.478299 kWh. Total GPU Power : 439.97153721911457 W
[codecarbon INFO @ 06:14:14] 0.589190 kWh of electricity used since the beginning.
{'train_runtime': 3105.1733, 'train_samples_per_second': 9.661, 'train_steps_per_second': 1.208, 'train_loss': 0.6672996693929036, 'num_tokens': 3079518.0, 'mean_token_accuracy': 0.9303302190303803, 'epoch': 3.0}
100%|██████████████████████████████████████████████████████████| 3750/3750 [51:44<00:00,  1.16it/s][codecarbon INFO @ 06:14:27] Energy consumed for RAM : 0.044872 kWh. RAM Power : 54.0 W
[codecarbon INFO @ 06:14:27] Delta energy consumed for CPU with cpu_load : 0.000284 kWh, power : 80.0 W
[codecarbon INFO @ 06:14:27] Energy consumed for All CPU : 0.066495 kWh
100%|██████████████████████████████████████████████████████████| 3750/3750 [51:45<00:00,  1.21it/s]
```
### Model Saved at : 
```
root@user:/home/user/FineTuneAjith# ls -l -h Meta-Llama-3.2-3B-Instruct-Ajith/checkpoint-3750
total 36G
-rw-r--r-- 1 root root 3.8K May 16 06:14 chat_template.jinja
-rw-r--r-- 1 root root  877 May 16 06:14 config.json
-rw-r--r-- 1 root root  189 May 16 06:14 generation_config.json
-rw-r--r-- 1 root root 4.7G May 16 06:14 model-00001-of-00003.safetensors
-rw-r--r-- 1 root root 4.6G May 16 06:14 model-00002-of-00003.safetensors
-rw-r--r-- 1 root root 2.8G May 16 06:14 model-00003-of-00003.safetensors
-rw-r--r-- 1 root root  21K May 16 06:14 model.safetensors.index.json
-rw-r--r-- 1 root root  24G May 16 06:14 optimizer.pt
-rw-r--r-- 1 root root  15K May 16 06:14 rng_state.pth
-rw-r--r-- 1 root root 1.5K May 16 06:14 scheduler.pt
-rw-r--r-- 1 root root  296 May 16 06:14 special_tokens_map.json
-rw-r--r-- 1 root root  17M May 16 06:14 tokenizer.json
-rw-r--r-- 1 root root  50K May 16 06:14 tokenizer_config.json
-rw-r--r-- 1 root root 2.4K May 16 06:14 trainer_state.json
-rw-r--r-- 1 root root 6.1K May 16 06:14 training_args.bin
root@user:/home/user/FineTuneAjith#
```
