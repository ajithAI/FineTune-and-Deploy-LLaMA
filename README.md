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
python3 run_finetune_alpaca.py --pre-trained-model-path /home/user/Meta-Llama-3.2-3B
```
##### Fine-Tuned Model will get saved at : ```/home/user/FineTuneAjith/Meta-Llama-3.2-3B-Instruct-Ajith```

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
root@user:/home/user/FineTuneAjith# ls -l -h Meta-Llama-3.2-3B-Instruct-Ajith/checkpoint-375
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
```

### Optimize the Fine-Tuned Model with TRT-LLM : 

### Create Triton TRT-LLM Docker Container : 
```
sudo docker run -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus '"device=0"' \
    -v /mnt/Scratch_space/ajith/:/home/user -w /home/user --name Ajith_Triton_Server \
    nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3 bash
```
### Install Tensor-RT LLM Backend : 
```
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git  --branch v0.19.0
cd tensorrtllm_backend
# Install git-lfs if needed
apt-get update && apt-get install git-lfs -y --no-install-recommends
git lfs install
git submodule update --init --recursive
```

```
cd /home/user/tensorrtllm_backend/tensorrt_llm/examples/llama
python3 convert_checkpoint.py --model_dir /home/user/FineTuneAjith/Meta-Llama-3.2-3B-Instruct-Ajith/checkpoint-375 \
      --output_dir /home/user/FineTuneAjith/Meta-Llama-3.2-3B-Instruct-Ajith-CHECKPOINT \
      --dtype bfloat16 --tp_size 1

trtllm-build --checkpoint_dir /home/user/FineTuneAjith/Meta-Llama-3.2-3B-Instruct-Ajith-CHECKPOINT \
    --output_dir /home/user/FineTuneAjith/Meta-Llama-3.2-3B-Instruct-Ajith-TRT-Engine --gemm_plugin auto
```


### Triton Server Setup 
```
TOKENIZER_DIR=/home/user/Meta-Llama-3.2-3B-Instruct
TOKENIZER_TYPE=auto
ENGINE_DIR=/home/user/FineTuneAjith/Meta-Llama-3.2-3B-Instruct-Ajith-TRT-Engine
DECOUPLED_MODE=false
MODEL_FOLDER=/opt/tritonserver/inflight_batcher_llm
MAX_BATCH_SIZE=4
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=10000
TRITON_BACKEND=tensorrtllm
LOGITS_DATATYPE="TYPE_FP32"
FILL_TEMPLATE_SCRIPT=/home/user/tensorrtllm_backend/tools/fill_template.py

python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT} 
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},logits_datatype:${LOGITS_DATATYPE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},logits_datatype:${LOGITS_DATATYPE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:${TRITON_BACKEND},triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,encoder_input_features_data_type:TYPE_FP16,logits_datatype:${LOGITS_DATATYPE}
```

### Triton Server Initiation 

```
python3 /home/user/tensorrtllm_backend/scripts/launch_triton_server.py --world_size=1 --model_repo=/opt/tritonserver/inflight_batcher_llm
```
![{F6943BFE-B11B-4AF1-8E22-99BCE4BB4D07}](https://github.com/user-attachments/assets/669f2ee4-f878-4c12-a5a7-359145dd1b9f)


### Send Request to Triton Server : 
```
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n ### Instruction:\n What are the three primary colors? ", "max_tokens": 500, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
```

### Result : 
![{07E02FF6-5835-4EC4-AD2B-3DC83642E77F}](https://github.com/user-attachments/assets/77bfc69c-f5b5-4045-b56e-3e42ceac7a17)
