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
