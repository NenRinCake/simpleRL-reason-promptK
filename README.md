

<div align="center">

# Added the construction of k-step prompts on top of simpleRL-reason

</div>


## Links 

* **[The code of simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)**



## Quick Start

### Installation

```bash
conda create -n verl python==3.9
conda activate verl
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e . 
```

To install from docker image or utilize Megatron-lm, please refer to [Verl's documentation](https://verl.readthedocs.io/en/v0.2.x/start/install.html).

### Reproducing

The training process leverages GRPO with Ray and vLLM for acceleration. So firstly, you need to launch the ray cluster using the command below:
```bash
# launch the master node of ray 
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8
```
The main script for training is train_grpo_math_tune_ray.sh. You need to specify the required environment variables in this script. Once configured, submit the training job from the master node.

For other models, use the same command, adjusting the `--model_name` argument accordingly. 

### Evaluate

```bash
bash eval_math_nodes.sh \
    --run_name verl-grpo_Qwen-2.5-32B_max_response8192_batch1024_rollout8_klcoef0.001_entcoef0.001_simplelr_math_35   \
    --init_model Qwen-2.5-32B \
    --template qwen-boxed  \
    --tp_size 8 \
    --add_step_0 true  \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks aime24,amc23,math500,olympiadbench,gsm8k,minerva_math \
    --n_sampling 1 
```

After running the script, the evaluation results will be saved in `$RUN_NAME/eval_results`

### Visualization

You can run the following script to access this tool:

```bash
# install gradio and httpx
pip install gradio
pip install httpx==0.23.0

bash launch_gradio.sh \
    --data_dir SimpleRL-verl/checkpoints \
    --run_names verl-grpo_Qwen-2.5-32B_max_response8192_batch1024_rollout8_klcoef0.001_entcoef0.001_simplelr_math_35  \
    --temperature 1.0   # temperature for evaluation
```





