import os
import pandas as pd
import argparse
from tqdm import tqdm
import multiprocessing as mp
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_model', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--gpu_num', type=int, default='1')
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--k_steps', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--max_tokens_per_call', type=int, default=4096)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--total_epochs', type=int, required=True)
    parser.add_argument('--use_k_ratio', type=float, default=0.3)
    return parser.parse_args()

def get_k_step(output, k, delimiter="\n"): 
    # 先去除多余的'\n'再进行分割 
    output = re.sub(r'\n+', '\n', output) 
    parts = output.split(delimiter) 
    truncated_parts = parts[:k] 
    return delimiter.join(truncated_parts)


def main():
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    df = pd.read_parquet(args.train_file)
    print("=" * 20 + "\toriginal sample\t" + "=" * 20)
    print(df.iloc[0].to_dict())


    total_training_steps = (len(df) // args.train_batch_size) * args.total_epochs
    k_ratio = int(total_training_steps * args.use_k_ratio)
    assert not (args.k_steps > k_ratio), \
        f"k_steps={args.k_steps} 不应大于 total_training_steps * use_k_ratio={k_ratio}"
    assert not (args.use_k_ratio > 1), \
            f"use_k_ratio 不应大于 1"


    os.environ["VLLM_LOG_LEVEL"] = "error"
    from vllm import LLM, SamplingParams
    
    original_prompts = []
    for ind in range(len(df)):
        original_prompts.append(df.iloc[ind]["prompt"][0]["content"]) 


    llm = LLM(
        model=args.prompt_model,
        tensor_parallel_size=args.gpu_num,
    )

    outputs = llm.generate(
                original_prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                ),
            )

    # 构造 prompt
    df_new = df.copy()
    for ind in range(len(df)):

        
        content = df_new.at[ind, "prompt"][0]["content"]
        question = df_new.at[ind, "question"]
        extra_info = df_new.at[ind, "extra_info"]["question"]
        k_text = get_k_step(outputs[ind].outputs[0].text, args.k_steps)

        new_content = content + '\n' + k_text + '\n'
        new_question = question + '\n' + k_text + '\n'
        new_extra_info = extra_info + '\n' + k_text + '\n'

        # 去除多余的'\n'
        new_content = re.sub(r'\n+', '\n', new_content)
        new_question = re.sub(r'\n+', '\n', new_question)
        new_extra_info = re.sub(r'\n+', '\n', new_extra_info)

        df_new.at[ind, "prompt"][0]["content"] = new_content
        df_new.at[ind, "question"] = new_question
        df_new.at[ind, "extra_info"]["question"] = new_extra_info



    print("=" * 20 + "\tnew sample\t" + "=" * 20)
    print(df_new.iloc[0].to_dict())
    

    # 保存新数据
    os.makedirs(args.output_folder, exist_ok=True)
    file_path = f"{args.output_folder}/train_kStep{args.k_steps}_len{len(df_new)}_tem{args.temperature}_topP{args.top_p}_topK{args.top_k}.parquet"
    df_new.to_parquet(file_path, index=False)
    print(f"Saved new prompts to {file_path}")


if __name__ == "__main__":
    main()
