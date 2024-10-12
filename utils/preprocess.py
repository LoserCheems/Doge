from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import argparse
import shutil

def load_tokenizer(tokenizer_path, add_eos_token=False):
    return AutoTokenizer.from_pretrained(tokenizer_path, add_eos_token=add_eos_token)

def preprocess_cosmopedia_v2_function(prompt, text):
    text = f"[INST]{prompt}[/INST]{text}"
    return tokenizer(prompt, text, max_length=2048, padding="max_length", truncation=True)

def preprocess_python_edu_function(text):
    return tokenizer(text, max_length=2048, padding="max_length", truncation=True)

def preprocess_fineweb_edu_dedup_function(text):
    return tokenizer(text, max_length=2048, padding="max_length", truncation=True)

def delete_cache(cache_dir):
    if cache_dir is not None:
        shutil.rmtree(cache_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--cosmopedia_v2_path", type=str, default="datasets/smollm-corpus/cosmopedia-v2")
    parser.add_argument("--python_edu_path", type=str, default="datasets/smollm-corpus/python-edu")
    parser.add_argument("--fineweb_edu_dedup_path", type=str, default="datasets/smollm-corpus/fineweb-edu-dedup")
    parser.add_argument("--output_path", type=str, default="datasets/smollm-corpus")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_proc", type=int, default=16)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path, add_eos_token=True)
    dataset = load_from_disk(args.cosmopedia_v2_path)
    dataset = dataset.map(preprocess_cosmopedia_v2_function, input_columns=["prompt", "text"], remove_columns=["prompt", "text", "token_length", "audience", "format", "seed_data"], num_proc=args.num_proc)
    dataset.save_to_disk(args.output_path + "/cosmopedia-v2_tmp", num_proc=args.num_proc)

    tokenizer = load_tokenizer(args.tokenizer_path, add_eos_token=False)
    dataset = load_from_disk(args.python_edu_path)
    dataset = dataset.map(preprocess_python_edu_function, input_columns=["text"], remove_columns=["text", "token_length", "audience", "format", "seed_data"], num_proc=args.num_proc)
    dataset.save_to_disk(args.output_path + "/python-edu_tmp", num_proc=args.num_proc)

    tokenizer = load_tokenizer(args.tokenizer_path, add_eos_token=False)
    dataset = load_from_disk(args.fineweb_edu_dedup_path)
    dataset = dataset.map(preprocess_fineweb_edu_dedup_function, input_columns=["text"], remove_columns=["text", "token_length", "audience", "format", "seed_data"], num_proc=args.num_proc)
    dataset.save_to_disk(args.output_path + "/fineweb-edu-dedup_tmp", num_proc=args.num_proc)

    cosmopedia_v2 = load_from_disk(args.output_path + "/cosmopedia-v2_tmp")
    python_edu = load_from_disk(args.output_path + "/python-edu_tmp")
    fineweb_edu_dedup = load_from_disk(args.output_path + "/fineweb-edu-dedup_tmp")
    dataset = concatenate_datasets([cosmopedia_v2['train'], python_edu['train'], fineweb_edu_dedup['train']])
    dataset = dataset.train_test_split(test_size=1000, shuffle=True, seed=233)
    dataset.save_to_disk(args.output_path + "/smollm-corpus", num_proc=args.num_proc)

    delete_cache(args.output_path + "/cosmopedia-v2_tmp")
    delete_cache(args.output_path + "/python-edu_tmp")
    delete_cache(args.output_path + "/fineweb-edu-dedup_tmp")
