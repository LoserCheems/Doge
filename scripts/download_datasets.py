import os
import boto3
import gzip
from datasets import load_dataset
from botocore.exceptions import ClientError
from argparse import ArgumentParser
import hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

s3 = boto3.client('s3')
bucket_name = "softwareheritage"

def download_contents(blob_id):
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=obj['Body']) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
        return {"text": content, "download_success": True}
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"File not found: {key}")
            return {"text": "", "download_success": False}
        else:
            raise



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()

    # # 下载fineweb-edu数据集
    # # Download fineweb-edu dataset
    # dataset = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", num_proc=args.num_proc, cache_dir=args.cache_dir)
    # print(dataset)
    # dataset.save_to_disk(args.save_dir + "/fineweb-edu", num_proc=args.num_proc)

    # # 下载宇宙百科v2数据集
    # # Download Cosmopedia-v2 dataset
    # dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", num_proc=args.num_proc, cache_dir=args.cache_dir)
    # print(dataset)
    # dataset.save_to_disk(args.save_dir + "/cosmopedia-v2", num_proc=args.num_proc)

    # # 下载中文宇宙百科数据集
    # # Download Chinese Cosmopedia dataset
    # dataset = load_dataset("opencsg/chinese-cosmopedia", split="train", num_proc=args.num_proc, cache_dir=args.cache_dir)
    # print(dataset)
    # dataset.save_to_disk(args.save_dir + "/chinese-cosmopedia", num_proc=args.num_proc)

    # # 下载Python教育数据集
    # # Download Python Education dataset
    # dataset = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", num_proc=args.num_proc, cache_dir=args.cache_dir)
    # dataset = dataset.map(download_contents, input_columns="blob_id", num_proc=args.num_proc)
    # dataset = dataset.filter(lambda x: x['download_success'])
    # print(dataset)
    # dataset.save_to_disk(args.save_dir + "/python-edu", num_proc=args.num_proc)

    # # 下载无限指令数据集
    # # Download Infinity Instruct dataset
    # dataset = load_dataset('BAAI/Infinity-Instruct', '0625', split='train', num_proc=args.num_proc, cache_dir=args.cache_dir)
    # print(dataset)
    # dataset.save_to_disk(args.save_dir + "/infinity-instruct-0625", num_proc=args.num_proc)

    # dataset = load_dataset('BAAI/Infinity-Instruct', '7M', split='train', num_proc=args.num_proc, cache_dir=args.cache_dir)
    # print(dataset)
    # dataset.save_to_disk(args.save_dir + "/infinity-instruct-7M", num_proc=args.num_proc)

    # dataset = load_dataset('BAAI/Infinity-Instruct', 'Gen', split='train', num_proc=args.num_proc, cache_dir=args.cache_dir)
    # print(dataset)
    # dataset.save_to_disk(args.save_dir + "/infinity-instruct-Gen", num_proc=args.num_proc)