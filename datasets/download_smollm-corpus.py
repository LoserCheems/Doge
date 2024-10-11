from datasets import load_dataset
from argparse import ArgumentParser
import boto3
import gzip
from botocore.exceptions import ClientError
import shutil
from transformers.utils import logging

logger = logging.get_logger(__name__)


s3 = boto3.client('s3')
bucket_name = "softwareheritage"

# python-edu 处理函数
def download_python_edu_contents(blob_id):
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

# 缓存删除函数
def delete_cache(cache_dir):
    if cache_dir is not None:
        shutil.rmtree(cache_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--save_dir", type=str, default="datasets")
    args = parser.parse_args()
    logger.info(
        "You'd better reserve 1T storage space, the dataset is really large..."
        "您最好预留 1T 的存储空间, 数据集真的很大..."
    )

    num_proc = args.num_proc

    # cosmopedia-v2
    dataset = load_dataset("HuggingfaceTB/smollm-corpus", "cosmopedia-v2", split="train", num_proc=num_proc)
    dataset.save_to_disk(args.save_dir + "/smollm-corpus/cosmopedia-v2", num_proc=num_proc) 
    dataset = None
    delete_cache(args.cache_dir)

    # python-edu
    dataset = load_dataset("HuggingfaceTB/smollm-corpus", "python-edu", split="train", num_proc=num_proc)
    dataset = dataset.map(download_python_edu_contents, input_columns="blob_id", num_proc=num_proc)
    dataset = dataset.filter(lambda x: x['download_success'])
    dataset.save_to_disk(args.save_dir + "/smollm-corpus/python-edu", num_proc=num_proc)
    dataset = None
    delete_cache(args.cache_dir)

    # fineweb-edu-dedup
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", num_proc=num_proc)
    dataset.save_to_disk(args.save_dir + "/smollm-corpus/fineweb-edu-dedup", num_proc=num_proc)
    dataset = None
    delete_cache(args.cache_dir)


