# Doge

<h4 align="center">
<p>

English | [简体中文](./README_zh.md)

</p>
</h4>

![Doge](./assets/doge_architecture.png)
> **Wonderful Matrices: More Efficient and Effective Architecture for Language Modeling Tasks**\
> Jingze Shi*, Bingheng Wu*, Lu He*, Luchang Jiang*\
> Paper: [arXiv:2407.16958](https://arxiv.org/abs/2407.16958)

## About

This project is a continuation of the discussion section of the [Wonderful Matrices](https://arxiv.org/abs/2407.16958) paper.

We hope to further explore whether the Transformer framework allows for more complex feedforward network structures by training a small language model (SLM) based on the `Doge` architecture, enabling the model to have fewer cache states and larger knowledge capacity.

We also hope to use open-source tools and frameworks as much as possible to simplify the process from data processing to model training, so that beginners can easily understand and use them.


## Requirements

- Windows or Linux
- NVIDIA GPU
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+

We highly recommend that you install the latest version of PyTorch and CUDA for optimal performance.

Of course, you can also use the open-source [Docker PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) image to avoid the hassle of configuring the environment.

```bash
docker pull nvcr.io/nvidia/pytorch:24.10-py3
docker run --privileged --gpus all -it --name PyTorch --shm-size=32g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 -v <your code path>:/workspace -v <your datasets path>:/workspace/Doge/datasets nvcr.io/nvidia/pytorch:24.10-py3
```

- `pip install transformers`: The core framework for all subsequent work.
- `pip install datasets sentencepiece boto3`: Used to download and process datasets.
- `pip install accelerate`: Used for distributed training.
- `pip install einx`: Fast implementation dependency for the CDMoE module.


## Usage

We have written a [notebook](./notebook.ipynb) (still being updated) to demonstrate the entire process of datasets processing, model training, and model evaluation. Of course, you can also use some of the following modules independently.

### Inner Function Attention

The sequence transformation module of the Doge model.

Source code: [model/modules/innerfuncattn.py](./model/modules/innerfuncattn.py)

Usage:

```python
import torch
from model.modules.innerfuncattn import InnerFuncAttn

batch, seq_len, dim = 2, 16, 64
x = torch.rand(batch, seq_len, dim)
attention_mask = torch.ones(batch, seq_len)
attn = InnerFuncAttn(
    d_model=dim,
    n_heads=1,
    n_inner_values=1,
    d_inner_values_retrieval=64,
    max_position_embeddings=seq_len,
    layer_idx=0,
)
y, past_key_values = attn(x, attention_mask)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
```

### CDMoE

The state transformation module of the Doge model.

Source code: [model/modules/cdmoe.py](./model/modules/cdmoe.py)

Usage:

```python
import torch
from model.modules.cdmoe import CDMoE

batch, seq_len, dim = 2, 16, 64
x = torch.rand(batch, seq_len, dim)
cdmoe = CDMoE(
    d_model=dim,
    act_fn="silu",
    d_ff=dim * 4,
    d_private_expert_retrieval=64,
    n_experts=64,
    n_experts_heads=1,
    n_experts_per_head=2,
)
y = cdmoe(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
```

## Citation

If you use this codebase, or otherwise find our work valuable, please cite Doge:

```bibtex
@misc{shi2024wonderfulmatrices,
      title={Wonderful Matrices: More Efficient and Effective Architecture for Language Modeling Tasks}, 
      author={Jingze Shi and Bingheng Wu and Lu He and Luchang Jiang},
      year={2024},
      eprint={2407.16958},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.16958}, 
}
```




