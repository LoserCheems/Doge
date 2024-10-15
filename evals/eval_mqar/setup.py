from setuptools import setup


_REQUIRED = [
    "numpy",
    "einops",
    "tqdm",
    # TODO: remove this upper bound 
    # currently, when using ray we get:
    # "AttributeError: module 'pydantic._internal' has no attribute '_model_construction'"
    "pandas",
    "seaborn",
    "matplotlib",
    "rich",
    "ray",
    "PyYAML",
    "pydantic>=2.0.0,<2.5.0",
    "wandb",
]


# ensure that torch is installed, and send to torch website if not
try:
    import torch
except ModuleNotFoundError:
    raise ValueError("Please install torch first: https://pytorch.org/get-started/locally/")


setup(
    name="eval_mqar",
    version="0.0.1",
    description="",
    author="jingze shi",
    packages=["eval_mqar"],
    install_requires=_REQUIRED,
    entry_points={
        'console_scripts': ['eval_mqar=eval_mqar.cli:cli'],
    },
)