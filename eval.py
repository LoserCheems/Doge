import torch

import transformers
from transformers import AutoTokenizer

from models.configuration_doge import DogeConfig
from models.modeling_doge import DogeForCausalLM

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate


@register_model("doge")
class DogeEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    _DEFAULT_MAX_LENGTH = 8192

    def __init__(
        self, 
        pretrained="", 
        max_length=2048, 
        batch_size=None, 
        device="cpu"
    ):
        LM.__init__(self)
        # config = DogeConfig()
        self.pretrained = pretrained
        self._model = DogeForCausalLM.from_pretrained(pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(r"E:\cheems\Doge\tokenizer")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self.backend = "causal"
        self.add_bos_token = True
        self.logits_cache = True
        self.truncation = True
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size



if __name__ == "__main__":
    cli_evaluate()
    # lm_eval --model doge --model_args pretrained='./models/instruction' --tasks piqa --device cpu --batch_size 1
    # python eval.py --model doge --model_args pretrained='./models/instruction' --tasks lambada_openai --device cpu --batch_size 4 --output_path results