import os
from typing import Dict, List

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.generator.generator import Generator
import torch
import torch.nn.functional as F
from transformers.utils import ModelOutput


class HuggingFaceGenerator(Generator):
    def __init__(
        self,
        model: str,
        model_type: str,
        tokenizer: str,
        use_fast: bool = False,
        change_pad_token: bool = False,
        adapter: str = None,
        cache_dir: str = None,
        device: str = None,
        use_accelerate: bool = False,
    ):
        self.__model_name = model
        self.__tokenizer_name = tokenizer
        self.__use_fast = use_fast
        self.__cache_dir = cache_dir
        self.__model_type = self.__load_model_type(model_type=model_type)
        self.__device = device
        self.__use_accelerate = use_accelerate
        self.__adapter_name = adapter
        self.__change_pad_token = change_pad_token

        self.__model_args = self.__get_model_args()
        self.__tokenizer_args = self.__get_tokenizer_args()

        self.tokenizer = self.__load_tokenizer()
        self.model = self.__load_model()

        if os.path.isdir(model):
            self.model_name = model.split("/")[-1]
        else:
            self.model_name = self.model.name_or_path.split("/")[-1]

    def __load_model_type(self, model_type: str):
        if model_type == "AutoModelForCausalLM":
            return AutoModelForCausalLM
        elif model_type == "AutoModelForSeq2SeqLM":
            return AutoModelForSeq2SeqLM
        else:
            raise ValueError(f'Invalid model type "{model_type}"passed.')

    def __get_model_args(self) -> Dict[str, str]:
        model_args = {
            "pretrained_model_name_or_path": self.__model_name,
            # "torch_dtype": torch.float16,
        }

        if "falcon" in self.__model_name:
            model_args["trust_remote_code"] = True

        if self.__use_accelerate:
            model_args["device_map"] = "auto"

        if self.__cache_dir:
            model_args["cache_dir"] = self.__cache_dir

        return model_args

    def __get_tokenizer_args(self) -> Dict[str, str]:
        tokenizer_args = {
            "pretrained_model_name_or_path": self.__tokenizer_name,
            "use_fast": self.__use_fast,
        }

        if self.__cache_dir:
            tokenizer_args["cache_dir"] = self.__cache_dir

        return tokenizer_args

    def __load_model(self):
        try:
            model = self.__model_type.from_pretrained(**self.__model_args)
        except:
            raise ValueError(
                f'The passed model type: "{self.__model_type.__name__}" '
                f'is not suitable for the model "{self.__model_name}".'
            )

        if self.__adapter_name:
            model = PeftModel.from_pretrained(model=model, model_id=self.__adapter_name)

        if self.__device:
            if self.__use_accelerate:
                # os.environ["CUDA_VISIBLE_DEVICES"] = self.__device
                # print("Using CUDA devices:", os.environ["CUDA_VISIBLE_DEVICES"])
                pass
            else:
                print("loading model to", self.__device)
                model = model.to(self.__device)
                print("Using CUDA devices:", self.__device)

        return model

    def __load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(**self.__tokenizer_args)

        if self.__change_pad_token or self.__adapter_name:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def generate(
        self,
        texts: List[str],
        max_new_tokens: int = 10,
        split_lines: bool = True,
        temperature: float = 0,
        frequency_penalty: float = 2.0,
        presence_penalty: float = 2.0,
    ) -> List[str]:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        if self.__device and not self.__use_accelerate:
            input_ids = inputs.input_ids.to(self.__device)
            attention_mask = inputs.attention_mask.to(self.__device)
        else:
            input_ids = inputs.input_ids.to(0)
            attention_mask = inputs.attention_mask.to(0)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_ids = []
        # below post-processing taken as-is from
        # https://github.com/huggingface/trl/blob/3ef21a24e7df53d0e0e6fe26b448b94ed3ec7cda/trl/environment/base_environment.py#L459
        for generation, mask in zip(generated_ids, attention_mask):
            if not self.model.config.is_encoder_decoder:
                output = generation[(1 - mask).sum() :]  # remove padding
            else:
                output = generation

            if not self.model.config.is_encoder_decoder:
                output = output[(mask).sum() :]  # remove prompt

            new_ids.append(output)

        responses = self.tokenizer.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        responses = [response.strip() for response in responses]
        if split_lines:
            responses_post = []
            for response in responses:
                if response:
                    responses_post.append(response.splitlines()[0])
                else:
                    responses_post.append(response)

            responses = responses_post

        return responses

    def vectorize(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        if self.__device and not self.__use_accelerate:
            input_ids = inputs.input_ids.to(self.__device)
            attention_mask = inputs.attention_mask.to(self.__device)
        else:
            input_ids = inputs.input_ids.to(0)
            attention_mask = inputs.attention_mask.to(0)

        with torch.no_grad():
            if self.model.config.is_encoder_decoder:
                model_output = self.model.encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            else:
                model_output = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
        sentence_embeddings = self.__mean_pooling(model_output, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        result = self.__turn_into_list_of_floats(sentence_embeddings)
        return result

    def __mean_pooling(self, model_output: ModelOutput, attention_mask: torch.Tensor):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def __turn_into_list_of_floats(self, sentence_embeddings: torch.Tensor):
        list_of_tensors = list(sentence_embeddings)
        result = []
        for vector in list_of_tensors:
            result.append([value.item() for value in vector])
        return result
