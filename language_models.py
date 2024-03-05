import openai
import os
import time
import torch
import gc
from typing import Dict, List

import fastchat
from vllm import LLM, SamplingParams


def get_compute_capability():
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this device!")

    capability_str = torch.cuda.get_device_capability()
    capability = float(f"{capability_str[0]}.{capability_str[1]}")
    return capability


def check_bf16_support():
    capability = get_compute_capability()
    if capability >= 8.0:
        return True
    return False


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(
        self, prompts_list: List, max_n_tokens: int, temperature: float
    ):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError


class HuggingFace(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(
        self,
        full_prompts_list,
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        inputs = self.tokenizer(full_prompts_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1,  # To prevent warning messages
            )

        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1] :]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to("cpu")
        output_ids.to("cpu")
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([self.tokenizer.encode("}")[1], 29913, 9092, 16675])


class vllmModel(LanguageModel):
    def __init__(self, model_name, tensor_parallel_size=None):
        if check_bf16_support():
            dtype = "bfloat16"
        else:
            dtype = "float16"

        if tensor_parallel_size is None:
            gpu_count = torch.cuda.device_count()
            tensor_parallel_size = gpu_count
        else:
            tensor_parallel_size = tensor_parallel_size

        self.model_name = model_name
        self.model = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=0.5,
        )

        conv_template = self.get_conv_template()
        self.stop_str = conv_template.stop_str
        self.stop_token_ids = conv_template.stop_token_ids

    def get_conv_template(self):
        conv_template = fastchat.model.get_conversation_template(self.model_name)
        return conv_template

    def batched_generate(
        self,
        prompts_list: List,
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        """
        Generates responses for a batch of prompts using a language model.
        """
        sampling_params = self.load_generation_config(max_n_tokens, temperature, top_p)
        responses = self.model.generate(prompts_list, sampling_params)
        responses = [output.outputs[0].text for output in responses]
        responses = self.post_process(responses)

        return responses

    def load_generation_config(self, max_new_tokens, temperature, top_p):
        self.generation_config = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            stop=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )
        return self.generation_config

    def post_process(self, responses: List[str]):
        return [i.strip() for i in responses]


class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(
        self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float
    ):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    engine=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    request_timeout=self.API_TIMEOUT,
                )
                output = response["choices"][0]["message"]["content"]
                break
            except openai.error.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]
