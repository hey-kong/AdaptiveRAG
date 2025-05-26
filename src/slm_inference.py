import time

import torch
import torch.serialization
from transformers import AutoTokenizer, AutoModelForCausalLM

from customed_statistic import global_statistic

QWEN_PROMPT_PREFIX = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
"""

LLAMA_PROMPT_PREFIX = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
"""


def get_model_type(model_name):
    model_lower = model_name.lower()
    if 'qwen3' in model_lower:
        return 'qwen3'
    elif 'llama' in model_lower:
        return 'llama'
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def _build_suffix(model_type):
    suffix_map = {
        'qwen3': (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n\n"
        ),
        'llama': (
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    }
    return suffix_map[model_type]


def judge_complexity_prompt(query, model_name):
    model_type = get_model_type(model_name)
    prefix = QWEN_PROMPT_PREFIX if model_type == 'qwen3' else LLAMA_PROMPT_PREFIX

    return (
        f"{prefix}For the given question: {query}\n\n"
        f"Classify the question as easy or hard to answer.\n"
        f"If the question is simple, factual, or straightforward, respond with \"Easy\".\n"
        f"If the question is complex, nuanced, requires multi-step reasoning or in-depth analysis, respond with \"Hard\".\n\n"
        f"Respond with \"Easy\" or \"Hard\" only, do not output any other words."
        f"{_build_suffix(model_type)}"
    )


def query_prompt(chunk_list, query, model_name):
    model_type = get_model_type(model_name)
    prefix = QWEN_PROMPT_PREFIX if model_type == 'qwen3' else LLAMA_PROMPT_PREFIX
    chunks = "\n\n".join(chunk_list)

    return (
        f"{prefix}{chunks}\n\n"
        f"Given the above context, answer the question: {query}\n\n"
        f"Only give me the answer and do not output any other words."
        f"{_build_suffix(model_type)}"
    )


class CustomModelWrapper:
    def init(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.model.config.eos_token_id
        self.model_type = get_model_type(model_path)

    def judge_complexity(self, query):
        start = time.perf_counter()
        prompt = judge_complexity_prompt(query, self.model_type)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False
            )
        generated_ids = outputs[0]
        input_length = inputs.input_ids.shape[1]
        difficulty_label = self.tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()
        end = time.perf_counter()
        global_statistic.add_to_list("judge_complexity_time", end - start)
        return difficulty_label

    def generate_answer(self, query, nodes):
        chunk_list = [node.text for node in nodes]
        prompt = query_prompt(chunk_list, query, self.model_type)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        generated_ids = outputs[0]
        input_length = inputs.input_ids.shape[1]
        answer = self.tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()
        end = time.perf_counter()
        global_statistic.add_to_list("slm_generate_time", end - start)
        return answer


slm = CustomModelWrapper()
