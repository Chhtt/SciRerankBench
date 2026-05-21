"""
LLM Model Factory for SciRerankBench

Supports multiple LLM backends:
- vLLM (local GPU serving)
- OpenAI-compatible API (for OpenAI, vLLM server, etc.)

Usage:
    from llm_model import LLMModelFactory
    factory = LLMModelFactory()
    model = factory.get_model("Qwen-72B")
    answer = model.generate("What is X?")
"""

import os


class LLMModelFactory:
    """Factory for creating LLM model instances."""

    def __init__(self, api_base=None, api_key=None):
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

    def get_model(self, model_name, **kwargs):
        """Get an LLM model by name."""
        if "Qwen" in model_name or "Llama" in model_name or "Mistral" in model_name:
            return VLLMModel(model_name, api_base=self.api_base, api_key=self.api_key, **kwargs)
        elif "gpt" in model_name.lower():
            return OpenAIModel(model_name, api_key=self.api_key, **kwargs)
        else:
            raise ValueError(f"Unknown LLM: {model_name}")


class BaseLLM:
    """Base class for LLM models."""

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        raise NotImplementedError


class VLLMModel(BaseLLM):
    """vLLM-compatible model (OpenAI API interface)."""

    MODEL_PATH_MAP = {
        "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
        "Qwen-14B": "Qwen/Qwen2.5-14B-Instruct",
        "Qwen-72B": "Qwen/Qwen2.5-72B-Instruct",
        "Llama2-7B": "meta-llama/Llama-2-7b-chat-hf",
        "Llama2-13B": "meta-llama/Llama-2-13b-chat-hf",
        "Llama2-70B": "meta-llama/Llama-2-70b-chat-hf",
        "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    }

    def __init__(self, model_name, api_base="http://localhost:8000/v1", api_key="EMPTY", **kwargs):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.model_path = self.MODEL_PATH_MAP.get(model_name, model_name)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


class OpenAIModel(BaseLLM):
    """OpenAI API model."""

    def __init__(self, model_name, api_key=None, **kwargs):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    factory = LLMModelFactory()
    model = factory.get_model("Qwen-72B")
    print(model.generate("What is retrieval-augmented generation?"))
