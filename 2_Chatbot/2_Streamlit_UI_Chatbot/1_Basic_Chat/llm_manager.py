import torch
import warnings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from typing import Optional


class LLMManager:
    """
    A singleton-like class to initialize and manage the Mistral LLM
    using 4-bit quantization, HuggingFace, and LangChain wrappers.

    The model is loaded only once upon instantiation.
    """

    # Class variable to hold the initialized LangChain Chat Model
    _chat_model: Optional[ChatHuggingFace] = None

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initializes the model, tokenizer, and LangChain wrappers.
        """
        if LLMManager._chat_model is not None:
            warnings.warn("LLMManager is already initialized. Returning existing instance.")
            return

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This model configuration requires a GPU (device_map='cuda')."
            )

        print(f"Initializing LLM: {model_id}...")

        # 1. Quantization Configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 2. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # 3. Load Model
        model_load = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",  # Use only GPU
            quantization_config=bnb_config,  # 4-bit quantization
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )

        # 4. Build Pipeline
        pipe = pipeline(
            task="text-generation",
            model=model_load,
            tokenizer=tokenizer,
            max_new_tokens=256,  # Increased token limit for better response depth
            do_sample=True,
            temperature=0.7,
            use_cache=True
        )

        # 5. Wrap in LangChain HuggingFacePipeline
        llm_hf_pipeline = HuggingFacePipeline(pipeline=pipe)

        # 6. Final LangChain Chat Model Instance
        LLMManager._chat_model = ChatHuggingFace(llm=llm_hf_pipeline)

        print(f"LLM initialization complete for {model_id}.")

    def get_chat_model(self) -> ChatHuggingFace:
        """
        Returns the initialized LangChain ChatHuggingFace instance.
        """
        if LLMManager._chat_model is None:
            raise RuntimeError("LLMManager has not been initialized. Call LLMManager() first.")
        return LLMManager._chat_model


# Helper function for easy access in other modules
def get_llm_instance() -> ChatHuggingFace:
    """
    Ensures the LLMManager is initialized and returns the chat model.
    Use this function throughout your repository.
    """
    # Initialize only if it hasn't been done yet
    if LLMManager._chat_model is None:
        LLMManager()
    return LLMManager().get_chat_model()

# Example: If you want to initialize it immediately on import
# LLMManager()