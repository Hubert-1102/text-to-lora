import sys
import os
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.generate_lora import generate_save
from peft import PeftModel


def load_lora_from_desc(model:PeftModel, model_str, desc):
    """
    generate lora from task description and load it to model
    - model: the LLM instance (PeftModel) need to be loaded adapter
    - model_path: the t2l model path(e.g. text-to-lora/trained_t2l/mistral_7b_t2l) 
    - desc: the task description text
    """
    lora_dir = generate_save(model_str, desc)
    model.load_adapter(lora_dir , desc)
    return model


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model,"trained_t2l/mistral_7b_t2l/extras/user_generated/20250618_083634_bHw50WxB","marco")
    load_lora_from_desc(model,'trained_t2l/mistral_7b_t2l','math')
