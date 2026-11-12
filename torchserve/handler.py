import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ts.torch_handler.base_handler import BaseHandler

class StoryHandler(BaseHandler):
    def initialize(self, context):
        props = context.system_properties
        gpu_id = props.get("gpu_id")
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu")
        model_dir = props.get("model_dir")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        body = data[0].get("body") or data[0].get("data")
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            body = json.loads(body)
        prompt = body.get("prompt", "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        if len(prompt) > 1800:
            raise ValueError("prompt must be 1800 characters or fewer")
        return prompt, body.get("genre","General"), max(30,min(int(body.get("max_new_tokens",100)),180)), max(0.3,min(float(body.get("temperature",0.75)),1.2))

    def inference(self, data):
        prompt, genre, length, temperature = data
        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=850)
        tokens = {k:v.to(self.device) for k,v in tokens.items()}
        with torch.inference_mode():
            output = self.model.generate(**tokens, max_new_tokens=length, do_sample=True,
                temperature=temperature, top_k=40, top_p=0.90, repetition_penalty=1.25,
                no_repeat_ngram_size=3, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def postprocess(self, data):
        return [{"generated_text": data}]
