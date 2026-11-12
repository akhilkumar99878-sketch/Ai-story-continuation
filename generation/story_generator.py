import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
GENRES = {
    "Fantasy":"a vivid fantasy tale with wonder and magic",
    "Dark Fantasy":"an atmospheric dark fantasy tale with eerie suspense",
    "Mystery":"a suspenseful mystery with clues and unanswered questions",
    "Horror":"an unsettling horror tale that builds tension gradually",
    "Romance":"a warm romantic tale focused on believable emotions",
    "Science Fiction":"a science-fiction tale with imaginative technology",
    "Adventure":"an exciting adventure tale with discovery and action",
    "General":"a coherent and engaging piece of fiction",
}

class StoryGenerator:
    def __init__(self, model_name=MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, genre="General", max_new_tokens=100, temperature=0.75):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A story prompt is required.")
        if len(prompt) > 1800:
            raise ValueError("Prompt must be 1,800 characters or fewer.")
        max_new_tokens = max(30, min(int(max_new_tokens), 180))
        temperature = max(0.3, min(float(temperature), 1.2))
        style = GENRES.get(genre, GENRES["General"])
        context = (f"Genre: {genre}. Write {style}. Continue the following story naturally. "
                   "Keep the same characters, setting, and point of view. Avoid unrelated events.\n\n"
                   f"Story:\n{prompt.rstrip()}")
        tokens = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=850)
        tokens = {k:v.to(self.device) for k,v in tokens.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **tokens, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_k=40, top_p=0.90,
                repetition_penalty=1.25, no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        new_tokens = output[0][tokens["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return re.sub(r"\\s+", " ", text).strip()
