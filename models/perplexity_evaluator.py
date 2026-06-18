import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import consts

class PerplexityEvaluator:
    def __init__(self, weights_dir: str = consts.FLUENCY_MODEL_PATH):
        """
        Initializes the GPT-2 Perplexity Evaluator completely offline.
        Forces execution onto Apple Silicon MPS and freezes all layers.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir, local_files_only=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            weights_dir,
            local_files_only=True,
            dtype=consts.DTYPE
        ).to(consts.DEVICE)
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def evaluate_fluency(self, text: str) -> float:
        """
        Calculates the perplexity of the input string.
        Lower scores mean more natural, fluent English. 
        Gibberish tokens will return massive scores.
        """

        text = text.strip()
        if not text:
            return 0.0

        inputs = self.tokenizer(text, return_tensors="pt").to(consts.DEVICE)
        input_ids = inputs["input_ids"]

        if input_ids.shape[1] < 2:
            return 0.0

        with torch.no_grad():
            outputs = self.model(**inputs, labels=input_ids)
            loss = outputs.loss.item()

        try:
            perplexity = math.exp(loss)
        except OverflowError:
            perplexity = float("inf")

        return float(perplexity)