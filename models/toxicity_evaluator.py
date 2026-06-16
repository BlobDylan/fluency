import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import consts

class ToxicityEvaluator:
    def __init__(self, weights_dir: str = consts.TOXICITY_MODEL_PATH):
        """
        Initializes the Toxicity Evaluator entirely offline from a local directory.
        Forces the model onto the Apple Silicon MPS backend and freezes all parameters.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir, local_files_only=True)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            weights_dir,
            local_files_only=True,
            torch_dtype=torch.float16
        ).to("mps")
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def evaluate_text(self, text: str) -> float:
        """
        Takes an input string and returns a continuous toxicity probability between 0.0 and 1.0.
        """

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to("mps")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits).squeeze()
            
        if probabilities.dim() > 0:
            toxic_score = probabilities[0].item() 
        else:
            toxic_score = probabilities.item()
            
        return float(toxic_score)