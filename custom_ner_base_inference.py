import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForTokenClassification
from process_input_file import ProcessFile

class BaseNERInference:
    def __init__(self, model_dir: str, label_to_id_path: str, id_to_label_path: str, max_length=256, device=None):
        self.model_dir = model_dir
        self.label_to_id = {label: int(i) for label, i in ProcessFile.read_json(label_to_id_path).items()}
        self.id_to_label = {int(i): label for i, label in ProcessFile.read_json(id_to_label_path).items()}
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sentence: list) -> list:
        inputs = self.tokenizer(
            sentence,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs).logits

        predictions = torch.argmax(outputs, dim=2)[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_ids = inputs.word_ids(batch_index=0)

        word_preds = []
        seen = set()
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            token = sentence[word_id]
            label_id = predictions[idx]
            label = self.id_to_label.get(label_id, "O")
            word_preds.append((token, label))

        return word_preds