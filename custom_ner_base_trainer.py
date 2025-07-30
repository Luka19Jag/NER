import numpy as np
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoConfig,
    TrainerCallback
)
from datasets import Dataset
from seqeval.metrics import classification_report
from sklearn.metrics import accuracy_score

from process_input_file import ProcessFile

PROJECT_NAME="wikigold_ner"

class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.wait_count = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        score = metrics.get("eval_f1")
        if score is None:
            return control

        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count > self.patience:
                control.should_training_stop = True
                print(f"Early stopping: No improvement beyond min_delta: {self.min_delta}. Current score: {score}, Best score: {self.best_score}")
        
        return control

class BaseNERTrainer:
    def __init__(self, model_checkpoint: str, label_to_id_path: str, id_to_label_path: str, max_length=128):
        self.model_checkpoint = model_checkpoint
        self.label_to_id = {label:int(i) for label,i in ProcessFile.read_json(label_to_id_path).items()}
        self.id_to_label = {int(i):label for i, label in ProcessFile.read_json(id_to_label_path).items()}
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            add_prefix_space=True
        )
        config = AutoConfig.from_pretrained(
            model_checkpoint,
            num_labels=len(self.label_to_id),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        )      
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            config=config,
            ignore_mismatched_sizes=True
        )

    def tokenize_and_align_labels(self, sentences: list, labels: list) -> list:
        tokenized_inputs = self.tokenizer(
            sentences,
            truncation=True,
            is_split_into_words=True,
            padding=True,
            max_length=self.max_length
        )

        aligned_labels = []
        for i, label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[label[word_idx]])
                else:
                    label_ids.append(self.label_to_id[label[word_idx]] if label[word_idx].startswith("I-") else -100)
                previous_word_idx = word_idx
            aligned_labels.append(label_ids)
        
        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    def prepare_dataset(self, sentences: list, labels: list) -> Dataset:
        dataset = Dataset.from_dict({"tokens": sentences, "ner_tags": labels})
        return dataset.map(lambda x: self.tokenize_and_align_labels(x["tokens"], x["ner_tags"]), batched=True)

    def compute_metrics(self, p) -> dict:
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = [[self.id_to_label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.id_to_label[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]

        report = classification_report(true_labels, true_predictions, output_dict=True)
        true_labels_flat = [l for sublist in true_labels for l in sublist]
        true_preds_flat = [p for sublist in true_predictions for p in sublist]
        acc = accuracy_score(true_labels_flat, true_preds_flat)

        wandb.log({"report": report, "acc": acc})

        return {
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
            "accuracy": acc,
            "report": report
        }

    def train(self, train_dataset: Dataset, eval_dataset: Dataset, output_dir="./ner_model", 
              version=None, epochs=10, batch_size=16, seed=42) -> None:
        output_dir_version = f"{output_dir}/{self.model_checkpoint}/{version}" if version is not None else f"{output_dir}/{self.model_checkpoint}"
        model_checkpoint = f"{self.model_checkpoint}_{version}" if version is not None else self.model_checkpoint

        wandb.init(project=PROJECT_NAME, name=model_checkpoint)

        training_args = TrainingArguments(
            output_dir=output_dir_version,
            report_to="wandb",
            run_name=model_checkpoint,
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            seed=seed,
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_steps=50,
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[CustomEarlyStoppingCallback(patience=1, min_delta=0.01)]
        )

        trainer.train()
        trainer.save_model(f"{output_dir_version}/best_model")
