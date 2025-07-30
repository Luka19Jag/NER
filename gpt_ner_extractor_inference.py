import time
import random
import logging
logging.basicConfig(level=logging.INFO)

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

from prepare_training_ner_data import ProcessFile
from gpt_ner_extractor_base import GPTNERExtractor
from gpt_base import PromptType
from few_shot_ner_sample import EXAMPLE

class EvaluateGPTNERExtractor:

    def __init__(self, model: GPTNERExtractor, llm_method=PromptType.ONE_SHOT,):
        self.model = model
        self.llm_method = llm_method

    def run_ner_gpt(self, sentences_file_path: str, labels_file_path: list,
                    limit_documents=None, to_save_stats="inference_data", to_save_predictions="ner_model", run_parallel=True) -> dict:
        sentences = ProcessFile.read_json(sentences_file_path)
        labels = ProcessFile.read_json(labels_file_path)
        dic_labels = EvaluateGPTNERExtractor.ner_tags_to_dict(sentences, labels)
        limit_documents = len(sentences) if limit_documents is None else limit_documents
        zipped_data = zip(sentences[:limit_documents], labels[:limit_documents], dic_labels[:limit_documents])

        if run_parallel:
            dic_stats, all_predictions, all_errors = self.run_parallel(zipped_data)
        else:
            dic_stats, all_predictions, all_errors = self.run_sequential(zipped_data)
            
        dic_stats_final = self.compute_metrics(dic_stats)
        
        ProcessFile.save_json(f"{to_save_predictions}/{self.model.model}/test_predictions.json", all_predictions)
        ProcessFile.save_json(f"{to_save_predictions}/{self.model.model}/test_errors.json", all_errors)
        ProcessFile.save_json(f"{to_save_stats}/{self.model.model}/test_statistics.json", dic_stats_final)

    def run_parallel(self, zipped_data: zip, max_workers=5) -> Tuple:
        all_predictions, all_errors = [], []
        dic_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.process_sample, sentence, label, dic_label)
                for sentence, label, dic_label in zipped_data
            ]

        for future in as_completed(futures):
            prediction = future.result()
            all_predictions.append(prediction)
            if prediction.get("dic_one_sample_stats", None) is not None:
                self.add_local_to_global_stats(dic_stats, prediction["dic_one_sample_stats"])
            else:
                all_errors.append(prediction)
        
        return (dic_stats, all_predictions, all_errors)

    def run_sequential(self, zipped_data: zip) -> Tuple:
        all_predictions, all_errors = [], []
        dic_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})

        for sentence, label, dic_label in zipped_data:
            prediction = self.process_sample(sentence, label, dic_label)
            all_predictions.append(prediction)
            if prediction.get("dic_one_sample_stats", None) is not None:
                self.add_local_to_global_stats(dic_stats, prediction["dic_one_sample_stats"])
            else:
                all_errors.append(prediction)
        
        return (dic_stats, all_predictions, all_errors)

    def ner_tags_to_dict(sentences: list, labels: list) -> list:
        all_results = []

        for sentence, label_seq in zip(sentences, labels):
            result = defaultdict(list)
            current_entity = []
            current_tag = None

            for token, tag in zip(sentence, label_seq):
                if tag == "O":
                    if current_entity:
                        result[current_tag].append(" ".join(current_entity))
                        current_entity = []
                        current_tag = None
                    continue

                prefix, label = tag.split("-", 1) if "-" in tag else (None, tag)

                if prefix == "B":
                    if current_entity:
                        result[current_tag].append(" ".join(current_entity))
                    current_entity = [token]
                    current_tag = tag 
                elif prefix == "I" and tag == current_tag:
                    current_entity.append(token)
                else:
                    if current_entity:
                        result[current_tag].append(" ".join(current_entity))
                    current_entity = [token]
                    current_tag = tag

            if current_entity:
                result[current_tag].append(" ".join(current_entity))

            all_results.append(dict(result))
        
        return all_results
    
    def process_sample(self, sentence: list, label: list, dic_label: dict) -> dict:
        time.sleep(random.uniform(0.01, 3))
        
        prompt = self.model.get_prompt(sentence, self.llm_method)
        if prompt is None:
            return dict(
                sentence=sentence,
                label=label,
                dic_label=dic_label,
                gpt_text=None,
                dic_one_sample_stats=None
            )

        response, raw_text = self.model.request(prompt)

        gpt_text, dic_one_sample_stats = None, None
        if raw_text is not None:
            gpt_text = self.model.parse_output(raw_text)
            dic_one_sample_stats = self.evaluate_one_sample(gpt_text, dic_label)
            logging.info(f"\n\nInput:\n{sentence}\nRaw output:\n{raw_text}\nOutput:\n{gpt_text}\n\n")
        
        return dict(
            sentence=sentence,
            label=label,
            dic_label=dic_label,
            raw_text=raw_text,
            gpt_text=gpt_text,
            dic_one_sample_stats=dic_one_sample_stats
        )
    
    def evaluate_one_sample(self, dic_pred: dict, dic_gold: dict) -> dict:
        all_labels = set(dic_gold.keys()) | set(dic_pred.keys())

        gold_set = set((label, ent.strip().lower()) for label, ents in dic_gold.items() for ent in ents)
        pred_set = set((label, ent.strip().lower()) for label, ents in dic_pred.items() for ent in ents)

        dic_one_sample_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
        for label in all_labels:
            gold_ents = {ent for l, ent in gold_set if l == label}
            pred_ents = {ent for l, ent in pred_set if l == label}

            tp = len(gold_ents & pred_ents)
            fp = len(pred_ents - gold_ents)
            fn = len(gold_ents - pred_ents)
            tn = 0 

            dic_one_sample_stats[label]["TP"] += tp
            dic_one_sample_stats[label]["FP"] += fp
            dic_one_sample_stats[label]["FN"] += fn
            dic_one_sample_stats[label]["TN"] += tn
        
        return dic_one_sample_stats
    
    def add_local_to_global_stats(self, dic_global_stats: dict, dic_one_sample_stats: dict) -> None:
        all_labels = set(dic_one_sample_stats.keys())

        for label in all_labels:
            dic_global_stats[label]["TP"] += dic_one_sample_stats[label]["TP"]
            dic_global_stats[label]["FP"] += dic_one_sample_stats[label]["FP"]
            dic_global_stats[label]["FN"] += dic_one_sample_stats[label]["FN"]
            dic_global_stats[label]["TN"] += dic_one_sample_stats[label]["TN"]

    def compute_metrics(self, dic_stats: dict) -> dict:
        results = {}

        total_tp = total_fp = total_fn = total = 0
        weighted_precision = weighted_recall = weighted_f1 = 0

        for label, stats in dic_stats.items():
            tp = stats["TP"]
            fp = stats["FP"]
            fn = stats["FN"]
            support = tp + fn
            total += support

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
            accuracy = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0

            results[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "support": support
            }

            total_tp += tp
            total_fp += fp
            total_fn += fn

            weighted_precision += precision * support
            weighted_recall += recall * support
            weighted_f1 += f1 * support

        accuracy = total_tp / (total_tp + total_fp + total_fn)

        micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0.0

        weighted_precision = weighted_precision / total if total > 0 else 0.0
        weighted_recall = weighted_recall / total if total > 0 else 0.0
        weighted_f1 = weighted_f1 / total if total > 0 else 0.0

        results["__global__"] = {
            "accuracy": accuracy,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
        }

        return results
    

if __name__ == "__main__":
    sentences_file_path = "training_data/sentences_test.json"
    labels_file_path = "training_data/labels_test.json"
    
    gpt_ner_model = GPTNERExtractor(**{**ProcessFile.read_json("config.json")["LLM"]["GPT"], 
                                     **dict(prompt_examples=EXAMPLE)})
    evaluation_ner_model = EvaluateGPTNERExtractor(gpt_ner_model)

    evaluation_ner_model.run_ner_gpt(sentences_file_path, labels_file_path)