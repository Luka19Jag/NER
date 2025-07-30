import random
import time
import logging
logging.basicConfig(level=logging.INFO)

from concurrent.futures import ThreadPoolExecutor, as_completed

from gpt_ner_data_generator_base import GPTNERDataGeneratorBase
from process_input_file import ProcessFile
from gpt_base import PromptType

class EvaluateGPTNERDataGenerator:

    def __init__(self, model: GPTNERDataGeneratorBase, llm_method=PromptType.ZERO_SHOT):
        self.model = model
        self.llm_method = llm_method

    def generate_from_sentences(self, version: str, sentences_file_path: str, labels_file_path: str, 
                                to_save_path="training_data", limit_documents=500, min_tokens=20, run_parallel=True) -> list:
        sentences = ProcessFile.read_json(sentences_file_path)[:]
        labels = ProcessFile.read_json(labels_file_path)[:]
        
        indexes = sorted(random.sample([i for i,e in enumerate(sentences) if len(e) >= min_tokens], limit_documents))
        sentences = [e for i,e in enumerate(sentences) if i in indexes]
        labels = [e for i,e in enumerate(labels) if i in indexes]
        zipped_data = zip(sentences, labels)
        
        if run_parallel:
            synthetic_sentences, synthetic_labels, syntnetic_errors, seen_sentences = self.run_parallel(zipped_data)
        else:
            synthetic_sentences, synthetic_labels, syntnetic_errors, seen_sentences = self.run_sequential(zipped_data)
        
        ProcessFile.save_json(f"{to_save_path}/sentences_train_{version}.json", synthetic_sentences)
        ProcessFile.save_json(f"{to_save_path}/labels_train_{version}.json", synthetic_labels)
        ProcessFile.save_json(f"{to_save_path}/sentences_train_errors_{version}.json", syntnetic_errors)
    
    def run_sequential(self, zipped_data: zip) -> tuple:
        synthetic_sentences, synthetic_labels, syntnetic_errors, seen_sentences = [], [], [], set()
        
        for _, (sentence, label) in enumerate(zipped_data):
            synthetic_sentence, synthetic_label, random_sample = self.process_sample(sentence, label)
            
            if synthetic_sentence is not None and random_sample not in seen_sentences:
                seen_sentences.add(random_sample)
                synthetic_sentences.append(synthetic_sentence)
                synthetic_labels.append(synthetic_label)
                
            else:
                syntnetic_errors.append(synthetic_sentence)
                
        return (synthetic_sentences, synthetic_labels, syntnetic_errors, seen_sentences)
    
    def run_parallel(self, zipped_data: zip, max_workers=5) -> tuple:
        synthetic_sentences, synthetic_labels, syntnetic_errors, seen_sentences = [], [], [], set()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.process_sample, sentence, label)
                for sentence, label in zipped_data
            ]

        for future in as_completed(futures):
            synthetic_sentence, synthetic_label, random_sample = future.result()
            if synthetic_sentence is not None and random_sample not in seen_sentences:
                seen_sentences.add(random_sample)
                synthetic_sentences.append(synthetic_sentence)
                synthetic_labels.append(synthetic_label)
                
            else:
                syntnetic_errors.append(synthetic_sentence)
        
        return (synthetic_sentences, synthetic_labels, syntnetic_errors, seen_sentences)
    
    def process_sample(self, sentence: list, label: list) -> dict:
        time.sleep(random.uniform(0.01, 3))
        
        prompt = self.model.get_prompt((sentence, label), method=self.llm_method)
        response, raw_text = self.model.request(prompt, temperature=1.0)
        
        if raw_text is None:
            return None, None, None

        text = self.model.parse_output(raw_text)
        if text:
            synthetic_sentence = text["sentence"]
            synthetic_label = text["label"]
            # random_sample = " ".join(sorted(random.sample(synthetic_sentence, min(len(synthetic_label), 10))))
            random_sample = " ".join(synthetic_sentence[:5])
            logging.info(f"\n\n***** Original sentence:\n{sentence}\n***** Synthetic sentence:\n{synthetic_sentence}\n***** Syntnetic label:\n{synthetic_label}")
            return (synthetic_sentence, synthetic_label, random_sample)
        
        return None, None, None
    
if __name__ == "__main__":
    version = "v2"
    sentences_file_path = "training_data/sentences_train.json"
    labels_file_path = "training_data/labels_train.json"
    
    gpt_ner_sytnetic_data_generator_model = GPTNERDataGeneratorBase(**ProcessFile.read_json("config.json")["LLM"]["GPT"])
    evaluation_ner_sytnetic_data_generator_model = EvaluateGPTNERDataGenerator(gpt_ner_sytnetic_data_generator_model)

    evaluation_ner_sytnetic_data_generator_model.generate_from_sentences(version, sentences_file_path, labels_file_path)
