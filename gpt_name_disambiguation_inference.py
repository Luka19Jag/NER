import random
import time
import logging
logging.basicConfig(level=logging.INFO)

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from gpt_name_disambiguation_base import GPTNameDisambiguationBase
from process_input_file import ProcessFile
from gpt_base import PromptType

class EvaluateGPTNameDisambiguation:
    
    def __init__(self, model: GPTNameDisambiguationBase, llm_method=PromptType.ZERO_SHOT):
        self.model = model
        self.llm_method = llm_method

    def evaluate_name_disambiguation(self, person_profile_data_file_path: str, 
                                     to_save_path="inference_data", run_parallel=True) -> list:
        person_profile_data = ProcessFile.read_json(person_profile_data_file_path)
        # person_profile_data = {k:v for k,v in person_profile_data.items() if k in list(person_profile_data)[:5]}
        
        if run_parallel:
            predictions, errors = self.run_parallel(person_profile_data)
        else:
            predictions, errors = self.run_sequential(person_profile_data)
        
        predictions_updated, dic_stats = self.evaluate_pairwise_f1(predictions)
        b3_metrics = self.evaluate_b3_metrics(predictions_updated)
        
        ProcessFile.save_json(f"{to_save_path}/name_disambiguation/test_predictions.json", predictions_updated)
        ProcessFile.save_json(f"{to_save_path}/name_disambiguation/test_errors.json", errors)
        ProcessFile.save_json(f"{to_save_path}/name_disambiguation/test_statistics.json", dic_stats)
        ProcessFile.save_json(f"{to_save_path}/name_disambiguation/test_statistics_b3.json", b3_metrics)
    
    def run_sequential(self, person_profile_data: dict) -> tuple:
        predictions, errors = {}, []
        
        for i,person in enumerate(person_profile_data):

            prediction = self.process_sample(person_profile_data[person], person)
            if prediction:
                predictions[prediction["person"]] = prediction
            else:
                errors.append(person)
        
        return predictions, errors
    
    def run_parallel(self, person_profile_data: zip, max_workers=5) -> tuple:
        predictions, errors = {}, []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.process_sample, person_profile_data[person], person)
                for person in person_profile_data
            ]

        for future in as_completed(futures):
            prediction = future.result()
            if prediction:
                predictions[prediction["person"]] = prediction
            else:
                errors.append(prediction)
        
        return predictions, errors
    
    def process_sample(self, person_data: dict, person: str) -> dict:
        time.sleep(random.uniform(0.01, 3))
        
        custom_labels = person_data['custom_labels']
        if custom_labels == []:
            return None
        
        try:
            target_name = person_data['metadata']['person']['FullName']
            publications = person_data['metadata']['person']['publication']
            formatted_pubs, dic_pubs = self.prepare_publications(publications, target_name)
                
            prompt = self.model.get_prompt((formatted_pubs, target_name), method=self.llm_method)
            response, raw_text = self.model.request(prompt, temperature=0.0)
            
            if raw_text is None:
                return None

            text = self.model.parse_output(raw_text)
            if text:
                mapped_titles_names = defaultdict(dict)
                for k,v in text.items():
                    mapped_titles_names[k]["reasoning"] = v["reasoning"]
                    mapped_titles_names[k]["key_identifiers"] = v["key_identifiers"]
                    mapped_titles_names[k]["publication_titles"] = []
                    for t in v["publication_ids"]:
                        mapped_titles_names[k]["publication_titles"].append(dic_pubs[t])
                logging.info(f"\n\n***** Target name:\n{target_name}\n***** Publications:\n{formatted_pubs}\n***** Cluster per IDs:\n{text}\n***** Cluster per Titles:\n{mapped_titles_names}")
                return dict(
                    person=person,
                    target_name=target_name,
                    clusters_per_ids=text,
                    clusters_per_titles=mapped_titles_names,
                    custom_labels=custom_labels
                )
                
            return None
        
        except:
            logging.error("\n\n\n ERROR \n\n\n")
            return None

    def prepare_publications(self, publications: list, target_name: str, missing_info="N/A") -> tuple:
        formatted_pubs = []
        dic_pubs = defaultdict(str)
        
        for pub in publications:
            id = pub["id"]
            title = pub["title"]
            if title == "null" or id == "null":
                continue
            
            coauthors = pub["authors"].replace(target_name, "").strip(", ")
            year = pub.get("year", missing_info) if pub.get("year") != "null" else missing_info
            venue = pub.get("jconf", missing_info) if pub.get("jconf") != "null" else missing_info
            organization = pub.get("organization", missing_info) if pub.get("organization") != "null" else missing_info
            
            formatted_pubs.append(
                f'- ID: {id}, Title: "{title}", Year: {year}, '
                f'Venue: {venue}, Co-authors: [{coauthors}], Organization: {organization}'
            )
            dic_pubs[id] = title
            
        return "\n".join(formatted_pubs), dic_pubs
    
    def evaluate_pairwise_f1(self, data: dict) -> tuple:
        
        def create_pub_to_cluster_map(clusters: dict) -> dict:
            pub_to_cluster = {}
            for cluster_id, pub_list in clusters.items():
                if pub_list != [] and pub_list is not None:
                    for pub_id in pub_list:
                        pub_to_cluster[pub_id] = cluster_id
            return pub_to_cluster
        
        def get_metrics(true_positive, true_negative, false_positive, false_negative) -> dict:
            total = (true_positive + false_positive + false_negative + true_negative)
            accuracy = (true_positive + true_negative) / total if total > 0 else 0.0
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                "total": total,
                "accuracy": round(accuracy, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative
            }
            
        true_positive = 0   
        false_positive = 0  
        false_negative = 0  
        true_negative = 0  
        
        for _, person_cluster in data.items():
            local_true_positive = 0   
            local_false_positive = 0  
            local_false_negative = 0  
            local_true_negative = 0  
            
            try:
                gt_map = create_pub_to_cluster_map({k:v["publication_ids"] for k,v in person_cluster["clusters_per_ids"].items()})
                pred_map = create_pub_to_cluster_map(person_cluster["custom_labels"])
            except:
                continue
            
            all_pubs = set(gt_map.keys()) & set(pred_map.keys())
            all_pubs = list(all_pubs)
            
            for i in range(len(all_pubs)):
                for j in range(i + 1, len(all_pubs)):
                    pub1, pub2 = all_pubs[i], all_pubs[j]
                    
                    gt_same = (gt_map[pub1] == gt_map[pub2])
                    
                    pred_same = (pred_map[pub1] == pred_map[pub2])
                    
                    if pred_same and gt_same:
                        true_positive += 1
                        local_true_positive += 1
                    elif pred_same and not gt_same:
                        false_positive += 1
                        local_false_positive += 1
                    elif not pred_same and gt_same:
                        false_negative += 1
                        local_false_negative += 1
                    else:
                        true_negative += 1
                        local_true_negative += 1
            
            person_cluster["stats"] = get_metrics(local_true_positive, local_true_negative, local_false_positive, local_false_negative)
            
        return data, get_metrics(true_positive, true_negative, false_positive, false_negative)
    
    def evaluate_b3_metrics(self, data: dict) -> dict:
        def create_pub_to_cluster_map(clusters):
            pub_to_cluster = {}
            for cluster_id, pub_list in clusters.items():
                if pub_list != [] and pub_list is not None:
                    for pub_id in pub_list:
                        pub_to_cluster[pub_id] = cluster_id
            return pub_to_cluster
        
        def create_cluster_members_map(clusters):
            cluster_members = {}
            for cluster_id, pub_list in clusters.items():
                if pub_list != [] and pub_list is not None:
                    cluster_members[cluster_id] = set(pub_list)
            return cluster_members
        
        total_publications = 0.0
        total_precision = 0.0
        total_recall = 0.0
        pub_details = {}
        
        for _, person_cluster in data.items():
            
            try:
                predictions = {k:v["publication_ids"] for k,v in person_cluster["clusters_per_ids"].items()}
                ground_truth = person_cluster["custom_labels"]
                    
                gt_pub_to_cluster = create_pub_to_cluster_map(ground_truth)
                pred_pub_to_cluster = create_pub_to_cluster_map(predictions)
                
                gt_cluster_members = create_cluster_members_map(ground_truth)
                pred_cluster_members = create_cluster_members_map(predictions)
            except:
                continue
            
            common_pubs = set(gt_pub_to_cluster.keys()) & set(pred_pub_to_cluster.keys())
            total_publications += len(common_pubs)
            
            for pub in common_pubs:
                gt_cluster_id = gt_pub_to_cluster[pub]
                gt_same_cluster = gt_cluster_members[gt_cluster_id]
                gt_same_cluster = gt_same_cluster & common_pubs
                
                pred_cluster_id = pred_pub_to_cluster[pub]
                pred_same_cluster = pred_cluster_members[pred_cluster_id]
                pred_same_cluster = pred_same_cluster & common_pubs
                
                intersection = gt_same_cluster & pred_same_cluster
                pub_precision = len(intersection) / len(pred_same_cluster) if len(pred_same_cluster) > 0 else 0.0
                pub_recall = len(intersection) / len(gt_same_cluster) if len(gt_same_cluster) > 0 else 0.0
                total_precision += pub_precision
                total_recall += pub_recall
                
                pub_details[pub] = {
                    "gt_cluster": gt_cluster_id,
                    "pred_cluster": pred_cluster_id,
                    "gt_same_cluster": list(gt_same_cluster),
                    "pred_same_cluster": list(pred_same_cluster),
                    "intersection": list(intersection),
                    "precision": round(pub_precision, 3),
                    "recall": round(pub_recall, 3)
                }
        
        if total_publications == 0:
            return {
                "b3_precision": 0.0,
                "b3_recall": 0.0,
                "b3_f1": 0.0,
                "n_publications": 0,
                "publication_details": {}
            }
        
        avg_precision = total_precision / total_publications
        avg_recall = total_recall / total_publications
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        return {
            "b3_precision": round(avg_precision, 3),
            "b3_recall": round(avg_recall, 3),
            "b3_f1": round(avg_f1, 3),
            "n_publications": total_publications,
            "publication_details": pub_details
        }

if __name__ == "__main__":
    person_profile_data_file_path = "input_data/person_profile_data_clean.json"
    
    gpt_name_disambiguation_model = GPTNameDisambiguationBase(**ProcessFile.read_json("config.json")["LLM"]["GPT"])
    evaluation_ner_sytnetic_data_generator_model = EvaluateGPTNameDisambiguation(gpt_name_disambiguation_model)

    evaluation_ner_sytnetic_data_generator_model.evaluate_name_disambiguation(person_profile_data_file_path)
