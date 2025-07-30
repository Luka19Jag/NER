import xmltodict
import json
import os

from bs4 import BeautifulSoup
from collections import defaultdict

class ProcessFile:
    @staticmethod
    def save_json(file_path: str, data: dict) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f)
            print(f"File has been saved: {file_path}")

    @staticmethod
    def read_json(file_path: str) -> dict:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def read_xml_format(input_file_path: str) -> None:
        with open(input_file_path, "r", encoding="utf-8") as f:
            xml_data = json.load(f)

        profile_data = defaultdict(dict)
        for key, xml_string in xml_data.items():
            new_key = key.replace("_classify_txt", "").replace("_xml", "").strip()

            if "classify_txt" in key:
                profile_data[new_key]["labels"] =  ProcessFile.parse_profile_labels(xml_string)
            else:
                profile_data[new_key]["metadata"], profile_data[new_key]["custom_labels"], profile_data[new_key]["custom_publication_labels"] = \
                    ProcessFile.parse_profile_metadata(xml_string)

        file_path_save = "".join(input_file_path.split(".")[:-1]) + "_clean.json"
        ProcessFile.save_json(file_path_save, profile_data)
    
    @staticmethod
    def parse_profile_metadata(xml_string: str) -> tuple:
        clean_xml_string = BeautifulSoup(xml_string, "xml").prettify()
        parsed_dict = xmltodict.parse(clean_xml_string)
        custom_labels, custom_publication_labels = ProcessFile.parse_custom_prfile_labels(parsed_dict)
        return parsed_dict, custom_labels, custom_publication_labels

    @staticmethod
    def parse_custom_prfile_labels(profile_data: dict) -> dict:
        labels, publications = defaultdict(list), defaultdict(list)
        if profile_data.get("person", None) is None or profile_data["person"].get("publication", None) is None:
            return {}, {}
        for pub in profile_data['person']['publication']:
            labels[pub["label"]].append(pub["id"])
            publications[pub["label"]].append(pub["title"])
        return dict(sorted(labels.items())), dict(sorted(publications.items()))

    @staticmethod
    def parse_profile_labels(raw_text: str) -> dict:
        profile = defaultdict(list)
        lines = raw_text.strip().split('\n')
        person_name = None

        for line in lines:
            if line.startswith('#') and ':' not in line:
                person_name = line.lstrip('#').strip()
            elif line.startswith('#') and ':' in line:
                key, values = line[1:].split(':', 1)
                if key.startswith("!"):
                    continue
                profile[f"{int(key)-1}"] = values.strip().split()

        return dict(profile)

    @staticmethod
    def read_conll_format(input_file_path: str) -> None:
        sentences = []
        labels = []
        sentence = []
        label = []

        with open(input_file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if sentence:
                        sentences.append(sentence)
                        labels.append(label)
                        sentence = []
                        label = []
                else:
                    splits = line.split()
                    if len(splits) >= 2:
                        token, ner = splits[0], splits[-1]
                        sentence.append(token)
                        label.append(ner)
        
        all_labels = sorted(set([v for e in labels for v in e]))
        label_to_id = {str(label): str(i) for i, label in enumerate(all_labels)}
        id_to_label = {str(i): str(label) for i, label in enumerate(all_labels)}
        ProcessFile.save_json(input_file_path.split("/")[0] + "/" + "label_to_id.json", label_to_id)
        ProcessFile.save_json(input_file_path.split("/")[0] + "/" + "id_to_label.json", id_to_label)

        file_path_base = "".join(input_file_path.split(".")[:-1])
        ProcessFile.save_json(file_path_base + "_sentences.json", sentences)
        ProcessFile.save_json(file_path_base + "_labels.json", labels)        

if __name__ == "__main__":
    ProcessFile.read_xml_format("input_data/person_profile_data.json")
    ProcessFile.read_conll_format("input_data/wikigold_conll.txt")