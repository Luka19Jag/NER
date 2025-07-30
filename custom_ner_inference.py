from collections import defaultdict
from sklearn.metrics import classification_report

from custom_ner_base_inference import BaseNERInference
from process_input_file import ProcessFile

class EvaluateCustomNerModel:

    @staticmethod
    def evaluate_multiple_models(dic_models: dict, label_to_id_path: str, id_to_label_path: str,
                            test_sentences_filepath: str, test_labels_filepath: str, to_save_folder="inference_data") -> None:
        test_sentences = ProcessFile.read_json(test_sentences_filepath)
        test_labels = ProcessFile.read_json(test_labels_filepath)

        for model_name, model_info in dic_models.items():
            model = BaseNERInference(model_info["model_path"], label_to_id_path, id_to_label_path)
            all_labels, all_predictions = [], []

            for sentence, labels in zip(test_sentences, test_labels):
                predictions_pairs = model.predict(sentence)
                predictions = [label for _, label in predictions_pairs]

                assert len(predictions) == len(labels), \
                    f"Mismatch in lengths. Predicted: {len(predictions)}, True: {len(labels)}"

                all_labels.extend(labels)
                all_predictions.extend(predictions)

            report = classification_report(
                all_labels, all_predictions, output_dict=True, zero_division=0
            )
            ProcessFile.save_json(f"{to_save_folder}/{model_name}/test_statistics.json", report)

if __name__ == "__main__":
    dic_models = {
        # v1
        "bert-base-cased/v1": {
            "model_path": "ner_model/bert-base-cased/v1/best_model"
        },
        "dslim/bert-base-NER/v1": {
            "model_path": "ner_model/dslim/bert-base-NER/v1/best_model"
        },
        "roberta-base/v1": {
            "model_path": "ner_model/roberta-base/v1/best_model"
        },
        
        # v2
        "bert-base-cased/v2": {
            "model_path": "ner_model/bert-base-cased/v2/best_model"
        },
        "dslim/bert-base-NER/v2": {
            "model_path": "ner_model/dslim/bert-base-NER/v2/best_model"
        },
        "roberta-base/v2": {
            "model_path": "ner_model/roberta-base/v2/best_model"
        }
    }
    label_to_id_path = "input_data/label_to_id.json"
    id_to_label_path = "input_data/id_to_label.json"
    test_sentences_filepath = "training_data/sentences_test.json"
    test_labels_filepath = "training_data/labels_test.json"

    EvaluateCustomNerModel.evaluate_multiple_models(dic_models, label_to_id_path, id_to_label_path,
                                        test_sentences_filepath, test_labels_filepath)