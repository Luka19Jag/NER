from sklearn.model_selection import train_test_split

from process_input_file import ProcessFile 

class PrepareData:

    @staticmethod
    def split_data(file_path_sentences: str, file_path_labels: str, 
                   base_folder="training_data", random_state=42) -> None:
        
        sentences = ProcessFile.read_json(file_path_sentences)
        labels = ProcessFile.read_json(file_path_labels)

        sentences_train, sentences_temp, labels_train, labels_temp = train_test_split(
            sentences, labels, test_size=0.3, random_state=random_state
        )

        sentences_val, sentences_test, labels_val, labels_test = train_test_split(
            sentences_temp, labels_temp, test_size=0.5, random_state=random_state
        )

        # train
        ProcessFile.save_json(f"{base_folder}/sentences_train.json", sentences_train)
        ProcessFile.save_json(f"{base_folder}/labels_train.json", labels_train)

        # val
        ProcessFile.save_json(f"{base_folder}/sentences_val.json", sentences_val)
        ProcessFile.save_json(f"{base_folder}/labels_val.json", labels_val)

        # test
        ProcessFile.save_json(f"{base_folder}/sentences_test.json", sentences_test)
        ProcessFile.save_json(f"{base_folder}/labels_test.json", labels_test)

if __name__ == "__main__":    
    PrepareData.split_data("input_data/wikigold_conll_sentences.json", "input_data/wikigold_conll_labels.json")