import logging
logging.basicConfig(level=logging.INFO)

from gpt_base import GPTBase

class GPTNERDataGeneratorBase(GPTBase):
    def __init__(self, api_key: str, api_url: str, model = "gpt-4.1-mini", retry_count = 3):
        super().__init__(api_key, api_url, model, retry_count)
    
    def get_system_message(self) -> str:
        return "You are an NLP expert specialized in generating synthetic training data for Named Entity Recognition (NER)."

    def get_user_message(self, zipped_data: tuple) -> str:
        sentence, label = zipped_data
        return f"""Generate a synthetic example for Named Entity Recognition (NER) based on the given tokenized sentence and NER labels.
    
Requirements:
- Generate a **new sentence** and **new labels** with the random number of tokens.
- Be highly creative when generating synthetic sample.
- You can modify the entities to similar alternatives (e.g., replace one location with another), but preserve the label types at the correct token positions.
- Do **not** reuse the original tokens.
- Ensure the output sentence is realistic and grammatically correct.
- Each output token MUST be aligned with one label.
- Vary sentence structure, vocabulary, and phrasing.
- The number of tokens and labels MUST match exactly!

NER Labels follow this format:
- I-PER: Person
- I-LOC: Location
- I-ORG: Organization
- I-MISC: Miscellaneous
- O: Outside any entity

Input sentence:
{sentence}

NER label:
{label}

Before final JSON output provide reasoning:
**reasoning**
I created synthetic sample based on ....

Final JSON output structure: 
```json
{{
    "sentence": [...list of new tokens...],
    "label": [...list of NER labels (same length, aligned)...]
}}
```
"""