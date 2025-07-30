import json
import logging
logging.basicConfig(level=logging.INFO)

from gpt_base import GPTBase

class GPTNERExtractor(GPTBase):

    def __init__(self, api_key: str, api_url: str, model = "gpt-4.1-mini", retry_count = 3, prompt_examples=None):
        super().__init__(api_key, api_url, model, retry_count, prompt_examples)

    def get_system_message(self) -> str:
        return "You are an expert in Named Entity Recognition (NER)."

    def get_user_message(self, sentence: list) -> str:
        sentence = " ".join(sentence)
        return f"""Extract named entities from the text below and classify them using the following labels only:
- I-LOC: Location (cities, countries, rivers, mountains)
- I-MISC: Miscellaneous (events, products, works of art, etc.)
- I-ORG: Organizations (companies, institutions, government bodies)
- I-PER: Persons (individual people or fictional characters)

Note:
- The input text has been created by joining tokens, so spacing, punctuation, or formatting may be inconsistent. You must still interpret the intended sentence structure and extract named entities correctly.

Instruction:
- Join adjacent words if they form a single named entity (e.g., "New" + "York" → "New York").
- All identified entities should be included, even if they appear more than once.
- If no entities are found for a label, return an empty list for that label in the JSON.
- The JSON must be valid, compact, and must contain no explanations or commentary.

Return:
- A short reasoning section for each label, explaining why each entity was classified that way. Use markdown headers (e.g., **PERSON** → *I-PER*).
- A JSON dictionary where each label maps to a list of extracted strings.

Text:
"{sentence}"
"""
    