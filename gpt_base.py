import requests
import time
import logging
import json
logging.basicConfig(level=logging.INFO)

from typing import Tuple
from enum import Enum

from abc import ABC, abstractmethod

class LLMBase(ABC):
    @abstractmethod
    def request(self, prompt: str) -> str:
        pass

class PromptType(Enum):
    ZERO_SHOT = "zero_shot"
    ONE_SHOT = "one_shot"
    FEW_SHOT = "few_shot"

class GPTBase(LLMBase):
    def __init__(self, api_key: str, api_url: str, model: str, 
                 retry_count: int, prompt_examples=None):        
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.retry_count = retry_count
        self.prompt_examples = prompt_examples
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        self.system_role = "system"
        self.user_role = "user"
        self.assistant_role = "assistant"

    def request(self, prompt: list, temperature = 0.0) -> Tuple:
        payload = {
            "model": self.model,
            "messages": [
                {"role": role, "content": message} for role, message in prompt
            ],
            "temperature": temperature
        }

        for attempt in range(1, self.retry_count + 1):
            response = requests.post(self.api_url, 
                                     headers=self.headers, 
                                     json=payload)
            
            if response.status_code == 200:
                raw_text = response.json()
                raw_text = raw_text['choices'][0]['message']['content']
                return response, raw_text
            elif response.status_code in {429, 500, 502, 503, 504}:
                wait_time = 2 ** attempt
                print(f"Retry {attempt}/{self.retry_count} - Waiting {wait_time}s due to error {response.status_code}")
                time.sleep(wait_time)
            else:
                logging.error(f"API request failed with status code {response.status_code}: {response.text}")

        logging.error(f"Request failed after {self.retry_count} retries.")

        return response, None
    
    def get_prompt(self, sentence: list, method=PromptType) -> list:
        if method == PromptType.ZERO_SHOT:
            return self.zero_shot_prompt(sentence)

        elif method in [PromptType.ONE_SHOT, PromptType.FEW_SHOT]:
            return self.few_shot_prompt(sentence)

        else:
            logging.error("Wrong method for LLM.")
            return None
    
    def zero_shot_prompt(self, sentence: list) -> dict:
        return [
            (self.system_role, self.get_system_message()),
            (self.user_role, self.get_user_message(sentence))
        ]
    
    def few_shot_prompt(self, sentence: list) -> dict:
        commands = [
            (self.system_role, self.get_system_message())
        ]
        
        for example in self.prompt_examples:
            example_sentence = example["sentence"]
            example_reasoning = example["reasoning"]
            example_label = example["label"]    

            example_prompt = f"""{example_reasoning}
```json
{json.dumps(example_label, indent=2)}
```"""
            commands.append((self.user_role, self.get_user_message(example_sentence)))
            commands.append((self.assistant_role, example_prompt))

        commands.append((self.user_role, self.get_user_message(" ".join(sentence))))

        return commands
        
    def parse_output(self, raw_text: str) -> dict:
        try:
            if "```json" in raw_text:
                start = raw_text.find('{')
                end = raw_text.rfind('}') + 1
                if start == -1 or end == -1:
                    raise ValueError("JSON block not found.")
                json_str = raw_text[start:end]
                return json.loads(json_str)
            
            elif text := json.loads(raw_text):
                return text
            
            else:
                logging.error(f"Failed to parse output: {e}. Check !!!")
                return None
            
        except Exception as e:
            logging.error(f"Failed to parse output: {e}")
            return None
    
    def get_system_message(self) -> str:
        raise NotImplementedError("This is meant to be implemented in lower class.")

    def get_user_message(self, sentence: str) -> str:
        raise NotImplementedError("This is meant to be implemented in lower class.")