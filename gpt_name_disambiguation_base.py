import logging
logging.basicConfig(level=logging.INFO)

from gpt_base import GPTBase

class GPTNameDisambiguationBase(GPTBase):
    def __init__(self, api_key: str, api_url: str, model = "gpt-4.1-mini", retry_count = 3):
        super().__init__(api_key, api_url, model, retry_count)
    
    def get_system_message(self) -> str:
        return (
            "You are an expert in bibliometric analysis and name disambiguation. "
            "Your job is to distinguish between different individuals who share the same name, "
            "based on patterns in their publication metadata. "
            "Apply conservative clustering - prefer keeping publications together unless there's "
            "strong evidence they belong to different people. Consider that researchers often "
            "change topics, move institutions, and collaborate with new people throughout their careers."
        )

    def get_user_message(self, zipped_data: tuple) -> str:
        publications, target_name = zipped_data
        return f"""You are analyzing publications attributed to the name **{target_name}**. 
Your goal is to identify how many **distinct individuals** this name may refer to, based on differences in coauthors, research topics, venues, publication years, and organizations.

Instructions:
- Group the publications into clusters where each cluster belongs to the same real-world person.
- Use your reasoning to support the clustering: e.g., consistent coauthors, overlapping publication years, similar venues, or similar research fields.
- Consider temporal gaps (>5 years with no publications), dramatic topic shifts, complete change in collaborator networks, and venue changes as potential indicators of distinct people.
- **However, be cautious**: Career transitions, interdisciplinary work, and sabbaticals are normal - don't split too aggressively.
- If publications have overlapping coauthors OR similar research domains OR institutional continuity, they likely belong to the same person.
- Only separate into different people when there's strong evidence of distinct research identities with no connecting factors.
- For each cluster, explain your reasoning and identify the key discriminating features.

Publications:
{publications}

Final JSON output format:
```json
{{
    "0": {{
        "publication_ids": ["<id1>", "<id2>", ...],
        "reasoning": "Consistent coauthors X, Y, Z across 2015-2020, all in machine learning venues...",
        "key_identifiers": ["coauthor_overlap", "venue_consistency", "topic_coherence"]
    }},
    "1": {{
        "publication_ids": ["<id4>", "<id5>", ...],
        "reasoning": "Completely different research area (biology vs CS), different time period (2008-2012), no shared coauthors...",
        "key_identifiers": ["topic_divergence", "temporal_gap", "no_coauthor_overlap"]
    }}
}}
```"""