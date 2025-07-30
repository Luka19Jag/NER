EXAMPLE = [
    {
        "sentence": "Angela Merkel gave a speech at the United Nations summit held in Berlin.",
        
        "reasoning": """
**I-PER**
"Angela Merkel" is a known political leader and individual person.

**I-ORG**
"United Nations" is a global international organization.

**I-LOC**
"Berlin" is the capital city of Germany.
""",
        "label": {
            "I-PER": ["Angela Merkel"],
            "I-ORG": ["United Nations"],
            "I-LOC": ["Berlin"]  
        }
    }
]