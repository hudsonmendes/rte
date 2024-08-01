# Third-Party Libraries
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = """
# Inputs
# Entity Types <- {entities}
# Relationships <- {relations}
# Text <- {document}
# Output -> (subject > predicate > object)""".strip()


model = AutoModelForCausalLM.from_pretrained("sciphi/triplex", trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained("sciphi/triplex", trust_remote_code=True)

entity_types = ["LOCATION", "POSITION", "DATE", "CITY", "COUNTRY", "NUMBER"]
predicates = ["POPULATION", "AREA"]
text = """
San Francisco, officially the City and County of San Francisco, is a commercial, financial, and cultural center in Northern California.
With a population of 808,437 residents as of 2022, San Francisco is the fourth most populous city in the U.S. state of California behind Los Angeles, San Diego, and San Jose.
"""


def extract_triplets(doc) -> str:
    plain_entitites = ", ".join(entity_types)
    plain_predicates = ", ".join(predicates)
    x = tokenizer(prompt.format(entities=plain_entitites, relations=plain_predicates, document=doc), return_tensors="pt")
    return model.generate(x["input_ids"])


if __name__ == "__main__":
    print(extract_triplets(text))
