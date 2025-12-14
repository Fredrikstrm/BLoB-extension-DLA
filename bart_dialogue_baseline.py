from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load DialogueSum
dset = load_dataset("knkarthick/dialogsum")

val = dset["validation"].select(range(200))  # small subset is enough for a check

# 2) Load plain BART (no LoRA, no BLoB)
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()

# 3) ROUGE metric
rouge = evaluate.load("rouge")

preds, refs = [], []

for ex in tqdm(val):
    inputs = tokenizer(
        ex["dialogue"],
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
        )

    preds.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    refs.append(ex["summary"])

scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
print(scores)