import modal
import os
import subprocess

app = modal.App("bayesian-peft-blob")

repo_volume = modal.Volume.from_name("bayesian-peft-repo", create_if_missing=True)

# Build image with CUDA + pip deps
# taken from: https://hub.docker.com/r/pytorch/pytorch/tags
image = (
    modal.Image.from_registry("pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime")
    .apt_install("bash")
    .pip_install(
        "transformers",
        "datasets",
        "evaluate",
        "absl-py",        
        "rouge-score",    
        "accelerate",
        "bitsandbytes",
        "jaxtyping",
        "torchmetrics",
        "setproctitle",
        "peft",
        "wandb",
        "nltk",
        "scikit-learn",
        "ipdb",
    )
) 

wandb_secret = modal.Secret.from_name("wandb-api-key")

@app.local_entrypoint()
def sync_repo():
    """
    Sync of local bayesian-peft repo into the Modal volume.
    """ 
    repo_volume.remove_dir("/workspace/bayesian-peft/wandb")
    with repo_volume.batch_upload() as batch:
        batch.put_directory(
            local_path=".",                
            remote_path="/workspace/bayesian-peft",
        )
    print("Repo synced to Modal volume.")

import time
@app.local_entrypoint()
def clone_repo():
    ts = time.strftime("%Y%m%d_%H%M%S")
    remote = f"/workspace/bayesian-peft_{ts}"

    with repo_volume.batch_upload() as batch:
        batch.put_directory(
            local_path=".",
            remote_path=remote,
        )

    print("Repo synced to:", remote)
    print("Use this path in Modal:", remote)

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],
)
def run_blob_roberta_all():
    os.chdir("/workspace/workspace/bayesian-peft")
    subprocess.run(["bash", "scripts/blob/blob-roberta-all.sh"], check=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],
)
def run_blob_roberta_base_all():
    os.chdir("/workspace/workspace/bayesian-peft_20251212_223404")
    subprocess.run(["bash", "scripts/blob/blob-roberta-base-all.sh"], check=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/workspace": repo_volume},
)
def debug_boolq_seed1():
    os.chdir("/workspace/workspace/bayesian-peft")
    subprocess.run(
        [
            "python", "-m", "accelerate.commands.launch",
            "run/main.py",
            "--dataset-type", "bertds",
            "--dataset", "boolq",
            "--model", "roberta-large",
            "--model-type", "seqcls",
            "--modelwrapper", "blob",
            "--lr", "5e-4",
            "--batch-size", "32",
            "--max-train-steps", "200",
            "--eval-per-steps", "200",
            "--apply-classhead-lora",
            "--lora-r", "8","--lora-alpha","8",
            "--lora-dropout","0",
            "--nowand",        
        ],
        check=True,
    )

@app.function(volumes={"/workspace": repo_volume})
def debug_ls():
    import os
    print("ROOT:", os.listdir("/"))
    print("WORKSPACE:", os.listdir("/workspace"))
    print("WS/WORKSPACE:", os.listdir("/workspace/workspace"))

@app.function(
    image=image,
    gpu="A100",                
    timeout=60 * 60,
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],  
)
def test_dialoguesum_blob_summarization():

    os.chdir("/workspace/workspace/bayesian-peft")

    subprocess.run(
        [
            "python", "-m", "run.main",
            "--dataset-type", "oedataset",
            "--dataset", "dialoguesum",
            "--model", "facebook/bart-base",
            "--model-type", "seq2seq",
            "--modelwrapper", "blob_summarization",
            "--batch-size", "4",
            "--max-train-steps", "10",
            "--eval-per-steps", "1000",
        ],
        check=True,
        )


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],
)
def run_blob_dialoguesum_bart():
    os.chdir("/workspace/workspace/bayesian-peft")
    subprocess.run(["bash", "scripts/blob/blob-dialoguesum-bart.sh"], check=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],
)
def run_blob_dialoguesum_bart_new():
    os.chdir("/workspace/workspace/bayesian-peft")
    subprocess.run(["bash", "scripts/blob/blob-dialoguesum-bart-new.sh"], check=True)

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],
)
def run_blob_dialoguesum_bart_ablation():
    os.chdir("/workspace/workspace/bayesian-peft")
    subprocess.run(["bash", "scripts/blob/blob-dialoguesum-bart-ablation.sh"], check=True)

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],
)
def run_blob_dialoguesum_bart_hps():
    os.chdir("/workspace/workspace/bayesian-peft")
    subprocess.run(["bash", "scripts/blob/blob-dialoguesum-bart-hps.sh"], check=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],
)
def test_checkpoint_saving():
    import glob

    os.chdir("/workspace/workspace/bayesian-peft")

    # tiny run on RTE with blob-roberta
    subprocess.run(
        [
            "python", "-m", "run.main",
            "--dataset-type", "bertds",
            "--dataset", "rte",
            "--model", "roberta-large",
            "--model-type", "seqcls",
            "--modelwrapper", "blob",
            "--batch-size", "8",
            "--max-train-steps", "5",
            "--eval-per-steps", "10",
            "--seed", "1",
            "--wandb-project", "checkpoint-test",
            "--wandb-name", "blob-rte-checkpoint-test",
        ],
        check=True,
    )

    # List saved files so you can see them in Modal logs
    ckpts = glob.glob(
        "checkpoints/blob/roberta-large/rte/blob-rte-checkpoint-test/*"
    )
    print("Saved checkpoint files:", ckpts)


@app.function(
    image=image,
    gpu="A100",                
    timeout=60 * 60,
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],  
)
def test_dialoguesum_summarization():

    os.chdir("/workspace/workspace/bayesian-peft")

    subprocess.run(
        [
            "python", "-m", "run.main",
            "--dataset-type", "oedataset",
            "--dataset", "dialoguesum",
            "--model", "facebook/bart-base",
            "--model-type", "seq2seq",
            "--modelwrapper", "blob_summarization",
            "--batch-size", "4",
            "--max-train-steps", "10",
            "--eval-per-steps", "1000",
            "--wandb-project", "BLoB-dialoguesum-bart",
            "--wandb-name", "debug-no-kl",
        ],
        check=True,
    )


@app.function(
    image=image,
    gpu="A100",                
    timeout=60 * 60,
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],  
)
def tine_debug_train_eval():

    os.chdir("/workspace/workspace/bayesian-peft")

    subprocess.run(
        [
            "python", "-m", "run.main",
            "--dataset-type", "oedataset",
            "--dataset", "dialoguesum",
            "--model", "facebook/bart-base",
            "--model-type", "seq2seq",
            "--modelwrapper", "blob_summarization",
            "--batch-size", "4",
            "--max-train-steps", "10",
            "--eval-per-steps", "5",
            "--subset-size", "128",
            "--bayes-eval-n-samples-final", "1",
            "--wandb-project", "BLoB-dialoguesum-bart",
            "--wandb-name", "debug-blob-summarization",
        ],
        check=True,
    )

@app.function(
    image=image,
    gpu="A100",                
    timeout=60 * 60 * 12,
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],  
)
def lora_dialoguesum_summarization():

    os.chdir("/workspace/workspace/bayesian-peft")

    subprocess.run(["bash", "scripts/lora/lora-summarization.sh"], check=True)

@app.function(  # same image/volume as training
    image=image,
    gpu="A100",
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],
)
def compare_summaries():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    blob_base = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(device)

    blob = PeftModel.from_pretrained(
        blob_base,
        "/workspace/workspace/bayesian-peft/checkpoints/blob_summarization/facebook/bart-base/dialoguesum/blob_summarization-dialoguesum-sample10-eps0.05-kllr0.002-beta0.2-gamma8-seed1"  # your dir
    ).to(device)
    blob.eval(); base.eval()

    lora_base = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(device)

    lora = PeftModel.from_pretrained(
        lora_base,
        "/workspace/workspace/bayesian-peft/checkpoints/blob_summarization/facebook/bart-base/dialoguesum/lora-dialoguesum-bart-sample10-seed1"  # your dir
    ).to(device)
    blob.eval(); base.eval()

    ds = load_dataset("knkarthick/dialogsum")

    def summarize(m, dialogue):
        text = "Summarize: " + dialogue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            ids = m.generate(**inputs, max_new_tokens=128)
        return tokenizer.decode(ids[0], skip_special_tokens=True)

    for i in range(10):
        sample = ds["test"][i]
        d = sample["dialogue"]; gold = sample["summary"]
        bart_sum = summarize(base, d)
        blob_sum = summarize(blob, d)
        lora_sum = summarize(lora, d)
        print(f"\n=== Example {i} ===")
        print("GOLD:", gold)
        print("BART:", bart_sum)
        print("BLoB:", blob_sum)
        print("LoRA:", lora_sum)

@app.function(
    image=image,
    gpu="A100",
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],
)
def compare_logits():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel
    from datasets import load_dataset

    device = "cuda"
    ds = load_dataset("knkarthick/dialogsum")

    bart = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    bart_for_blob = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(device)
    blob = PeftModel.from_pretrained(
        bart_for_blob,
        "/workspace/workspace/bayesian-peft/checkpoints/blob_summarization/facebook/bart-base/dialoguesum/blob_summarization-dialoguesum-sample10-eps0.05-kllr0.002-beta0.2-gamma8-seed1",
    ).to(device)


    dialogue = ds["test"][0]["dialogue"]
    inputs = tokenizer("Summarize: " + dialogue, return_tensors="pt").to(device)

    with torch.no_grad():
        logits_bart = bart(**inputs).logits
        logits_blob = blob(**inputs).logits

    print("max |Δlogits| =", (logits_bart - logits_blob).abs().max().item())


@app.function(
    image=image,
    gpu="A100",                
    timeout=60 * 60 * 12,
    volumes={"/workspace": repo_volume},
    secrets=[wandb_secret],  
)
def lora_dialoguesum_summarization_test():

    os.chdir("/workspace/workspace/bayesian-peft")

    subprocess.run(["bash", "scripts/lora/lora-sum-debug.sh"], check=True)