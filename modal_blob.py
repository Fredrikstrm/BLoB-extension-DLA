import modal
import os
import subprocess

app = modal.App("bayesian-peft-blob")

repo_volume = modal.Volume.from_name("bayesian-peft-repo", create_if_missing=True)

code_volume = modal.Volume.from_name("bayesian-peft-code", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("bayesian-peft-models", create_if_missing=True)

VOLUMES = {
    "/mnt/repo": code_volume,
    "/mnt/ckpt": ckpt_volume,
}

wandb_secret = modal.Secret.from_name("wandb-api-key")

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
        "safetensors",
    )
) 


@app.local_entrypoint()
def sync_code():
    with code_volume.batch_upload() as batch:
        batch.put_directory(local_path=".", remote_path="/bayesian-peft")
    print("Synced code → /mnt/repo/bayesian-peft")

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

old_volume = modal.Volume.from_name("bayesian-peft-repo")
@app.function(volumes={"/mnt/old": old_volume, "/mnt/ckpt": ckpt_volume})
def migrate_checkpoints():
    src = "/mnt/old/workspace/bayesian-peft/checkpoints"   
    dst = "/mnt/ckpt/checkpoints"
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print("Copied checkpoints → /mnt/ckpt/checkpoints")

# Roberta-Large
@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes=VOLUMES,
    secrets=[wandb_secret],
)
def run_blob_roberta_all():
    os.chdir("/mnt/repo/bayesian-peft")
    subprocess.run(["bash", "scripts/blob/blob-roberta-all.sh"], check=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes=VOLUMES,
    secrets=[wandb_secret],
)
def run_blob_roberta_base_all():
    os.chdir("/workspace/workspace/bayesian-peft_20251212_223404")
    subprocess.run(["bash", "scripts/blob/blob-roberta-base-all.sh"], check=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes=VOLUMES,
    secrets=[wandb_secret],
)
def run_blob_dialoguesum_bart():
    os.chdir("/mnt/repo/bayesian-peft")
    subprocess.run(["bash", "scripts/blob/blob-dialoguesum-bart.sh"], check=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes=VOLUMES,
    secrets=[wandb_secret],
)
def run_blob_dialoguesum_bart_new():
    os.chdir("/mnt/repo/bayesian-peft")
    subprocess.run(["bash", "scripts/blob/blob-dialoguesum-bart-new.sh"], check=True)

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes=VOLUMES,
    secrets=[wandb_secret],
)
def run_blob_dialoguesum_bart_ablation():
    os.chdir("/mnt/repo/bayesian-peft")
    subprocess.run(["bash", "scripts/blob/blob-dialoguesum-bart-ablation.sh"], check=True)

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes=VOLUMES,
    secrets=[wandb_secret],
)
def run_blob_dialoguesum_bart_hps():
    os.chdir("/mnt/repo/bayesian-peft")
    subprocess.run(["bash", "scripts/blob/blob-dialoguesum-bart-hps.sh"], check=True)


@app.function(
    image=image,
    gpu="A100",                
    timeout=60 * 60 * 12,
    volumes=VOLUMES,
    secrets=[wandb_secret],  
)
def lora_dialoguesum_summarization():

    os.chdir("/mnt/repo/bayesian-peft")

    subprocess.run(["bash", "scripts/lora/lora-summarization.sh"], check=True)

@app.function( 
    image=image,
    gpu="A100",
    volumes=VOLUMES,
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
        "/mnt/ckpt/checkpoints/blob_summarization/facebook/bart-base/dialoguesum/blob_summarization-dialoguesum-sample10-eps0.05-kllr0.002-beta0.2-gamma8-seed1"  # your dir
    ).to(device)
    blob.eval(); base.eval()

    lora_base = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(device)

    lora = PeftModel.from_pretrained(
        lora_base,
        "/mnt/ckpt/checkpoints/blob_summarization/facebook/bart-base/dialoguesum/lora-dialoguesum-bart-seed1"  # your dir
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
    volumes=VOLUMES,
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
        "/mnt/ckpt/checkpoints/blob_summarization/facebook/bart-base/dialoguesum/blob_summarization-dialoguesum-sample10-eps0.05-kllr0.002-beta0.2-gamma8-seed1",
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
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=[wandb_secret],
)
def sample_summaries_blob_gpu_clean(
    blob_ckpt_dir: str = "/mnt/ckpt/checkpoints/blob_summarization/facebook/bart-base/dialoguesum/blob_summarization-dialoguesum-sample10-eps0.05-kllr0.002-beta0.2-gamma8-seed3",
    split: str = "test",
    num_examples: int = 1,
    num_weight_samples: int = 10,
    max_new_tokens: int = 128,
):
    import os, sys
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from accelerate import Accelerator
    from peft import PeftConfig, PeftModel

    # --- make repo importable ---
    os.chdir("/mnt/repo/bayesian-peft")
    sys.path.insert(0, os.getcwd())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator()

    ds = load_dataset("knkarthick/dialogsum")

    # load base + tokenizer 
    base = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(device)
    tok = AutoTokenizer.from_pretrained("facebook/bart-base")

    # construct args via parser (minimal required) 
    from modelwrappers.blob_summarization import get_parser, BLoBSummarization

    parser = get_parser()
    args = parser.parse_args(
        [
            "--dataset-type", "oedataset",
            "--modelwrapper", "blob_summarization",
            "--model", "facebook/bart-base",
            "--model-type", "seq2seq",
        ]
    )
    args.dataset = "dialoguesum"

    # keep BLoB noise ON
    args.disable_blob_noise = False
    args.batch_size = 1
    args.outdim = 0
    args.num_samples = 1
    args.n_epochs = 1

    # init wrapper 
    peft_config = PeftConfig.from_pretrained(blob_ckpt_dir)
    model = BLoBSummarization(
        model=base,
        peft_config=peft_config,
        args=args,
        accelerator=accelerator,
        adapter_name="default",
    )

    # Attach adapter to the PEFT-wrapped model using PEFT's loader
    if hasattr(model.base_model, "load_adapter"):
        model.base_model.load_adapter(blob_ckpt_dir, adapter_name="default", is_trainable=False)
        model.base_model.set_adapter("default")
    else:
        # fallback (for older PEFT versions)
        model.base_model = PeftModel.from_pretrained(
            model.base_model,
            blob_ckpt_dir,
            adapter_name="default",
            is_trainable=False,
        )
        model.base_model.set_adapter("default")

    model.base_model.set_adapter("default")

    model.base_model.to(device)
    model.base_model.eval()

    # Reliable freezer: directly flips flags on BLoB modules
    def freeze_blob_noise(peft_model, freeze: bool):
        n = 0
        for m in peft_model.modules():
            if hasattr(m, "blobsample"):
                n += 1
                # ensure BLoB sampling path is active
                m.blobsample = True
                m.use_frozen_blob_noise = freeze
                if not freeze:
                    if hasattr(m, "frozen_lora_noise_a"):
                        m.frozen_lora_noise_a = None
                    if hasattr(m, "frozen_r_A"):
                        m.frozen_r_A = None
                    if hasattr(m, "frozen_s_A"):
                        m.frozen_s_A = None
        return n
    
    # Force init of frozen noise tensors by doing a forward pass
    @torch.no_grad()
    def force_init_frozen_noise(enc):
        start_id = model.base_model.config.decoder_start_token_id or tok.bos_token_id
        dec_ids = torch.tensor([[start_id]], device=device)
        _ = model.base_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=dec_ids,
            use_cache=False,
            return_dict=True,
        )

    # Manual greedy decoder
    @torch.no_grad()
    def greedy_decode_one_sample(dialogue: str) -> str:
        enc = tok(
            "Summarize: " + dialogue,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        encoder = model.base_model.get_encoder()
        enc_out = encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            return_dict=True,
        )

        start_id = model.base_model.config.decoder_start_token_id or tok.bos_token_id
        dec_ids = torch.tensor([[start_id]], device=device)

        eos_id = model.base_model.config.eos_token_id

        for _ in range(max_new_tokens):
            out = model.base_model(
                encoder_outputs=enc_out,
                attention_mask=enc["attention_mask"],
                decoder_input_ids=dec_ids,
                use_cache=False,
                return_dict=True,
            )
            next_logits = out.logits[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            dec_ids = torch.cat([dec_ids, next_id], dim=1)
            if eos_id is not None and int(next_id.item()) == int(eos_id):
                break

        gen = dec_ids[:, 1:]
        return tok.decode(gen[0], skip_special_tokens=True).strip()

    # Run examples
    for i in range(num_examples):
        ex = ds[split][i]
        dialogue = ex["dialogue"]
        gold = ex["summary"]

        print("\n" + "=" * 80)
        print(f"Example {i} | split={split}")
        print("GOLD:", gold)

        enc_dbg = tok(
            "Summarize: " + dialogue,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        start_id = model.base_model.config.decoder_start_token_id or tok.bos_token_id

        # prove sampling affects logits (A vs B) 
        freeze_blob_noise(model.base_model, True)
        force_init_frozen_noise(enc_dbg)
        freeze_blob_noise(model.base_model, False)

        freeze_blob_noise(model.base_model, True)
        force_init_frozen_noise(enc_dbg)
        freeze_blob_noise(model.base_model, False)

        # actual summary samples 
        for s in range(num_weight_samples):
            freeze_blob_noise(model.base_model, True)

            # force init before printing sig so it won't be None just because no forward happened
            force_init_frozen_noise(enc_dbg)

            # grab one frozen noise norm (after init)
            for m in model.base_model.modules():
                t = getattr(m, "frozen_lora_noise_a", None)
                if t is not None:
                    break

            txt = greedy_decode_one_sample(dialogue)

            freeze_blob_noise(model.base_model, False)

            print(f"\n[BLoB weight-sample {s+1}/{num_weight_samples}] {txt}")