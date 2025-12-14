import torch
import torch.nn as nn
from torch.optim import SGD
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
import math
from tqdm import tqdm
import evaluate

from .blob import BLoB
from .wrapperbase import AverageMeter, get_linear_schedule_with_warmup
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from transformers import PreTrainedModel, AutoTokenizer
from peft.config import PeftConfig

def get_parser() -> ArgumentParser:
    # Reuse BLoB parser but we might want to add summarization specific args if needed
    # For now, just use the one from blob.py or import it?
    # We can just import get_parser from blob
    from .blob import get_parser as blob_get_parser
    return blob_get_parser()

class BLoBSummarization(BLoB):
    """BLoB model for Summarization."""

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        args,
        accelerator,
        adapter_name: str = "default",
    ):
        super().__init__(model, peft_config, args, accelerator, adapter_name)
        self.rouge = evaluate.load("rouge")

        self.global_train_step = 0
        # use max_train_steps from args; fall back to 1 to avoid div-by-zero
        total_steps = max(1, int(getattr(self.args, "max_train_steps", 1)))
        self.kl_warmup_steps = max(1, int(0.2 * total_steps))  # 20% warmup

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True, padding_side="right"
        )
        self.tokenizer

    def forward_logits(self, batch, sample=True, n_samples=1, **kwargs) -> torch.Tensor:
        """
        Forward pass that returns stacked logits for seq2seq models.
        """
        if len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets, _ = batch

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        label_ids = targets["input_ids"]  # keep for reference, but don't pass to model

        if not sample:
            #self.sample(self.base_model, False) 
            # Don't pass labels - just get logits
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label_ids,  # Use decoder_input_ids instead
            )
            logits = outputs.logits
            #self.sample(self.base_model, True) 
            return logits.unsqueeze(1)

        # Bayesian sampling
        logits_list = []
        for _ in range(n_samples):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label_ids,  # Use decoder_input_ids instead
            )
            logits_list.append(outputs.logits)
        return torch.stack(logits_list, dim=1)

    def fit(self, train_loader, eval_loader):
        nll_losses = AverageMeter()
        kl_losses = AverageMeter()
        elbo_losses = AverageMeter()
        # accs = AverageMeter() # Not using accuracy for summarization
        samples_seen = 0

        # Debug info
        if self.accelerator.is_local_main_process:
            print(
                "[DEBUG] BLoBSummarization.fit:",
                "disable_blob_noise =", getattr(self, "disable_blob_noise", None),
                "klreweighting =", getattr(self, "klreweighting", None),
                "bayes_kllr =", getattr(self.args, "bayes_kllr", None),
            )
            
        # Loss function for seq2seq NLL; ignore padding tokens
        ignore_index = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)

        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {self.args.epoch+1}/{self.args.n_epochs}",
            leave=False,
        ) as pbar:
            for i, batch in enumerate(train_loader):
                if self.args.dataset_type != "oedataset":
                    raise NotImplementedError(
                        f"Dataset type {self.args.dataset_type} not implemented for BLoBSummarization."
                    )

                # Train loader from LMDataset.s2s_train_collate_fn_ori returns (inputs, targets)
                if len(batch) == 2:
                    prompts, targets = batch
                else:
                    prompts, targets, _ = batch

                # Forward with sampling
                # logits: (batch, n_samples, seq_len, vocab_size)
                logits_stacked = self.forward_logits(
                    (prompts, targets), sample=True, n_samples=self.train_n_samples
                )
                
                # Average logits over samples
                logits = logits_stacked.mean(1) 
                
                # Compute NLL loss
                # Flatten logits and labels: logits (B*T, V), labels (B*T)
                label_ids = targets["input_ids"]
                nll = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    label_ids.view(-1),
                )

                self.accelerator.backward(nll)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                # KL Divergence
                if not getattr(self, "disable_blob_noise", False):
                    kl_divs = []
                    for _ in range(self.train_n_samples):
                        if hasattr(self.base_model, "module"):
                            kl_divs.append(self.div_posterior_prior(self.base_model.module))
                        else:
                            kl_divs.append(self.div_posterior_prior(self.base_model))
                    kl = torch.mean(torch.stack(kl_divs), dim=0)


                    if self.klreweighting:
                        if self.i % self.M == 0:
                            i = self.M
                        else:
                            i = self.i % self.M
                        self.pi = 2**i / (2 ** (self.M + 1) - 1)
                        self.i += 1
                    else:
                        self.pi = 1 / self.M

                    kl_div = kl * self.pi
                    self.accelerator.backward(kl_div)
                    self.opt2.step()
                    self.opt2.zero_grad()
                    self.scheduler2.step()
                else:
                    kl = torch.zeros((), device=logits.device)
                    kl_div = torch.zeros((), device=logits.device)

                loss, nll_loss, kl = (
                    (kl + nll).detach().cpu().numpy(),
                    nll.detach().cpu().numpy(),
                    kl_div.detach().cpu().numpy(),
                )

                # Logging
                # references = self.accelerator.gather(labels) # Not strictly needed for loss logging unless we want exact count
                # Just use batch size
                len_batch = label_ids.shape[0]
                
                kl_losses.update(kl, len_batch)
                nll_losses.update(nll_loss, len_batch)
                elbo_losses.update(loss, len_batch)

                if self.accelerator.is_local_main_process:
                    if self.wandb_logger is not None:
                        self.wandb_logger.log(
                            {
                                "train_nll_loss": nll_losses.avg,
                                "kl_loss": kl_losses.avg,
                                "elbo_loss": elbo_losses.avg,
                                "lr": self.opt.param_groups[0]["lr"],
                                "pi": float(getattr(self, "pi", 0.0)),
                            }
                        )

                self.step += self.accelerator.num_processes
                pbar.update(1)
                if self.step >= self.args.eval_per_steps:
                    self.step -= self.args.eval_per_steps
                    self.evaluate(eval_loader)

    def evaluate(self, eval_loader):
        print("Evaluating...")
        self.eval()
        status = self.training

        # --- NEW: NLL setup ---
        ignore_index = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)

        total_nll = 0.0
        total_tokens = 0
        # -----------------------

        # Metrics
        rouge_sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        n_batches = 0

        for step, batch in enumerate(eval_loader):
            if self.args.dataset_type != "oedataset":
                continue

            # Eval loader from LMDataset.s2s_collate_fn returns (inputs, targets, None)
            if len(batch) == 2:
                prompts, targets = batch
            else:
                prompts, targets, _ = batch

            input_ids = prompts["input_ids"]
            attention_mask = prompts["attention_mask"]
            label_ids = targets["input_ids"]

            with torch.no_grad():
                # ========== 1) GENERATION FOR ROUGE ==========
                if getattr(self.args, "bayes_inference_notsample", False):
                    # deterministic baseline: NO BLoB noise at inference
                    self.sample(self.base_model, False)

                    generated_ids = self.base_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=128,
                        num_beams=1,
                    )

                    # re-enable sampling state for training later
                    self.sample(self.base_model, True)

                    # single set of generated ids
                    all_generated_ids = [generated_ids]

                else:
                    # Bayesian decoding: one weight sample per decode
                    summaries_per_batch = []
                    n_weight_samples = getattr(
                        self.args, "bayes_eval_n_samples_final", 1
                    )

                    for _ in range(n_weight_samples):
                        # enable 'one weight sample per decode' behaviour
                        self.set_frozen_blob_noise(self.base_model, True)

                        generated_ids = self.base_model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=128,
                            num_beams=1,
                        )

                        # disable & clear cached noise
                        self.set_frozen_blob_noise(self.base_model, False)

                        summaries_per_batch.append(generated_ids)

                    all_generated_ids = summaries_per_batch  # list of tensors
                # =============================================

                # ========== 2) NLL UNDER POSTERIOR PREDICTIVE ==========
                # Use same pattern as training: multiple weight samples → average logits
                # For LoRA (disable_blob_noise=True), this degenerates to 1 sample.
                n_eval_samples = getattr(self.args, "bayes_eval_n_samples_final", 1)
                if getattr(self, "disable_blob_noise", False):
                    n_eval_samples = 1  # deterministic LoRA

                logits_stacked = self.forward_logits(
                    (prompts, targets),
                    sample=True,
                    n_samples=n_eval_samples,
                )  # [B, S, T, V]

                # average over samples → posterior predictive logits
                logits = logits_stacked.mean(1)  # [B, T, V]

                # compute batch NLL (per token, padding ignored)
                batch_loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    label_ids.view(-1),
                )  # scalar (mean over non-pad tokens)

                # number of *non-pad* tokens in this batch
                token_mask = (label_ids != ignore_index)
                num_tokens = token_mask.sum().item()

                total_nll += batch_loss.item() * num_tokens  # undo mean
                total_tokens += num_tokens
                # ======================================================

            # ========== 3) ROUGE (averaged over samples) ==========
            decoded_labels = self.tokenizer.batch_decode(
                label_ids, skip_special_tokens=True
            )

            batch_rouges = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            for generated_ids in all_generated_ids:
                decoded_preds = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                results = self.rouge.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    use_stemmer=True,
                )
                for key in batch_rouges:
                    batch_rouges[key] += float(results[key])

            # average ROUGE over weight samples
            n_weight_samples = len(all_generated_ids)
            for key in batch_rouges:
                batch_rouges[key] /= n_weight_samples

            # accumulate
            for key in rouge_sums:
                rouge_sums[key] += batch_rouges[key]
            n_batches += 1
            # ======================================================

        # Average ROUGE
        if n_batches > 0:
            for key in rouge_sums:
                rouge_sums[key] /= n_batches

        # Final NLL (per token)
        if total_tokens > 0:
            avg_nll = total_nll / total_tokens
        else:
            avg_nll = float("nan")

        print(f"Validation ROUGE: {rouge_sums}")
        print(f"Validation NLL (per token): {avg_nll:.4f}")

        self.train(status)

        if self.accelerator.is_local_main_process:
            if self.wandb_logger is not None:
                log_dict = dict(rouge_sums)
                log_dict["val_nll"] = avg_nll
                self.wandb_logger.log(log_dict)

        return (
            rouge_sums["rouge1"],
            rouge_sums["rouge2"],
            rouge_sums["rougeL"],
            avg_nll,  # now real NLL instead of 0.0
        )
