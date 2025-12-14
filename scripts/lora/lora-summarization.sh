model="facebook/bart-base"
modelwrapper="blob_summarization"    # <--- keep this, wrapper exists
eps=0.05     # ignored if we disable blob noise
kllr=0.0     # KL lr zero for safety
beta=0.2
gamma=8

for seed in 1 2 3; do
  name="lora-dialoguesum-bart-seed${seed}"

  python -m run.main \
    --dataset-type oedataset --dataset dialoguesum --model "$model" \
    --model-type seq2seq --modelwrapper "$modelwrapper" \
    --lr 5e-4 --batch-size 16 \
    --opt adamw --warmup-ratio 0.06 \
    --max-seq-len 512 \
    --seed "$seed" \
    --wandb-name "$name" --wandb-project "BLoB-dialoguesum-bart" \
    --apply-qkv-head-lora --lora-r 8 --lora-alpha 8 --lora-dropout 0 \
    --log-path "$name" \
    --max-train-steps 12500 \
    --eval-per-steps 2000 \
    --disable_blob_noise \
    --bayes-kllr "$kllr"
done