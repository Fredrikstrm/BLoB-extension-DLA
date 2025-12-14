# testing model ablation for BLoB-dialoguesum-bart with different number of samples (5, 10)
model=facebook/bart-base
modelwrapper=blob_summarization
eps=0.05
kllr=0.002
beta=0.2
gamma=8

for sample in 0 5 10; do
    for seed in 1 2; do   # maybe only 2 seeds here to save compute
        name=$modelwrapper-dialoguesum-sample$sample-eps$eps-kllr$kllr-beta$beta-gamma$gamma-seed$seed
        python -m accelerate.commands.launch --num_processes=1 run/main.py \
            --dataset-type oedataset --dataset dialoguesum --model $model \
            --model-type seq2seq --modelwrapper $modelwrapper \
            --lr 5e-4 --batch-size 16 \
            --opt adamw --warmup-ratio 0.06 \
            --max-seq-len 512 \
            --seed $seed \
            --wandb-name $name --wandb-project "BLoB-dialoguesum-bart" \
            --apply-qkv-head-lora --lora-r 8 --lora-alpha 8 --lora-dropout 0 \
            --log-path $name \
            --max-train-steps 5000 \
            --eval-per-steps 6000 \
            --bayes-klreweighting \
            --bayes-eps $eps --bayes-beta $beta --bayes-gamma $gamma --bayes-kllr $kllr --bayes-datasetrescaling \
            --bayes-train-n-samples 1 --bayes-eval-n-samples $sample --bayes-eval-n-samples-final $sample
    done
done