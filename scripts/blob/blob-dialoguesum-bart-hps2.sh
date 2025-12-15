model=facebook/bart-base
modelwrapper=blob_summarization
eps=0.05
kllr=0.002

for gamma in 4; do
    for beta in 0.1 0.2 0.3; do
        for sample in 10; do
            for seed in 1; do
                name=$modelwrapper-dialoguesum-eps$eps-kllr$kllr-beta$beta-gamma$gamma
                python -m accelerate.commands.launch --num_processes=1 run/main.py \
                    --dataset-type oedataset --dataset dialoguesum --model $model \
                    --model-type seq2seq --modelwrapper $modelwrapper \
                    --lr 5e-4 --batch-size 16 \
                    --opt adamw --warmup-ratio 0.06 \
                    --max-seq-len 512 \
                    --seed $seed \
                    --wandb-name $name --wandb-project "BLoB-dialoguesum-bart-sensitivity" \
                    --apply-qkv-head-lora --lora-r 8 --lora-alpha 8 --lora-dropout 0 \
                    --log-path $name \
                    --max-train-steps 1500 \
                    --eval-per-steps 2000 \
                    --bayes-klreweighting \
                    --bayes-eps $eps --bayes-beta $beta --bayes-gamma $gamma --bayes-kllr $kllr --bayes-datasetrescaling \
                    --bayes-train-n-samples 1 --bayes-eval-n-samples $sample --bayes-eval-n-samples-final $sample
            done
        done
    done
done