#!/usr/bin/env bash

set -u
set -o pipefail

repo_dir="/home/kedong/projects/cs336/assignments/assignment1-basics"
cd "$repo_dir" || exit 1

mkdir -p experiment_logs experiment_state reports
run_tmp="$(mktemp -d /tmp/cs336-overnight.XXXXXX)"
export WANDB_MODE=offline
export TQDM_DISABLE=1

make_config() {
    local path="$1"
    local lr="$2"
    local min_lr="$3"
    local batch_size="$4"
    local max_steps="$5"

    {
        echo "optimizer:"
        echo "  lr: $lr"
        echo "  min_lr: $min_lr"
        echo "  warmup_steps: 500"
        echo "  cos_steps: 10000"
        echo
        echo "runtime:"
        echo "  max_steps: $max_steps"
        echo "  batch_size: $batch_size"
        echo "  eval_interval: 100"
        echo "  log_interval: 10"
        echo "  checkpoint_interval: 500"
    } > "$path"
}

run_one() {
    local exp_name="$1"
    local lr="$2"
    local min_lr="$3"
    local batch_size="$4"
    local max_steps="$5"
    local restore_path="${6:-}"
    local config_path="$run_tmp/$exp_name.yaml"
    local log_path="experiment_logs/$exp_name.log"

    if [[ -f "checkpoints/$exp_name/final.ckpt" ]]; then
        echo "SKIP $exp_name: final checkpoint already exists" | tee -a "$log_path"
        return 0
    fi

    make_config "$config_path" "$lr" "$min_lr" "$batch_size" "$max_steps"
    echo "START $exp_name lr=$lr min_lr=$min_lr batch=$batch_size steps=$max_steps" | tee "$log_path"

    if [[ -n "$restore_path" ]]; then
        python3 tests/trainer.py \
            data/tiny_story_train.npy data/tiny_story_val.npy \
            --exp_name "$exp_name" \
            --config_path "$config_path" \
            --restore_path "$restore_path" 2>&1 | tee -a "$log_path"
    else
        python3 tests/trainer.py \
            data/tiny_story_train.npy data/tiny_story_val.npy \
            --exp_name "$exp_name" \
            --config_path "$config_path" 2>&1 | tee -a "$log_path"
    fi

    local trainer_status="${PIPESTATUS[0]}"
    echo "END $exp_name status=$trainer_status" | tee -a "$log_path"
    return "$trainer_status"
}

tail_mean() {
    awk '
        /^step [0-9]+ val\/loss,/ {
            value = $4
            sub(/[^0-9eE+.-].*$/, "", value)
            values[count % 5] = value
            count += 1
        }
        END {
            n = count < 5 ? count : 5
            if (n == 0) {
                print "inf"
                exit
            }
            sum = 0
            for (i = 0; i < n; i += 1) {
                sum += values[i]
            }
            printf "%.10f\n", sum / n
        }
    ' "$1"
}

final_val() {
    awk '
        /^step [0-9]+ val\/loss,/ {
            value = $4
            sub(/[^0-9eE+.-].*$/, "", value)
        }
        END { if (value == "") print "inf"; else print value }
    ' "$1"
}

extract_val_curve() {
    local input_path="$1"
    local output_path="$2"
    awk '
        BEGIN { print "step,val_loss" }
        /^step [0-9]+ val\/loss,/ {
            step = $2
            gsub(",", "", step)
            value = $4
            sub(/[^0-9eE+.-].*$/, "", value)
            print step "," value
        }
    ' "$input_path" > "$output_path"
}

extract_val_curve_from_wandb_binary() {
    local input_path="$1"
    local output_path="$2"
    {
        echo "step,val_loss"
        strings -a "$input_path" |
            rg -o 'step [0-9]+ val/loss, [0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' |
            awk '{ gsub(",", "", $2); print $2 "," $4 }'
    } > "$output_path"
}

echo "Waiting for the already-running lr=3e-4 experiment."
while [[ ! -f checkpoints/overnight_lr_b128_3e-4_s3000/final.ckpt ]]; do
    sleep 30
done

lr3e4_log="experiment_state/lr3e4_tail_metrics.txt"
baseline_log="wandb/run-20260725_215230-xmmmgnvo/files/output.log"
lr3e3_log="experiment_logs/overnight_lr_b128_3e-3_s3000.log"

run_one "overnight_lr_b128_3e-3_s3000" "3.0e-3" "3.0e-4" 128 3000 || true

baseline_score="$(tail_mean "$baseline_log")"
lr3e4_score="$(tail_mean "$lr3e4_log")"
lr3e3_score="$(tail_mean "$lr3e3_log")"

{
    echo -e "label\tlr\ttail5_mean"
    echo -e "1e-3\t1.0e-3\t$baseline_score"
    echo -e "3e-4\t3.0e-4\t$lr3e4_score"
    echo -e "3e-3\t3.0e-3\t$lr3e3_score"
} > experiment_state/learning_rate_scores.tsv

ranking="$(
    {
        echo -e "1e-3\t$baseline_score"
        echo -e "3e-4\t$lr3e4_score"
        echo -e "3e-3\t$lr3e3_score"
    } | sort -t $'\t' -k2,2g
)"
best_label="$(echo "$ranking" | sed -n '1p' | cut -f1)"
runner_up_label="$(echo "$ranking" | sed -n '2p' | cut -f1)"
echo "$best_label" > experiment_state/best_lr_label.txt

lr_for_label() {
    case "$1" in
        3e-4) echo "3.0e-4 3.0e-5" ;;
        1e-3) echo "1.0e-3 1.0e-4" ;;
        3e-3) echo "3.0e-3 3.0e-4" ;;
    esac
}

restore_for_label() {
    case "$1" in
        3e-4) echo "checkpoints/overnight_lr_b128_3e-4_s3000/final.ckpt" ;;
        1e-3) echo "checkpoints/tiny_1e-3/final.ckpt" ;;
        3e-3) echo "checkpoints/overnight_lr_b128_3e-3_s3000/final.ckpt" ;;
    esac
}

read -r best_lr best_min_lr <<< "$(lr_for_label "$best_label")"
best_restore="$(restore_for_label "$best_label")"
best_10k_exp="overnight_lr_b128_${best_label}_s10000_resume"

if ! run_one "$best_10k_exp" "$best_lr" "$best_min_lr" 128 10000 "$best_restore"; then
    best_10k_exp="overnight_lr_b128_${best_label}_s10000_fresh"
    run_one "$best_10k_exp" "$best_lr" "$best_min_lr" 128 10000 || true
fi

best_10k_log="experiment_logs/$best_10k_exp.log"
best_10k_final="$(final_val "$best_10k_log")"
if awk -v loss="$best_10k_final" 'BEGIN { exit !(loss > 1.45) }'; then
    read -r runner_lr runner_min_lr <<< "$(lr_for_label "$runner_up_label")"
    runner_restore="$(restore_for_label "$runner_up_label")"
    run_one \
        "overnight_lr_b128_${runner_up_label}_s10000_resume" \
        "$runner_lr" "$runner_min_lr" 128 10000 "$runner_restore" || true
fi

case "$best_label" in
    3e-4)
        b1_specs=("2.34375e-6 2.34375e-7")
        b32_specs=("3.75e-5 3.75e-6" "7.5e-5 7.5e-6" "1.5e-4 1.5e-5")
        b64_specs=("7.5e-5 7.5e-6" "1.5e-4 1.5e-5" "3.0e-4 3.0e-5")
        ;;
    1e-3)
        b1_specs=("7.8125e-6 7.8125e-7")
        b32_specs=("1.25e-4 1.25e-5" "2.5e-4 2.5e-5" "5.0e-4 5.0e-5")
        b64_specs=("2.5e-4 2.5e-5" "5.0e-4 5.0e-5" "1.0e-3 1.0e-4")
        ;;
    3e-3)
        b1_specs=("2.34375e-5 2.34375e-6")
        b32_specs=("3.75e-4 3.75e-5" "7.5e-4 7.5e-5" "1.5e-3 1.5e-4")
        b64_specs=("7.5e-4 7.5e-5" "1.5e-3 1.5e-4" "3.0e-3 3.0e-4")
        ;;
esac

for spec in "${b1_specs[@]}"; do
    read -r lr min_lr <<< "$spec"
    run_one "overnight_batch1_lr${lr}_s1000" "$lr" "$min_lr" 1 1000 || true
done

for spec in "${b32_specs[@]}"; do
    read -r lr min_lr <<< "$spec"
    run_one "overnight_batch32_lr${lr}_s1000" "$lr" "$min_lr" 32 1000 || true
done

for spec in "${b64_specs[@]}"; do
    read -r lr min_lr <<< "$spec"
    run_one "overnight_batch64_lr${lr}_s1000" "$lr" "$min_lr" 64 1000 || true
done

mkdir -p reports/data
extract_val_curve "$baseline_log" reports/data/lr_b128_1e-3_s3000_val.csv
extract_val_curve_from_wandb_binary \
    wandb/offline-run-20260725_233707-p6vr636b/run-p6vr636b.wandb \
    reports/data/lr_b128_3e-4_s3000_val.csv
extract_val_curve "$lr3e3_log" reports/data/lr_b128_3e-3_s3000_val.csv
extract_val_curve \
    experiment_logs/overnight_lr_b128_1e-3_s10000_resume.log \
    reports/data/lr_b128_1e-3_s10000_resume_val.csv
extract_val_curve \
    experiment_logs/overnight_lr_b128_3e-3_s10000_resume.log \
    reports/data/lr_b128_3e-3_s10000_resume_val.csv

for log_path in experiment_logs/overnight_batch*_s1000.log; do
    if [[ -f "$log_path" ]]; then
        output_name="$(basename "$log_path" .log)_val.csv"
        extract_val_curve "$log_path" "reports/data/$output_name"
    fi
done

touch experiment_state/stage12_complete
echo "Stage 1 and stage 2 experiments complete."
