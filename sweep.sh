#!/usr/bin/env bash
# Usage: bash sweep.sh <study>
#
# Studies (run in order):
#   lr          Learning rate
#   target_kl   Target KL (adaptive controller)
#   kl_mode     KL penalty mode & fixed-β values
#   adv_norm    Advantage normalization
#   clip        PPO clipping range
#   ppo_epochs  Number of PPO update epochs per rollout
#   noisy_reward Fraction of reward labels randomly corrupted
#   gamma       Discount factor
#   lam         GAE lambda
#   critic      Value function loss coefficient
#
# Each non-lr study also runs all its configurations at lr=1e-6
# (run names suffixed with _lr1e6) in addition to the baseline lr=5e-6.
#
# Stdout for each run is saved to outputs/<experiment>/<run_name>/stdout.log
# and also printed to the terminal.
#
# Example:
#   bash sweep.sh lr

set -euo pipefail

PYTHON="python3"
STUDY="${1:-}"
TAG="${2:-}"          # optional: bash sweep.sh lr v2

run() {
    local experiment="$1"
    local run_name="$2"
    shift 2
    [[ -n "${TAG}" ]] && run_name="${run_name}_${TAG}"
    local out_dir="outputs/${experiment}/${run_name}"
    if [[ -f "${out_dir}/metrics.jsonl" ]]; then
        echo ""
        echo "  SKIP ${experiment}/${run_name} (metrics.jsonl already exists)"
        return
    fi
    mkdir -p "${out_dir}"
    echo ""
    echo "================================================================"
    echo "  STUDY=${experiment}  RUN=${run_name}"
    echo "================================================================"
    "${PYTHON}" run_tinystories.py \
        --experiment        "${experiment}" \
        --run_name          "${run_name}" \
        --save_every_n_steps 0 \
        --num_epochs 2 \
        --kl_horizon 500 \
        "$@" 2>&1 | tee "${out_dir}/stdout.log"
}

case "${STUDY}" in

    lr)
        run lr lr_1e-6 --learning_rate 1e-6
        run lr lr_5e-6 --learning_rate 5e-6

        run lr lr_1e-5 --learning_rate 1e-5
        run lr lr_5e-5 --learning_rate 5e-5
        ;;

    target_kl)
        run target_kl tkl_1_lr5e6 --learning_rate 5e-6 --target_kl 1.0
        run target_kl tkl_5_lr5e6 --learning_rate 5e-6 --target_kl 5.0
        run target_kl tkl_10_lr5e6 --learning_rate 5e-6 --target_kl 10.0
        run target_kl tkl_20_lr5e6 --learning_rate 5e-6 --target_kl 20.0

        run target_kl tkl_1_lr1e6  --learning_rate 1e-6 --target_kl 1.0
        run target_kl tkl_5_lr1e6  --learning_rate 1e-6 --target_kl 5.0
        run target_kl tkl_10_lr1e6 --learning_rate 1e-6 --target_kl 10.0
        run target_kl tkl_20_lr1e6 --learning_rate 1e-6 --target_kl 20.0
        ;;

    kl_mode)
        run kl_mode kl_adaptive_lr5e6    --learning_rate 5e-6 --kl_mode adaptive
        run kl_mode kl_fixed_b0.1_lr5e6  --learning_rate 5e-6 --kl_mode fixed --init_kl_coef 0.1
        run kl_mode kl_fixed_b0.3_lr5e6  --learning_rate 5e-6 --kl_mode fixed --init_kl_coef 0.3
        run kl_mode kl_fixed_b0.5_lr5e6  --learning_rate 5e-6 --kl_mode fixed --init_kl_coef 0.5
        run kl_mode kl_none_lr5e6        --learning_rate 5e-6 --kl_mode none

        run kl_mode kl_adaptive_lr1e6    --learning_rate 1e-6 --kl_mode adaptive
        run kl_mode kl_fixed_b0.1_lr1e6  --learning_rate 1e-6 --kl_mode fixed --init_kl_coef 0.1
        run kl_mode kl_fixed_b0.3_lr1e6  --learning_rate 1e-6 --kl_mode fixed --init_kl_coef 0.3
        run kl_mode kl_fixed_b0.5_lr1e6  --learning_rate 1e-6 --kl_mode fixed --init_kl_coef 0.5
        run kl_mode kl_none_lr1e6        --learning_rate 1e-6 --kl_mode none
        ;;

    adv_norm)
        run adv_norm adv_global_lr5e6 --learning_rate 5e-6 --advantage_normalize global
        run adv_norm adv_batch_lr5e6  --learning_rate 5e-6 --advantage_normalize batch
        run adv_norm adv_none_lr5e6   --learning_rate 5e-6 --advantage_normalize none

        run adv_norm adv_global_lr1e6 --learning_rate 1e-6 --advantage_normalize global
        run adv_norm adv_batch_lr1e6  --learning_rate 1e-6 --advantage_normalize batch
        run adv_norm adv_none_lr1e6   --learning_rate 1e-6 --advantage_normalize none
        ;;

    clip)
        run clip clip_005_lr5e6  --learning_rate 5e-6 --cliprange 0.05 --cliprange_value 0.05
        run clip clip_02_lr5e6   --learning_rate 5e-6 --cliprange 0.2  --cliprange_value 0.2
        run clip clip_none_lr5e6 --learning_rate 5e-6 --cliprange 1e9  --cliprange_value 1e9

        run clip clip_005_lr1e6  --learning_rate 1e-6 --cliprange 0.05 --cliprange_value 0.05
        run clip clip_02_lr1e6   --learning_rate 1e-6 --cliprange 0.2  --cliprange_value 0.2
        run clip clip_none_lr1e6 --learning_rate 1e-6 --cliprange 1e9  --cliprange_value 1e9
        ;;

    ppo_epochs)
        run ppo_epochs ppo_e1_lr5e6 --learning_rate 5e-6 --ppo_epochs 1
        run ppo_epochs ppo_e2_lr5e6 --learning_rate 5e-6 --ppo_epochs 2
        run ppo_epochs ppo_e4_lr5e6 --learning_rate 5e-6 --ppo_epochs 4
        run ppo_epochs ppo_e8_lr5e6 --learning_rate 5e-6 --ppo_epochs 8

        run ppo_epochs ppo_e1_lr1e6 --learning_rate 1e-6 --ppo_epochs 1
        run ppo_epochs ppo_e2_lr1e6 --learning_rate 1e-6 --ppo_epochs 2
        run ppo_epochs ppo_e4_lr1e6 --learning_rate 1e-6 --ppo_epochs 4
        run ppo_epochs ppo_e8_lr1e6 --learning_rate 1e-6 --ppo_epochs 8
        ;;

    noisy_reward)
        run noisy_reward noise_0_lr5e6   --learning_rate 5e-6 --reward_noise_frac 0.0
        run noisy_reward noise_0.1_lr5e6 --learning_rate 5e-6 --reward_noise_frac 0.1
        run noisy_reward noise_0.2_lr5e6 --learning_rate 5e-6 --reward_noise_frac 0.2
        run noisy_reward noise_0.3_lr5e6 --learning_rate 5e-6 --reward_noise_frac 0.3
        run noisy_reward noise_0.4_lr5e6 --learning_rate 5e-6 --reward_noise_frac 0.4
        run noisy_reward noise_0.5_lr5e6 --learning_rate 5e-6 --reward_noise_frac 0.5

        run noisy_reward noise_0_lr1e6   --learning_rate 1e-6 --reward_noise_frac 0.0
        run noisy_reward noise_0.1_lr1e6 --learning_rate 1e-6 --reward_noise_frac 0.1
        run noisy_reward noise_0.2_lr1e6 --learning_rate 1e-6 --reward_noise_frac 0.2
        run noisy_reward noise_0.3_lr1e6 --learning_rate 1e-6 --reward_noise_frac 0.3
        run noisy_reward noise_0.4_lr1e6 --learning_rate 1e-6 --reward_noise_frac 0.4
        run noisy_reward noise_0.5_lr1e6 --learning_rate 1e-6 --reward_noise_frac 0.5

        ;;

    gamma)
        run gamma gamma_095_lr5e6 --learning_rate 5e-6 --gamma 0.95
        run gamma gamma_099_lr5e6 --learning_rate 5e-6 --gamma 0.99
        run gamma gamma_1_lr5e6   --learning_rate 5e-6 --gamma 1.0

        run gamma gamma_095_lr1e6 --learning_rate 1e-6 --gamma 0.95
        run gamma gamma_099_lr1e6 --learning_rate 1e-6 --gamma 0.99
        run gamma gamma_1_lr1e6   --learning_rate 1e-6 --gamma 1.0
        ;;

    lam)
        run lam lam_08_lr5e6  --learning_rate 5e-6 --lam 0.8
        run lam lam_095_lr5e6 --learning_rate 5e-6 --lam 0.95
        run lam lam_1_lr5e6   --learning_rate 5e-6 --lam 1.0

        run lam lam_08_lr1e6  --learning_rate 1e-6 --lam 0.8
        run lam lam_095_lr1e6 --learning_rate 1e-6 --lam 0.95
        run lam lam_1_lr1e6   --learning_rate 1e-6 --lam 1.0
        ;;

    critic)
        run critic vf_01_lr5e6 --learning_rate 5e-6 --vf_coef 0.1
        run critic vf_05_lr5e6 --learning_rate 5e-6 --vf_coef 0.5
        run critic vf_1_lr5e6  --learning_rate 5e-6 --vf_coef 1.0

        run critic vf_01_lr1e6 --learning_rate 1e-6 --vf_coef 0.1
        run critic vf_05_lr1e6 --learning_rate 1e-6 --vf_coef 0.5
        run critic vf_1_lr1e6  --learning_rate 1e-6 --vf_coef 1.0
        ;;

    *)
        echo "Usage: bash sweep.sh {lr|target_kl|kl_mode|adv_norm|clip|ppo_epochs|noisy_reward|gamma|lam|critic}"
        echo ""
        echo "  lr           4 runs: 1e-6, 5e-6 (base), 1e-5, 5e-5"
        echo "  target_kl    8 runs: 1.0, 5.0, 10.0 (base), 20.0  x {5e-6, 1e-6}"
        echo "  kl_mode     10 runs: adaptive (base), fixed x3, none  x {5e-6, 1e-6}"
        echo "  adv_norm     6 runs: global (base), batch, none  x {5e-6, 1e-6}"
        echo "  clip         6 runs: 0.05, 0.2 (base), disabled  x {5e-6, 1e-6}"
        echo "  ppo_epochs   8 runs: 1, 2, 4 (base), 8  x {5e-6, 1e-6}"
        echo "  noisy_reward 8 runs: 0.0 (base), 0.1, 0.2, 0.3  x {5e-6, 1e-6}"
        echo "  gamma        6 runs: 0.95, 0.99, 1.0 (base)  x {5e-6, 1e-6}"
        echo "  lam          6 runs: 0.8, 0.95 (base), 1.0  x {5e-6, 1e-6}"
        echo "  critic       6 runs: vf_coef 0.1, 0.5 (base), 1.0  x {5e-6, 1e-6}"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "  Study '${STUDY}' complete."
echo "  Logs: outputs/${STUDY}/<run>/stdout.log"
echo "================================================================"