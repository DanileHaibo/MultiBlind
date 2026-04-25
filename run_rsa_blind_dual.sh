export PYTHONPATH=$PYTHONPATH:"./mmdetection3d:./"
# 需要旧版 mmcv.Config 的环境（如 conda activate maptr）；勿用 base
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate maptr 2>/dev/null || true
fi

SCRIPT_START_TIME=$(date +%s)

attack_type="blind_dual"
attack_loss="rsa"
dataset="asymmetric"
model="maptr-bevpool"

tag=""
device_id=0

echo "=== Step 1/2: Running Dual Blind Attack ==="
echo "RSA dual-blind (two lens flares) on ${dataset}"

attack_options="attack.type=${attack_type} attack.loss=${attack_loss} attack.dataset=${dataset} attack.tag=${tag}"

show_dir="dataset/${model}"
config_file="./projects/configs/maptr/maptr_tiny_r50_24e_bevpool_asymmetric.py"
checkpoint_file="./ckpts/maptr_tiny_r50_24e_bevpool.pth"
attack_config_file="./attack_toolkit/configs/attack_cfg.yaml"

python -W ignore tools/attack.py $config_file $checkpoint_file \
    --attack_config_file $attack_config_file \
    --attack-options $attack_options \
    --show-dir $show_dir \
    --device-id $device_id \
    --eval chamfer

echo "=== Step 2/2: Running Planning ==="

exp_dir="dataset/${model}/train_${attack_type}_${attack_loss}_${dataset}"
if [ -n "$tag" ]; then
    exp_dir="${exp_dir}_${tag}"
fi
gt_traj_dir="${exp_dir}/results/planning/gt"
clean_traj_dir="${exp_dir}/results/planning/clean"
attack_traj_dir="${exp_dir}/results/planning/attack"

python attack_toolkit/src/planners/HybridAStar_planner.py \
    --dataset $dataset \
    --root_dir $exp_dir \
    --gt_traj_dir $gt_traj_dir \
    --clean_traj_dir $clean_traj_dir \
    --attack-options $attack_options \
    --collision_threshold 0.5

SCRIPT_END_TIME=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))

echo "=== Experiment Complete ==="
echo "Results saved to: $exp_dir"
echo "Total execution time: ${TOTAL_DURATION} seconds ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)"
