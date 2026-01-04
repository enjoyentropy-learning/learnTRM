#!/bin/bash
set -e
export DISABLE_COMPILE=1

run_name="pretrain_mlp_t_sudoku_trmModified2"

python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=25000 \
  lr=1e-4 puzzle_emb_lr=1e-4 \
  weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=${run_name} \
  ema=True


#git clone https://github.com/enjoyentropy-learning/learnTRM.git
#git switch TrmModified


#pip install --upgrade pip wheel setuptools


#nvidia-smi

# To check Python cuda stuff
#  305  python -c "import torch; print(torch.__version__)"
#  306  pwd
#  307  ls
#  308  python - <<EOF
#import torch
#print("Torch:", torch.__version__)
#print("CUDA available:", torch.cuda.is_available())
#print("CUDA version:", torch.version.cuda)
#EOF

# The above will probably work so nothing to be done.
#The below command if required run without index-url
#pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version


#pip install -r requirements.txt
#wandb login

#python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

#nohup ./run_pre.sh > pretrain.out 2>&1 &

# tail -f pretrain.out for log file
# GPU Status
#nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 2
#watch -n 1 nvidia-smi

