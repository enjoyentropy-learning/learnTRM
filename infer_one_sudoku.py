import os
import sys
import torch
from omegaconf import OmegaConf
import numpy as np
import argparse

# Load ONE test example from preprocessed dataset
DATA_DIR = "data/sudoku-extreme-1k-aug-1000"
INDEX = 2 # choose test index, index=2 results in answer in multiple iterations. index=8 one iteration, index=25 can't solve

#python3 infer_one_sudoku.py --index 8
parser = argparse.ArgumentParser()
parser.add_argument(
    "--index",
    type=int,
    default=0,
    help="Index of test Sudoku to run inference on"
)
args = parser.parse_args()
INDEX = args.index


def verify_solution_with_input(input_grid, output_grid):
    """
    input_grid:  (9,9) tensor or list, values 0–9 (0 = blank)
    output_grid: (9,9) tensor or list, values 1–9

    returns: (is_valid: bool, message: str)
    """
    import torch

    if isinstance(input_grid, torch.Tensor):
        input_grid = input_grid.cpu().tolist()
    if isinstance(output_grid, torch.Tensor):
        output_grid = output_grid.cpu().tolist()

    digits = set(range(1, 10))

    # 1. Check input clues preserved
    for i in range(9):
        for j in range(9):
            if input_grid[i][j] != 0:
                if output_grid[i][j] != input_grid[i][j]:
                    return (
                        False,
                        f"Input clue changed at ({i},{j}): "
                        f"{input_grid[i][j]} → {output_grid[i][j]}"
                    )

    # 2. Check rows
    for i in range(9):
        if set(output_grid[i]) != digits:
            return False, f"Invalid row {i}: {output_grid[i]}"

    # 3. Check columns
    for j in range(9):
        col = [output_grid[i][j] for i in range(9)]
        if set(col) != digits:
            return False, f"Invalid column {j}: {col}"

    # 4. Check 3x3 boxes
    for bi in range(3):
        for bj in range(3):
            box = []
            for i in range(3):
                for j in range(3):
                    box.append(output_grid[3*bi + i][3*bj + j])
            if set(box) != digits:
                return False, f"Invalid box ({bi},{bj}): {box}"

    return True, "Valid solution and all input clues preserved"

# ==================================================
# 1. Paths (adjust ONLY if directory names change)
# ==================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CKPT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "Sudoku-extreme-1k-aug-1000-ACT-torch",
    "pretrain_mlp_t_sudoku_full2",
)

CKPT_PATH = os.path.join(CKPT_DIR, "step_65100")
CONFIG_PATH = os.path.join(CKPT_DIR, "all_config.yaml")

# ==================================================
# 2. Ensure we use the EXACT model code from training
# ==================================================

sys.path.insert(0, CKPT_DIR)

from trm import TinyRecursiveReasoningModel_ACTV1

# ==================================================
# 3. Load config (architecture only is used)
# ==================================================

cfg = OmegaConf.load(CONFIG_PATH)

cfg.arch.batch_size = 1
cfg.arch.seq_len = 81
cfg.arch.num_puzzle_identifiers = 1
cfg.arch.vocab_size = 11

# ==================================================
# 4. Build model
# ==================================================

model = TinyRecursiveReasoningModel_ACTV1(cfg.arch)

# ==================================================
# 5. Load checkpoint (RAW state_dict)
# ==================================================

#state_dict = torch.load(CKPT_PATH, map_location="cpu")

raw_state_dict = torch.load(CKPT_PATH, map_location="cpu")

# --------------------------------------------------
# Strip "model." prefix from checkpoint keys
# --------------------------------------------------

state_dict = {}
for k, v in raw_state_dict.items():
    if k.startswith("model."):
        state_dict[k[len("model."):]] = v
    else:
        state_dict[k] = v

model.load_state_dict(state_dict)

model.eval()
model.cpu()

print("Model loaded successfully.")
def do_not_use_load_test_batch(data_dir, indices, device):
    """
    Load a batch of Sudoku puzzles from the TEST split.

    Returns:
        model_inputs : (B, 81) LongTensor, values 1..10
        grids        : (B, 9, 9) LongTensor, values 0..9
        gt_grids     : (B, 9, 9) LongTensor, values 0..9
    """
    test_dir = os.path.join(data_dir, "test")

    inputs = np.load(os.path.join(test_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(test_dir, "all__labels.npy"))

    # Ensure indices is iterable
    if isinstance(indices, int):
        indices = [indices]

    # Model inputs (keep tokenization!)
    model_inputs = torch.tensor(
        inputs[indices],
        dtype=torch.long,
        device=device
    )  # (B, 81), values 1..10

    # Human-readable grids
    grids = torch.tensor(
        inputs[indices] - 1,
        dtype=torch.long,
        device=device
    ).view(-1, 9, 9)

    gt_grids = torch.tensor(
        labels[indices] - 1,
        dtype=torch.long,
        device=device
    ).view(-1, 9, 9)

    return model_inputs, grids, gt_grids


# ==================================================
# 6. Sudoku input (81 chars, '.' or '0' = blank)
# ==================================================

#input from test file

sudoku_str1 = (
    "53..7...."
    "6..195..."
    ".98....6."
    "8...6...3"
    "4..8.3..1"
    "7...2...6"
    ".6....28."
    "...419..5"
    "....8..79"
)

#This one fails
sudoku_str = (
    "1.5.9.8.."
    ".9.6.1.5."
    "8..5.2.1."
    ".5.7.3.2."
    "7.3...6.9"
    ".6.9.2.7."
    ".7.1.4..6"
    ".4.2.6.9."
    "..6.8.5.1"
)

sudoku_str3 = (
    "53467891."
    "6721953.8"
    "198342.67"
    "8.9761423"
    "4268.3791"
    "71392485."
    "96.537284"
    "28741963."
    "345286179"
)

sudoku_str4 = (
    "53467891."
    "67219534."
    "1983425.7"
    "859761423"
    "4268.3791"
    "713924856"
    "96153728."
    "287419635"
    "345286179"
)



assert len(sudoku_str) == 81, "Sudoku must be exactly 81 characters"

# ==================================================
# 7. Convert Sudoku → tensor
# ==================================================

def encode_sudoku(grid: torch.Tensor) -> torch.Tensor:
    """
    grid: LongTensor of shape (...), values in {0,1,...,9}
          0 = blank, 1–9 = digits

    returns: LongTensor of same shape, values in {1,...,10}
    """
    return grid + 1

def decode_sudoku(tokens: torch.Tensor) -> torch.Tensor:
    """
    tokens: LongTensor of shape (...), values in {1,...,10}

    returns: LongTensor of same shape, values in {0,1,...,9}
    """
    return tokens - 1


def sudoku_to_tensor(s):
    s = s.replace(".", "0")
    arr = torch.tensor([int(c) for c in s], dtype=torch.long)
    return arr.unsqueeze(0)  # (1, 81)

digits = sudoku_to_tensor(sudoku_str)
inputs = encode_sudoku(digits)
inp_grid = digits[0].reshape(9, 9)


inputs_np = np.load(os.path.join(DATA_DIR, "test", "all__inputs.npy"))

# Model input (exactly what training used)
inputs = torch.tensor(inputs_np[INDEX], dtype=torch.long).unsqueeze(0)  # (1, 81)

# Grid form (only for printing / verification)
# -1 is to decode
inp_grid = (inputs[0] - 1).reshape(9, 9)

# Ground Truth
labels_np = np.load(os.path.join(DATA_DIR, "test", "all__labels.npy"))

# Ground-truth solution grid (0–9)
# -1 is to decode
gt_grid = (torch.tensor(labels_np[INDEX], dtype=torch.long) - 1).reshape(9, 9)

# ==================================================
# 8. Inference
# ==================================================


# --------------------------------------------------
# Build batch (exact structure required by trm.py)
# --------------------------------------------------

batch = {
    "inputs": inputs,
    "labels": inputs.clone(),
    "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
}

#with torch.no_grad():
#    carry = model.initial_carry(batch)
#    carry, outputs = model(
#        carry=carry,
#        batch=batch,
#    )

DEBUG = True

with torch.no_grad():
    carry = model.initial_carry(batch)

    for step in range(model.config.halt_max_steps):
        carry, outputs = model(
            carry=carry,
            batch=batch,
        )
        #This does not work in eval mode as this part is only done in the train mode.
        #so in eval mode this logic will result in max steps. (llm)
        # finished = carry.halted.all()
        #print ("halt? :", finished)

        q_halt_logits = outputs["q_halt_logits"]
        halt_prob = torch.sigmoid(q_halt_logits).item()

        if DEBUG:
            print(f"[step {step}] halt_prob = {halt_prob:.4f}")

            logits = outputs["logits"]
            pred_tokens = logits.argmax(dim=-1)
            pred_digits = decode_sudoku(pred_tokens)
            print(pred_digits[0].view(9, 9))

        #if halt_prob > 0.95:
            #break

        if q_halt_logits.item() > 0:
            print(f"Model confident - halting in {step} (0 indexed)")
            break

        # Will not work in eval mode see previous comment
        #if finished:
           # break



logits = outputs["logits"]        # (1, 81, vocab_size)
pred_tokens = logits.argmax(dim=-1)     # (1, 81)
preds = decode_sudoku(pred_tokens)



pred_grid = preds[0].reshape(9, 9)

# ==================================================
# 9. Pretty printing
# ==================================================

def pretty(grid):
    for i, row in enumerate(grid):
        if i % 3 == 0 and i != 0:
            print("------+-------+------")
        print(
            " ".join(str(x) if x != 0 else "." for x in row[0:3]) + " | " +
            " ".join(str(x) if x != 0 else "." for x in row[3:6]) + " | " +
            " ".join(str(x) if x != 0 else "." for x in row[6:9])
        )

print("\nINPUT:")
pretty(inp_grid.tolist())

print("\nPREDICTION:")
pretty(pred_grid.tolist())


is_valid, msg = verify_solution_with_input(inp_grid, pred_grid)
print("VERIFICATION:", msg)

is_correct = torch.equal(pred_grid, gt_grid)
print("MATCHES GROUND TRUTH:", is_correct)

