# Data Sampling Structures Setup Flow

## Overview

You're correct: **augmentation is pre-computed** in command 1, then command 2 sets up sampling structures to select from those pre-computed augmentations.

---

## Command 1: Data Preparation (One-time)

Creates these files in `data/sudoku-extreme-1k-aug-1000/train/`:
```
all__inputs.npy         # Shape: (1,001,000, 81) - all augmented puzzles
all__labels.npy         # Shape: (1,001,000, 81) - all solutions
all__puzzle_identifiers.npy  # Shape: (1,001,000,) - puzzle IDs
all__puzzle_indices.npy      # Shape: (1,001,001,) - [0,1,2,...,1001000]
all__group_indices.npy       # Shape: (1,001,) - [0,1001,2002,...,1001000]
dataset.json            # Metadata
```

**Structure established**:
- 1,000 groups (base puzzles)
- 1,001 examples per group (original + 1000 augs)
- Total: 1,001,000 pre-augmented examples stored on disk

---

## Command 2: Training Setup - Data Structures

### Call Stack for Data Structure Setup

```
launch() [pretrain.py:388]
│
├─ load_synced_config() [pretrain.py:416]
│  └─ Creates: PretrainConfig with epochs=50000, eval_interval=5000, batch_size=768
│
├─ train_epochs_per_iter = 5000 [pretrain.py:422]
├─ total_iters = 10 [pretrain.py:423]
│
└─ create_dataloader("train", epochs_per_iter=5000, global_batch_size=768) [pretrain.py:427]
   │
   └─ PuzzleDataset(config, split="train") [pretrain.py:87]
      │
      ├─ __init__() [puzzle_dataset.py:33]
      │  ├─ self.config = PuzzleDatasetConfig(epochs_per_iter=5000, global_batch_size=768)
      │  ├─ self.local_batch_size = 768 / 1 = 768 [puzzle_dataset.py:64]
      │  └─ self._data = None (lazy loading)
      │
      └─ Returns: DataLoader wrapping PuzzleDataset
```

**Key structures created**:
- `PuzzleDatasetConfig`: Contains sampling parameters
- `PuzzleDataset`: Iterator that will lazily load data and setup sampling

---

## Training Loop: Data Structure Initialization (Lazy)

### First Iteration of Training Loop

```
for set_name, batch, global_batch_size in train_loader: [pretrain.py:461]
│
└─ train_loader.__iter__() [puzzle_dataset.py:170]
   │
   └─ _lazy_load_dataset() [puzzle_dataset.py:71] ← FIRST TIME ONLY
      │
      ├─ self._data = {} [puzzle_dataset.py:84]
      ├─ for set_name in ["all"]: [puzzle_dataset.py:85]
      │  └─ self._data["all"] = {
      │        "inputs": np.load("all__inputs.npy", mmap_mode="r"),      # Memory-mapped
      │        "labels": np.load("all__labels.npy", mmap_mode="r"),      # Memory-mapped  
      │        "puzzle_identifiers": np.load("all__puzzle_identifiers.npy"),  # In memory
      │        "puzzle_indices": np.load("all__puzzle_indices.npy"),     # In memory
      │        "group_indices": np.load("all__group_indices.npy")        # In memory
      │     }
      │
      └─ _iter_train() [puzzle_dataset.py:132] ← MAIN SAMPLING STRUCTURE SETUP
         │
         ├─ self._iters += 1  (epoch counter, starts at 1)
         │
         ├─ rng = np.random.Generator(seed=0+1) [puzzle_dataset.py:136]
         │
         ├─ group_order = np.concatenate([
         │     rng.permutation(1000) for _ in range(5000)
         │  ]) [puzzle_dataset.py:138-140]
         │  # Creates: array of 5,000,000 group IDs (shuffled)
         │  # Example: [423, 17, 891, 234, ..., 555, 123] (5M entries)
         │
         ├─ start_index = 0 [puzzle_dataset.py:141]
         │
         └─ while start_index < 5,000,000: [puzzle_dataset.py:143]
            │
            └─ _sample_batch(rng, group_order, ..., start_index, batch_size=768) [puzzle_dataset.py:144]
               │
               ├─ batch = []
               ├─ for i in range(768): [puzzle_dataset.py:21-30]
               │  ├─ group_id = group_order[start_index]  # e.g., 423
               │  ├─ puzzle_id = rng.integers(
               │  │     group_indices[423],      # e.g., 423*1001 = 423423
               │  │     group_indices[423+1]     # e.g., 424*1001 = 424424
               │  │  )  # Random int in [423423, 424424) - picks 1 of 1001 puzzles
               │  ├─ batch.append(puzzle_id)
               │  └─ start_index += 1
               │
               └─ Returns: (new_start_index, batch_indices, batch_puzzle_indices)
```

---

## Key Data Structures Established

### 1. **On Disk (from Command 1)**
```python
# Pre-computed and stored
all__inputs.npy:         [1,001,000, 81] int8 array
all__group_indices.npy:  [1,001] int32 array = [0, 1001, 2002, ..., 1001000]
```

### 2. **In Memory (Command 2 - Lazy Loaded)**
```python
self._data["all"] = {
    "inputs": mmap([1,001,000, 81]),        # Memory-mapped, not fully loaded
    "labels": mmap([1,001,000, 81]),        # Memory-mapped
    "puzzle_identifiers": array([1,001,000]),  # Loaded in RAM (small)
    "puzzle_indices": array([1,001,001]),      # Loaded in RAM (small)
    "group_indices": array([1,001])            # Loaded in RAM (small)
}
```

### 3. **Per Iteration Sampling Structure**
```python
# Created fresh each iteration (every 5000 epochs)
group_order = array([5,000,000]) int32
# Example content: [342, 891, 23, 667, 234, ..., 445, 12, 998]
# This is 5000 shuffled permutations of [0,1,2,...,999] concatenated
```

**Purpose**: Defines which groups to sample from and in what order for the next 5000 epochs.

### 4. **Per Batch Sampling**
```python
# For each of 6510 batches:
start_index: int  # Current position in group_order (0 to 5,000,000)

# Sample 768 examples:
for i in range(768):
    group_id = group_order[start_index]          # e.g., 342
    puzzle_id = random(342*1001, 343*1001)      # Random from 1001 options
    example = inputs[puzzle_id]                  # Fetch via memory-map
    start_index += 1
```

---

## Timeline: When Structures Are Created

```
Time 0: Command 1 completes
  ✓ all__*.npy files on disk

Time 1: launch() starts
  ✓ PretrainConfig created (epochs=50000, batch_size=768)
  ✓ train_epochs_per_iter = 5000
  ✓ total_iters = 10

Time 2: create_dataloader()
  ✓ PuzzleDataset object created (but _data=None)
  ✓ DataLoader wrapper created

Time 3: First batch request (train_loader iteration starts)
  ✓ _lazy_load_dataset() → loads .npy files into self._data
  ✓ group_order created (5M shuffled group IDs)
  ✓ start_index = 0
  ✓ First batch sampled

Time 4: Batches 2-6510
  ✓ start_index advances by 768 each batch
  ✓ Random puzzle_id sampled from each selected group

Time 5: Iteration 2 starts (after 6510 batches)
  ✓ NEW group_order created (different shuffle, different seed)
  ✓ start_index reset to 0
  ✓ Next 6510 batches...

...repeats 10 times...
```

---

## Pseudo-Code Summary

```python
# === SETUP PHASE ===
config = load_config(epochs=50000, batch_size=768, eval_interval=5000)
dataset = PuzzleDataset(config, epochs_per_iter=5000)  # _data=None (lazy)

# === TRAINING LOOP ===
for iteration in range(10):  # 10 iterations of 5000 epochs each
    
    # === LAZY INIT (first time only) ===
    if dataset._data is None:
        dataset._data["all"] = {
            "inputs": memory_map("all__inputs.npy"),          # 1,001,000 examples
            "group_indices": load("all__group_indices.npy")   # [0,1001,2002,...,1001000]
        }
    
    # === EPOCH SHUFFLING ===
    seed = iteration + 1
    rng = RandomGenerator(seed)
    
    # Create sampling plan for 5000 epochs
    group_order = []
    for epoch in range(5000):
        shuffled = rng.permutation([0,1,2,...,999])  # Shuffle 1000 groups
        group_order.append(shuffled)
    group_order = flatten(group_order)  # [5,000,000] array
    
    # === BATCH SAMPLING ===
    start_idx = 0
    for batch_num in range(6510):  # 6510 batches per iteration
        
        batch = []
        for i in range(768):  # 768 examples per batch
            group_id = group_order[start_idx]
            
            # Randomly pick 1 of 1001 puzzles from this group
            puzzle_id = rng.randint(
                group_indices[group_id],      # Start of group
                group_indices[group_id + 1]   # End of group
            )
            
            batch.append(dataset._data["inputs"][puzzle_id])
            start_idx += 1
        
        # Train on batch (forward + backward)
        train_batch(batch)
```

---

## Key Insights

1. **Augmentation is pre-computed**: All 1,001,000 examples exist on disk before training starts

2. **group_order is the sampling schedule**: Created once per iteration (5000 epochs), defines which groups to visit

3. **Random puzzle selection happens per-batch**: Each group contributes 1 random puzzle from its 1,001 options

4. **Memory efficiency**: Only small indices loaded in RAM; actual data is memory-mapped

5. **Epoch semantics**: "1 epoch" = visit all 1000 groups once, but which specific example from each group is random

# Prompt used
I have better idea on the setup now, so don't give line by line but important line for the question I ask.

I issued this command to create the dataset.

```
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments
```

I issued the below command to train the model

```
run_name=”pretrain_mlp_t_sudoku”
python pretrain.py \
arch=trm \
data_paths=”[data/sudoku-extreme-1k-aug-1000]” \
evaluators=”[]” \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True


```

I want to understand what structures are established and when for picking the sample data. I assume after the first command, the augmentation is done and put in some file. Some place in the second command, the number of examples created by first command and the epoch, batch size, iterations all are used to setup some data strcutures. Give me the pseudo call stack too so that I can know the flow.