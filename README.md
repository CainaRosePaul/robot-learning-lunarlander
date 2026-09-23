# Learning to Land: PPO and Behavioral Cloning

Train a reinforcement learning agent in Gymnasium's `LunarLander-v3`, collect its demonstrations, and teach a second neural network to imitate its actions. Compare both agents in the simulator and watch them land.

The project explores whether high action imitation accuracy translates into useful control behavior.

## Approach

1. **Train PPO:** Stable-Baselines3 learns a policy through interaction with LunarLander.
2. **Continue training:** Load the initial policy and train for another 300,000 steps.
3. **Collect demonstrations:** Record observations and PPO actions over 100 episodes.
4. **Train behavioral cloning (BC):** Learn to predict the demonstrated actions using supervised learning.
5. **Evaluate:** Compare episode rewards and terminal outcomes on matching seeds.

The environment has eight observation values and four discrete actions. The BC network has two hidden layers of 128 units with ReLU activations. It is trained with Adam and cross-entropy loss for 30 epochs, using an 80% training / 20% validation split.

## Setup on Windows

The project was run with Python 3.11 on CPU. In a PowerShell terminal:

```powershell
git clone https://github.com/CainaRosePaul/robot-learning-lunarlander.git
cd robot-learning-lunarlander
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, use **Python: Select Interpreter** from the command palette and choose `.venv\Scripts\python.exe`.

Generated models and datasets are excluded from Git. Run the following pipeline to create them locally. Run all commands from the project root.

## Run the pipeline

Run each command after the previous one finishes:

```powershell
# Train the initial PPO policy (120,000 requested steps)
.\.venv\Scripts\python.exe train_ppo.py

# Continue from the initial policy for 300,000 additional steps
.\.venv\Scripts\python.exe continue_train.py

# Collect 100 episodes from the continued PPO policy
.\.venv\Scripts\python.exe collect_demonstrations.py

# Train the imitation policy and save its best validation checkpoint
.\.venv\Scripts\python.exe train_bc.py

# Compare rewards over 50 matching episode seeds
.\.venv\Scripts\python.exe evaluate_agents.py

# Compare rewards and terminal outcomes over 100 matching episode seeds
.\.venv\Scripts\python.exe evaluate_bc.py

# Open a window showing PPO and BC on three matching seeds
.\.venv\Scripts\python.exe watch_agents.py
```

The visual demo alternates PPO and BC. The window title identifies the agent and current reward. Press **Esc** in the simulation window to exit. It closes automatically after six episodes.

`continue_train.py` always loads the original `models/ppo_lunarlander.zip`; rerunning it does not resume the last continued checkpoint. Training uses fixed step budgets, not automatic performance-based stopping. PPO completes rollout batches, so actual steps can slightly exceed the requested budget. Rerunning scripts can overwrite their generated outputs.

## Files and outputs

| Script | Main output |
|---|---|
| `train_ppo.py` | `models/ppo_lunarlander.zip` |
| `continue_train.py` | `models/continued/ppo_lunarlander.zip` and periodic checkpoints |
| `collect_demonstrations.py` | `data/expert_demonstrations.npz` |
| `train_bc.py` | `models/bc_policy.pt` |
| `evaluate_agents.py` | Console comparison over 50 episodes |
| `evaluate_bc.py` | `data/bc_evaluation.json`, including per-episode results |
| `watch_agents.py` | Interactive simulation window |

The demonstration dataset contains observations, actions, and episode rewards. All episodes are included, including poor outcomes; the PPO teacher is imperfect.

## Observed results

These are measurements from one local run, not guarantees for future training runs.

- Collected **40,364 state-action pairs** from 100 demonstration episodes.
- Best BC validation action accuracy: **95.8%**, at epoch 13.
- The saved BC checkpoint is the one with the highest validation accuracy, not necessarily the final epoch.

### Reward comparison: 50 episodes

`evaluate_agents.py` uses seeds 1000–1049 for both policies.

| Agent | Mean episode reward | Standard deviation |
|---|---:|---:|
| PPO | 186.40 | 87.83 |
| BC | 184.66 | 86.53 |

### Outcome comparison: 100 episodes

`evaluate_bc.py` uses seeds 1000–1099, outside the demonstration seeds 123–222.

| Agent | Mean reward | Reward std | Environment landing outcomes | Crashes | Timeouts |
|---|---:|---:|---:|---:|---:|
| PPO | 169.02 | 99.00 | 78 | 20 | 2 |
| BC | 176.50 | 91.52 | 80 | 17 | 3 |

Neither agent went out of bounds in this evaluation. The 50-episode comparison is a subset of the 100-episode seed set, not an independent replication.

## Interpreting the results

**Mean episode reward** is the average total reward per episode. **Standard deviation** measures variation between episode scores; it is not a confidence interval or a minimum/maximum range. Reward is not percentage accuracy.

**Validation action accuracy** measures how often BC selects the teacher's recorded action on held-out examples. It is not a landing success rate. Small imitation errors can change the states the agent encounters when controlling the simulator.

**Landing outcomes** in `evaluate_bc.py` count the environment's terminal `+100` resting condition. They do not explicitly verify that the lander is between the flags. A precise landing-pad success metric would additionally check its final position.

BC performed similarly to PPO in these evaluations. The small differences do not establish that one policy is better.

## Limitations and reproducibility

- Validation samples are split randomly by individual state-action pair, not by episode. Correlated states from the same trajectory may appear in both subsets, making validation accuracy optimistic for new trajectories.
- The validation set selects the BC checkpoint, so its accuracy is not an independent test metric. Simulator evaluation provides a separate behavioral check.
- Results come from one training run. Multiple training seeds and larger evaluation sets are needed for stronger comparisons.
- Dependency versions in `requirements.txt` specify minimum versions rather than exact pins. Different package versions and hardware can change results.
- Evaluation uses deterministic action selection; training and evaluation rewards need not match.
