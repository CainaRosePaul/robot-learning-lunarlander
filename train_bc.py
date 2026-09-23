from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


DATA_PATH = Path("data/expert_demonstrations.npz")
MODEL_PATH = Path("models/bc_policy.pt")

SEED = 42
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=8, n_actions=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.network(x)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load PPO demonstration dataset
    data = np.load(DATA_PATH)

    observations = data["observations"]
    actions = data["actions"]

    print("Observations shape:", observations.shape)
    print("Actions shape:", actions.shape)

    # Convert NumPy arrays to PyTorch tensors
    x = torch.tensor(
        observations,
        dtype=torch.float32
    )

    y = torch.tensor(
        actions,
        dtype=torch.long
    )

    # Create dataset
    dataset = TensorDataset(x, y)

    # Split into 80% train / 20% validation
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Create BC neural network
    model = BCPolicy(
        obs_dim=observations.shape[1],
        n_actions=4
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):

        # TRAINING
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = loss_function(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * batch_x.size(0)
            )

        average_loss = total_loss / train_size

        # VALIDATION
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for batch_x, batch_y in val_loader:

                logits = model(batch_x)

                predicted_actions = (
                    logits.argmax(dim=1)
                )

                correct += (
                    predicted_actions == batch_y
                ).sum().item()

                total += batch_y.size(0)

        validation_accuracy = correct / total

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy:.3f}"
        )

        # Save best model
        if validation_accuracy > best_val_accuracy:

            best_val_accuracy = validation_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "obs_dim": observations.shape[1],
                    "n_actions": 4,
                    "validation_accuracy": validation_accuracy,
                },
                MODEL_PATH
            )

    print()
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.3f}"
    )

    print(
        f"Saved model to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()