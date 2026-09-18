from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.models.substitute.substitute_mlp import build_substitute_mlp


def get_feature_columns(df):
    feature_columns = [column for column in df.columns if column.startswith("feature_")]
    if feature_columns:
        return sorted(feature_columns, key=lambda column: int(column.split("_")[1]))

    metadata_columns = {
        "label",
        "source",
        "query_index",
        "boundary_id",
        "side",
        "epsilon",
    }
    return [column for column in df.columns if column not in metadata_columns]


def load_query_dataset(csv_path):
    df = pd.read_csv(csv_path)
    feature_columns = get_feature_columns(df)

    x = df[feature_columns].values
    y = df["label"].values

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return TensorDataset(x_tensor, y_tensor), feature_columns


def evaluate_with_labels(model, data_loader, device):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            pred = torch.argmax(logits, dim=1)

            total_loss += loss.item() * x.size(0)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def evaluate_on_test_set(model, test_csv_path, oracle, device):
    test_df = pd.read_csv(test_csv_path)
    feature_columns = get_feature_columns(test_df)

    x = torch.tensor(test_df[feature_columns].values, dtype=torch.float32)
    y_true = torch.tensor(test_df["label"].values, dtype=torch.long)
    y_oracle = oracle.query(x).long()

    model.eval()
    with torch.no_grad():
        logits = model(x.to(device)).cpu()
        pred = torch.argmax(logits, dim=1)

    accuracy = (pred == y_true).float().mean().item()
    fidelity = (pred == y_oracle).float().mean().item()

    return {
        "test_accuracy": accuracy,
        "test_fidelity": fidelity,
        "evaluation_queries": len(test_df),
    }


def train_initial_substitute_model(substitute_config, exp_config, oracle):
    device = torch.device(exp_config["experiment"].get("device", "cpu"))
    seed = exp_config["experiment"].get("seed", 42)

    query_set_dir = Path(exp_config["paths"]["query_set_dir"])
    ordinary_samples_path = query_set_dir / "ordinary_query_samples.csv"
    if not ordinary_samples_path.exists():
        raise FileNotFoundError(f"Ordinary query samples not found: {ordinary_samples_path}")

    dataset, feature_columns = load_query_dataset(ordinary_samples_path)

    training_config = substitute_config["training"]
    batch_size = training_config.get("batch_size", 32)
    epochs = training_config.get("epochs", 80)
    learning_rate = training_config.get("learning_rate", 0.001)
    weight_decay = training_config.get("weight_decay", 0.0001)
    validation_ratio = training_config.get("validation_ratio", 0.2)

    val_size = max(1, int(len(dataset) * validation_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=train_size > batch_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_substitute_mlp(substitute_config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    checkpoint_dir = Path(exp_config["paths"]["substitute_checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / substitute_config["save"]["best_model_name"]

    best_val_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += y.size(0)

        train_loss = total_loss / total
        val_loss, val_acc = evaluate_with_labels(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": substitute_config,
                    "feature_columns": feature_columns,
                    "val_accuracy": val_acc,
                    "training_data": str(ordinary_samples_path),
                },
                best_model_path,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_on_test_set(
        model=model,
        test_csv_path=Path(exp_config["paths"]["processed_data_dir"]) / "test.csv",
        oracle=oracle,
        device=device,
    )

    result = {
        "training_samples": train_size,
        "validation_samples": val_size,
        "best_val_accuracy": best_val_acc,
        "best_model_path": str(best_model_path),
        **test_metrics,
    }

    print("Initial substitute model training finished.")
    print(result)

    return result
