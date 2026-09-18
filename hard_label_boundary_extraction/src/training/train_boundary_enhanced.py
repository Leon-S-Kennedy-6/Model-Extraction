from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split

from src.models.substitute.substitute_mlp import build_substitute_mlp
from src.training.train_substitute import (
    evaluate_on_test_set,
    evaluate_with_labels,
    get_feature_columns,
    load_query_dataset,
)


def load_decision_point_dataset(csv_path):
    df = pd.read_csv(csv_path)
    feature_columns = get_feature_columns(df)
    x = torch.tensor(df[feature_columns].values, dtype=torch.float32)
    if {"label_minus", "label_plus"}.issubset(df.columns):
        y_minus = torch.tensor(df["label_minus"].values, dtype=torch.long)
        y_plus = torch.tensor(df["label_plus"].values, dtype=torch.long)
        return TensorDataset(x, y_minus, y_plus), feature_columns

    return TensorDataset(x), feature_columns


def load_multiscale_boundary_dataset(csv_path, epsilons):
    df = pd.read_csv(csv_path)
    feature_columns = get_feature_columns(df)
    direction_columns = [
        column for column in df.columns if column.startswith("direction_")
    ]
    direction_columns = sorted(
        direction_columns,
        key=lambda column: int(column.split("_")[1]),
    )

    features = []
    labels = []
    for _, row in df.iterrows():
        x = row[feature_columns].values.astype("float32")
        direction = row[direction_columns].values.astype("float32")
        label_minus = int(row["label_minus"])
        label_plus = int(row["label_plus"])

        for epsilon in epsilons:
            features.append(x - float(epsilon) * direction)
            labels.append(label_minus)
            features.append(x + float(epsilon) * direction)
            labels.append(label_plus)

    x_tensor = torch.tensor(np.asarray(features), dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(labels), dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor)


def load_saved_multiscale_boundary_dataset(csv_path):
    dataset, _ = load_query_dataset(csv_path)
    return dataset


def load_boundary_pair_dataset(csv_path, epsilons):
    df = pd.read_csv(csv_path)
    feature_columns = get_feature_columns(df)
    direction_columns = [
        column for column in df.columns if column.startswith("direction_")
    ]
    direction_columns = sorted(
        direction_columns,
        key=lambda column: int(column.split("_")[1]),
    )

    x_minus_values = []
    y_minus_values = []
    x_plus_values = []
    y_plus_values = []

    for _, row in df.iterrows():
        x = row[feature_columns].values.astype("float32")
        direction = row[direction_columns].values.astype("float32")
        label_minus = int(row["label_minus"])
        label_plus = int(row["label_plus"])

        for epsilon in epsilons:
            x_minus_values.append(x - float(epsilon) * direction)
            y_minus_values.append(label_minus)
            x_plus_values.append(x + float(epsilon) * direction)
            y_plus_values.append(label_plus)

    return TensorDataset(
        torch.tensor(np.asarray(x_minus_values), dtype=torch.float32),
        torch.tensor(np.asarray(y_minus_values), dtype=torch.long),
        torch.tensor(np.asarray(x_plus_values), dtype=torch.float32),
        torch.tensor(np.asarray(y_plus_values), dtype=torch.long),
    )


def load_boundary_pair_dataset_from_samples(csv_path):
    df = pd.read_csv(csv_path)
    feature_columns = get_feature_columns(df)

    x_minus_values = []
    y_minus_values = []
    x_plus_values = []
    y_plus_values = []

    grouped = df.groupby(["boundary_id", "epsilon"])
    for _, group in grouped:
        minus_rows = group[group["side"] == "minus"]
        plus_rows = group[group["side"] == "plus"]
        if minus_rows.empty or plus_rows.empty:
            continue

        minus_row = minus_rows.iloc[0]
        plus_row = plus_rows.iloc[0]

        if int(minus_row["label"]) == int(plus_row["label"]):
            continue

        x_minus_values.append(minus_row[feature_columns].values.astype("float32"))
        y_minus_values.append(int(minus_row["label"]))
        x_plus_values.append(plus_row[feature_columns].values.astype("float32"))
        y_plus_values.append(int(plus_row["label"]))

    if not x_minus_values:
        raise ValueError("No crossing boundary pairs found in multi-scale boundary samples.")

    return TensorDataset(
        torch.tensor(np.asarray(x_minus_values), dtype=torch.float32),
        torch.tensor(np.asarray(y_minus_values), dtype=torch.long),
        torch.tensor(np.asarray(x_plus_values), dtype=torch.float32),
        torch.tensor(np.asarray(y_plus_values), dtype=torch.long),
    )


def next_batch(iterator, data_loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(data_loader)
        batch = next(iterator)
    return batch, iterator


def split_dataset(dataset, validation_ratio, seed):
    val_size = max(1, int(len(dataset) * validation_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def gather_class_logit(logits, labels):
    return logits.gather(1, labels.view(-1, 1)).squeeze(1)


def pairwise_margin(logits, y_minus, y_plus):
    return gather_class_logit(logits, y_plus) - gather_class_logit(logits, y_minus)


def decision_zero_loss(model, x, y_minus=None, y_plus=None):
    logits = model(x)
    if y_minus is None or y_plus is None:
        boundary_score = logits[:, 1] - logits[:, 0]
    else:
        boundary_score = pairwise_margin(logits, y_minus, y_plus)
    return torch.mean(boundary_score.pow(2))


def boundary_cross_loss(model, x_minus, y_minus, x_plus, y_plus):
    logits_minus = model(x_minus)
    logits_plus = model(x_plus)

    margin_minus = pairwise_margin(logits_minus, y_minus, y_plus)
    margin_plus = pairwise_margin(logits_plus, y_minus, y_plus)

    minus_side_loss = F.softplus(margin_minus)
    plus_side_loss = F.softplus(-margin_plus)
    return 0.5 * (minus_side_loss.mean() + plus_side_loss.mean())


def evaluate_decision_zero(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                x, y_minus, y_plus = batch
                y_minus = y_minus.to(device)
                y_plus = y_plus.to(device)
            else:
                (x,) = batch
                y_minus = None
                y_plus = None

            x = x.to(device)
            loss = decision_zero_loss(model, x, y_minus, y_plus)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

    return total_loss / total


def evaluate_boundary_fidelity(model, data_loader, device):
    _, boundary_accuracy = evaluate_with_labels(model, data_loader, device)
    return boundary_accuracy


def train_boundary_enhanced_substitute_model(
    substitute_config,
    attack_config,
    exp_config,
    oracle,
):
    device = torch.device(exp_config["experiment"].get("device", "cpu"))
    seed = exp_config["experiment"].get("seed", 42)

    query_set_dir = Path(exp_config["paths"]["query_set_dir"])
    ordinary_samples_path = query_set_dir / "ordinary_query_samples.csv"
    boundary_side_path = query_set_dir / "boundary_side_samples.csv"
    decision_points_path = query_set_dir / "decision_points.csv"
    multiscale_boundary_path = query_set_dir / "multiscale_boundary_samples.csv"

    for path in [ordinary_samples_path, boundary_side_path, decision_points_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required query set not found: {path}")

    ordinary_dataset, feature_columns = load_query_dataset(ordinary_samples_path)
    original_boundary_dataset, _ = load_query_dataset(boundary_side_path)
    decision_dataset, _ = load_decision_point_dataset(decision_points_path)
    epsilons = attack_config.get("boundary_sampling", {}).get("epsilons", [0.001])
    if multiscale_boundary_path.exists():
        multiscale_boundary_dataset = load_saved_multiscale_boundary_dataset(
            multiscale_boundary_path,
        )
        boundary_pair_dataset = load_boundary_pair_dataset_from_samples(
            multiscale_boundary_path,
        )
        multiscale_source = str(multiscale_boundary_path)
    else:
        multiscale_boundary_dataset = load_multiscale_boundary_dataset(
            decision_points_path,
            epsilons,
        )
        boundary_pair_dataset = load_boundary_pair_dataset(decision_points_path, epsilons)
        multiscale_source = "generated_from_decision_points_without_requery"

    boundary_dataset = ConcatDataset([original_boundary_dataset, multiscale_boundary_dataset])

    enhance_config = substitute_config["boundary_enhancement"]
    validation_ratio = enhance_config.get("validation_ratio", 0.2)
    batch_size = enhance_config.get("batch_size", 32)
    epochs = enhance_config.get("epochs", 60)
    learning_rate = enhance_config.get("learning_rate", 0.0005)
    weight_decay = enhance_config.get("weight_decay", 0.0001)

    ordinary_train, ordinary_val = split_dataset(ordinary_dataset, validation_ratio, seed)
    boundary_train, boundary_val = split_dataset(boundary_dataset, validation_ratio, seed + 1)
    decision_train, decision_val = split_dataset(decision_dataset, validation_ratio, seed + 2)
    pair_train, _ = split_dataset(boundary_pair_dataset, validation_ratio, seed + 3)

    ordinary_loader = DataLoader(
        ordinary_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(ordinary_train) > batch_size,
    )
    boundary_loader = DataLoader(
        boundary_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(boundary_train) > batch_size,
    )
    decision_loader = DataLoader(
        decision_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(decision_train) > batch_size,
    )
    pair_loader = DataLoader(
        pair_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(pair_train) > batch_size,
    )

    ordinary_val_loader = DataLoader(ordinary_val, batch_size=batch_size, shuffle=False)
    boundary_val_loader = DataLoader(boundary_val, batch_size=batch_size, shuffle=False)
    decision_val_loader = DataLoader(decision_val, batch_size=batch_size, shuffle=False)

    model = build_substitute_mlp(substitute_config).to(device)

    checkpoint_dir = Path(exp_config["paths"]["substitute_checkpoint_dir"])
    initial_checkpoint_path = checkpoint_dir / enhance_config["initial_checkpoint_name"]
    if not initial_checkpoint_path.exists():
        raise FileNotFoundError(f"Initial substitute checkpoint not found: {initial_checkpoint_path}")

    initial_checkpoint = torch.load(initial_checkpoint_path, map_location=device)
    model.load_state_dict(initial_checkpoint["model_state_dict"])

    loss_weights = attack_config.get("loss_weights", {})
    lambda_ordinary = enhance_config.get(
        "lambda_ordinary",
        loss_weights.get("lambda_ce", 1.0),
    )
    lambda_boundary = enhance_config.get(
        "lambda_boundary",
        loss_weights.get("lambda_cross", 1.0),
    )
    lambda_cross = enhance_config.get(
        "lambda_cross",
        loss_weights.get("lambda_cross", 1.0),
    )
    lambda_zero = enhance_config.get(
        "lambda_zero",
        loss_weights.get("lambda_equal", 0.2),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_score = -1.0
    best_model_path = checkpoint_dir / enhance_config["best_model_name"]
    steps_per_epoch = max(
        len(ordinary_loader),
        len(boundary_loader),
        len(decision_loader),
        len(pair_loader),
    )

    for epoch in range(1, epochs + 1):
        model.train()
        ordinary_iter = iter(ordinary_loader)
        boundary_iter = iter(boundary_loader)
        decision_iter = iter(decision_loader)
        pair_iter = iter(pair_loader)

        total_loss = 0.0
        total_ordinary_loss = 0.0
        total_boundary_loss = 0.0
        total_cross_loss = 0.0
        total_zero_loss = 0.0

        for _ in range(steps_per_epoch):
            (ordinary_batch, ordinary_iter) = next_batch(ordinary_iter, ordinary_loader)
            (boundary_batch, boundary_iter) = next_batch(boundary_iter, boundary_loader)
            (decision_batch, decision_iter) = next_batch(decision_iter, decision_loader)
            (pair_batch, pair_iter) = next_batch(pair_iter, pair_loader)

            x_ordinary, y_ordinary = ordinary_batch
            x_boundary, y_boundary = boundary_batch
            if len(decision_batch) == 3:
                x_decision, y_decision_minus, y_decision_plus = decision_batch
                y_decision_minus = y_decision_minus.to(device)
                y_decision_plus = y_decision_plus.to(device)
            else:
                (x_decision,) = decision_batch
                y_decision_minus = None
                y_decision_plus = None
            x_minus, y_minus, x_plus, y_plus = pair_batch

            x_ordinary = x_ordinary.to(device)
            y_ordinary = y_ordinary.to(device)
            x_boundary = x_boundary.to(device)
            y_boundary = y_boundary.to(device)
            x_decision = x_decision.to(device)
            x_minus = x_minus.to(device)
            y_minus = y_minus.to(device)
            x_plus = x_plus.to(device)
            y_plus = y_plus.to(device)

            optimizer.zero_grad()

            ordinary_loss = criterion(model(x_ordinary), y_ordinary)
            boundary_loss = criterion(model(x_boundary), y_boundary)
            cross_loss = boundary_cross_loss(model, x_minus, y_minus, x_plus, y_plus)
            zero_loss = decision_zero_loss(
                model,
                x_decision,
                y_decision_minus,
                y_decision_plus,
            )

            loss = (
                lambda_ordinary * ordinary_loss
                + lambda_boundary * boundary_loss
                + lambda_cross * cross_loss
                + lambda_zero * zero_loss
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ordinary_loss += ordinary_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_cross_loss += cross_loss.item()
            total_zero_loss += zero_loss.item()

        ordinary_val_loss, ordinary_val_acc = evaluate_with_labels(
            model,
            ordinary_val_loader,
            device,
        )
        boundary_val_fidelity = evaluate_boundary_fidelity(
            model,
            boundary_val_loader,
            device,
        )
        val_zero_mse = evaluate_decision_zero(
            model,
            decision_val_loader,
            device,
        )

        score = 0.5 * ordinary_val_acc + 0.5 * boundary_val_fidelity
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": substitute_config,
                    "feature_columns": feature_columns,
                    "ordinary_data": str(ordinary_samples_path),
                    "boundary_side_data": str(boundary_side_path),
                    "decision_point_data": str(decision_points_path),
                    "multiscale_boundary_data": multiscale_source,
                    "ordinary_val_accuracy": ordinary_val_acc,
                    "boundary_val_fidelity": boundary_val_fidelity,
                    "decision_zero_mse": val_zero_mse,
                },
                best_model_path,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {total_loss / steps_per_epoch:.4f} | "
            f"Ord CE: {total_ordinary_loss / steps_per_epoch:.4f} | "
            f"Boundary CE: {total_boundary_loss / steps_per_epoch:.4f} | "
            f"Cross: {total_cross_loss / steps_per_epoch:.4f} | "
            f"Zero: {total_zero_loss / steps_per_epoch:.4f} | "
            f"Ord Val Acc: {ordinary_val_acc:.4f} | "
            f"Boundary Val Fidelity: {boundary_val_fidelity:.4f} | "
            f"Decision Zero MSE: {val_zero_mse:.4f}"
        )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    full_boundary_loader = DataLoader(
        original_boundary_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    full_augmented_boundary_loader = DataLoader(
        boundary_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    full_decision_loader = DataLoader(decision_dataset, batch_size=batch_size, shuffle=False)

    test_metrics = evaluate_on_test_set(
        model=model,
        test_csv_path=Path(exp_config["paths"]["processed_data_dir"]) / "test.csv",
        oracle=oracle,
        device=device,
    )
    boundary_fidelity = evaluate_boundary_fidelity(model, full_boundary_loader, device)
    augmented_boundary_fidelity = evaluate_boundary_fidelity(
        model,
        full_augmented_boundary_loader,
        device,
    )
    decision_zero_mse = evaluate_decision_zero(model, full_decision_loader, device)

    result = {
        "ordinary_samples": len(ordinary_dataset),
        "boundary_side_samples": len(original_boundary_dataset),
        "augmented_boundary_samples": len(boundary_dataset),
        "decision_points": len(decision_dataset),
        "multiscale_boundary_source": multiscale_source,
        "best_score": best_score,
        "boundary_fidelity": boundary_fidelity,
        "augmented_boundary_fidelity": augmented_boundary_fidelity,
        "decision_zero_mse": decision_zero_mse,
        "best_model_path": str(best_model_path),
        **test_metrics,
    }

    print("Boundary-enhanced substitute model training finished.")
    print(result)

    return result
