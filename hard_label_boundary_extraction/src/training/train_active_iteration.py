from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split

from src.models.substitute.substitute_mlp import build_substitute_mlp
from src.training.train_boundary_enhanced import decision_zero_loss
from src.training.train_substitute import (
    evaluate_on_test_set,
    evaluate_with_labels,
    get_feature_columns,
    load_query_dataset,
)


def existing_paths(paths):
    return [Path(path) for path in paths if Path(path).exists()]


def load_concat_query_dataset(paths):
    datasets = []
    feature_columns = None
    for path in existing_paths(paths):
        dataset, columns = load_query_dataset(path)
        datasets.append(dataset)
        if feature_columns is None:
            feature_columns = columns

    if not datasets:
        raise FileNotFoundError("No query datasets were found.")

    if len(datasets) == 1:
        return datasets[0], feature_columns
    return ConcatDataset(datasets), feature_columns


def load_concat_unlabeled_dataset(paths):
    existing = existing_paths(paths)
    any_pair_labels = any(
        {"label_minus", "label_plus"}.issubset(pd.read_csv(path, nrows=0).columns)
        for path in existing
    )

    datasets = []
    for path in existing:
        df = pd.read_csv(path)
        feature_columns = get_feature_columns(df)
        x = torch.tensor(df[feature_columns].values, dtype=torch.float32)
        if any_pair_labels and {"label_minus", "label_plus"}.issubset(df.columns):
            y_minus = torch.tensor(df["label_minus"].values, dtype=torch.long)
            y_plus = torch.tensor(df["label_plus"].values, dtype=torch.long)
            datasets.append(TensorDataset(x, y_minus, y_plus))
        elif any_pair_labels:
            y_minus = torch.zeros(len(df), dtype=torch.long)
            y_plus = torch.ones(len(df), dtype=torch.long)
            datasets.append(TensorDataset(x, y_minus, y_plus))
        else:
            datasets.append(TensorDataset(x))

    if not datasets:
        raise FileNotFoundError("No unlabeled feature datasets were found.")

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def split_dataset(dataset, validation_ratio, seed):
    val_size = max(1, int(len(dataset) * validation_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def next_batch(iterator, data_loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(data_loader)
        batch = next(iterator)
    return batch, iterator


def evaluate_zero_mse(model, data_loader, device):
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


def train_active_iteration_model(substitute_config, exp_config, oracle, round_dir):
    device = torch.device(exp_config["experiment"].get("device", "cpu"))
    seed = exp_config["experiment"].get("seed", 42)

    query_set_dir = Path(exp_config["paths"]["query_set_dir"])
    round_dir = Path(round_dir)

    ordinary_paths = [
        query_set_dir / "ordinary_query_samples.csv",
        query_set_dir / "dual_search_ordinary_samples.csv",
        round_dir / "active_ordinary_samples.csv",
        round_dir / "ordinary_query_samples.csv",
        round_dir / "dual_search_ordinary_samples.csv",
    ]
    boundary_paths = [
        query_set_dir / "boundary_side_samples.csv",
        query_set_dir / "multiscale_boundary_samples.csv",
        round_dir / "boundary_side_samples.csv",
        round_dir / "multiscale_boundary_samples.csv",
    ]
    dual_paths = [
        query_set_dir / "dual_neighborhood_samples.csv",
        round_dir / "dual_neighborhood_samples.csv",
    ]
    zero_paths = [
        query_set_dir / "dual_point_candidates.csv",
        round_dir / "dual_point_candidates.csv",
    ]

    ordinary_dataset, feature_columns = load_concat_query_dataset(ordinary_paths)
    boundary_dataset, _ = load_concat_query_dataset(boundary_paths)
    dual_dataset, _ = load_concat_query_dataset(dual_paths)
    zero_dataset = load_concat_unlabeled_dataset(zero_paths)

    active_config = substitute_config["active_iteration"]
    validation_ratio = active_config.get("validation_ratio", 0.2)
    batch_size = active_config.get("batch_size", 64)
    epochs = active_config.get("epochs", 100)
    learning_rate = active_config.get("learning_rate", 0.0005)
    weight_decay = active_config.get("weight_decay", 0.0001)

    ordinary_train, ordinary_val = split_dataset(ordinary_dataset, validation_ratio, seed)
    boundary_train, boundary_val = split_dataset(boundary_dataset, validation_ratio, seed + 1)
    dual_train, dual_val = split_dataset(dual_dataset, validation_ratio, seed + 2)
    zero_train, zero_val = split_dataset(zero_dataset, validation_ratio, seed + 3)

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
    dual_loader = DataLoader(
        dual_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dual_train) > batch_size,
    )
    zero_loader = DataLoader(
        zero_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(zero_train) > batch_size,
    )

    ordinary_val_loader = DataLoader(ordinary_val, batch_size=batch_size, shuffle=False)
    boundary_val_loader = DataLoader(boundary_val, batch_size=batch_size, shuffle=False)
    dual_val_loader = DataLoader(dual_val, batch_size=batch_size, shuffle=False)
    zero_val_loader = DataLoader(zero_val, batch_size=batch_size, shuffle=False)

    checkpoint_dir = Path(exp_config["paths"]["substitute_checkpoint_dir"])
    initial_checkpoint_path = checkpoint_dir / active_config["initial_checkpoint_name"]
    if not initial_checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {initial_checkpoint_path}")

    model = build_substitute_mlp(substitute_config).to(device)
    initial_checkpoint = torch.load(initial_checkpoint_path, map_location=device)
    model.load_state_dict(initial_checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    lambda_ordinary = active_config.get("lambda_ordinary", 1.5)
    lambda_boundary = active_config.get("lambda_boundary", 0.5)
    lambda_dual = active_config.get("lambda_dual", 0.8)
    lambda_zero = active_config.get("lambda_zero", 0.05)

    best_score = -1.0
    best_model_path = checkpoint_dir / active_config["best_model_name"]
    steps_per_epoch = max(
        len(ordinary_loader),
        len(boundary_loader),
        len(dual_loader),
        len(zero_loader),
    )

    for epoch in range(1, epochs + 1):
        model.train()
        ordinary_iter = iter(ordinary_loader)
        boundary_iter = iter(boundary_loader)
        dual_iter = iter(dual_loader)
        zero_iter = iter(zero_loader)

        total_loss = 0.0
        for _ in range(steps_per_epoch):
            ordinary_batch, ordinary_iter = next_batch(ordinary_iter, ordinary_loader)
            boundary_batch, boundary_iter = next_batch(boundary_iter, boundary_loader)
            dual_batch, dual_iter = next_batch(dual_iter, dual_loader)
            zero_batch, zero_iter = next_batch(zero_iter, zero_loader)

            x_ordinary, y_ordinary = ordinary_batch
            x_boundary, y_boundary = boundary_batch
            x_dual, y_dual = dual_batch
            if len(zero_batch) == 3:
                x_zero, y_zero_minus, y_zero_plus = zero_batch
                y_zero_minus = y_zero_minus.to(device)
                y_zero_plus = y_zero_plus.to(device)
            else:
                (x_zero,) = zero_batch
                y_zero_minus = None
                y_zero_plus = None

            x_ordinary = x_ordinary.to(device)
            y_ordinary = y_ordinary.to(device)
            x_boundary = x_boundary.to(device)
            y_boundary = y_boundary.to(device)
            x_dual = x_dual.to(device)
            y_dual = y_dual.to(device)
            x_zero = x_zero.to(device)

            optimizer.zero_grad()
            ordinary_loss = criterion(model(x_ordinary), y_ordinary)
            boundary_loss = criterion(model(x_boundary), y_boundary)
            dual_loss = criterion(model(x_dual), y_dual)
            zero_loss = decision_zero_loss(model, x_zero, y_zero_minus, y_zero_plus)

            loss = (
                lambda_ordinary * ordinary_loss
                + lambda_boundary * boundary_loss
                + lambda_dual * dual_loss
                + lambda_zero * zero_loss
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        _, ordinary_val_acc = evaluate_with_labels(model, ordinary_val_loader, device)
        _, boundary_val_fidelity = evaluate_with_labels(model, boundary_val_loader, device)
        _, dual_val_fidelity = evaluate_with_labels(model, dual_val_loader, device)
        zero_mse = evaluate_zero_mse(model, zero_val_loader, device)

        score = (
            0.55 * ordinary_val_acc
            + 0.15 * boundary_val_fidelity
            + 0.30 * dual_val_fidelity
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": substitute_config,
                    "feature_columns": feature_columns,
                    "ordinary_val_accuracy": ordinary_val_acc,
                    "boundary_val_fidelity": boundary_val_fidelity,
                    "dual_val_fidelity": dual_val_fidelity,
                    "zero_mse": zero_mse,
                    "round_dir": str(round_dir),
                },
                best_model_path,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {total_loss / steps_per_epoch:.4f} | "
            f"Ord Val: {ordinary_val_acc:.4f} | "
            f"Boundary Val: {boundary_val_fidelity:.4f} | "
            f"Dual Val: {dual_val_fidelity:.4f} | "
            f"Zero MSE: {zero_mse:.4f}"
        )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    full_boundary_loader = DataLoader(boundary_dataset, batch_size=batch_size, shuffle=False)
    full_dual_loader = DataLoader(dual_dataset, batch_size=batch_size, shuffle=False)

    test_metrics = evaluate_on_test_set(
        model=model,
        test_csv_path=Path(exp_config["paths"]["processed_data_dir"]) / "test.csv",
        oracle=oracle,
        device=device,
    )
    _, boundary_fidelity = evaluate_with_labels(model, full_boundary_loader, device)
    _, dual_region_fidelity = evaluate_with_labels(model, full_dual_loader, device)

    result = {
        "ordinary_samples": len(ordinary_dataset),
        "boundary_samples": len(boundary_dataset),
        "dual_samples": len(dual_dataset),
        "zero_samples": len(zero_dataset),
        "best_score": best_score,
        "boundary_fidelity": boundary_fidelity,
        "dual_region_fidelity": dual_region_fidelity,
        "best_model_path": str(best_model_path),
        **test_metrics,
    }

    print("Active iteration substitute model training finished.")
    print(result)
    return result
