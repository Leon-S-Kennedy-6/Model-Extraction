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


def load_unlabeled_feature_dataset(csv_path):
    df = pd.read_csv(csv_path)
    feature_columns = get_feature_columns(df)
    x = torch.tensor(df[feature_columns].values, dtype=torch.float32)
    if {"label_minus", "label_plus"}.issubset(df.columns):
        y_minus = torch.tensor(df["label_minus"].values, dtype=torch.long)
        y_plus = torch.tensor(df["label_plus"].values, dtype=torch.long)
        return TensorDataset(x, y_minus, y_plus)

    return TensorDataset(x)


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


def train_dual_enhanced_substitute_model(substitute_config, exp_config, oracle):
    device = torch.device(exp_config["experiment"].get("device", "cpu"))
    seed = exp_config["experiment"].get("seed", 42)

    query_set_dir = Path(exp_config["paths"]["query_set_dir"])
    ordinary_samples_path = query_set_dir / "ordinary_query_samples.csv"
    boundary_side_path = query_set_dir / "boundary_side_samples.csv"
    multiscale_boundary_path = query_set_dir / "multiscale_boundary_samples.csv"
    dual_neighborhood_path = query_set_dir / "dual_neighborhood_samples.csv"
    dual_candidates_path = query_set_dir / "dual_point_candidates.csv"

    required_paths = [
        ordinary_samples_path,
        boundary_side_path,
        dual_neighborhood_path,
        dual_candidates_path,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")

    ordinary_dataset, feature_columns = load_query_dataset(ordinary_samples_path)
    boundary_side_dataset, _ = load_query_dataset(boundary_side_path)
    dual_dataset, _ = load_query_dataset(dual_neighborhood_path)
    dual_candidate_dataset = load_unlabeled_feature_dataset(dual_candidates_path)

    if multiscale_boundary_path.exists():
        multiscale_boundary_dataset, _ = load_query_dataset(multiscale_boundary_path)
        boundary_dataset = ConcatDataset([boundary_side_dataset, multiscale_boundary_dataset])
        boundary_source = f"{boundary_side_path};{multiscale_boundary_path}"
    else:
        boundary_dataset = boundary_side_dataset
        boundary_source = str(boundary_side_path)

    dual_config = substitute_config["dual_enhancement"]
    validation_ratio = dual_config.get("validation_ratio", 0.2)
    batch_size = dual_config.get("batch_size", 32)
    epochs = dual_config.get("epochs", 60)
    learning_rate = dual_config.get("learning_rate", 0.0005)
    weight_decay = dual_config.get("weight_decay", 0.0001)

    lambda_ordinary = dual_config.get("lambda_ordinary", 1.0)
    lambda_boundary = dual_config.get("lambda_boundary", 1.0)
    lambda_dual = dual_config.get("lambda_dual", 1.0)
    lambda_dual_zero = dual_config.get("lambda_dual_zero", 0.1)

    ordinary_train, ordinary_val = split_dataset(ordinary_dataset, validation_ratio, seed)
    boundary_train, boundary_val = split_dataset(boundary_dataset, validation_ratio, seed + 1)
    dual_train, dual_val = split_dataset(dual_dataset, validation_ratio, seed + 2)
    dual_candidate_train, dual_candidate_val = split_dataset(
        dual_candidate_dataset,
        validation_ratio,
        seed + 3,
    )

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
    dual_candidate_loader = DataLoader(
        dual_candidate_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dual_candidate_train) > batch_size,
    )

    ordinary_val_loader = DataLoader(ordinary_val, batch_size=batch_size, shuffle=False)
    boundary_val_loader = DataLoader(boundary_val, batch_size=batch_size, shuffle=False)
    dual_val_loader = DataLoader(dual_val, batch_size=batch_size, shuffle=False)
    dual_candidate_val_loader = DataLoader(
        dual_candidate_val,
        batch_size=batch_size,
        shuffle=False,
    )

    checkpoint_dir = Path(exp_config["paths"]["substitute_checkpoint_dir"])
    initial_checkpoint_path = checkpoint_dir / dual_config["initial_checkpoint_name"]
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

    best_score = -1.0
    best_model_path = checkpoint_dir / dual_config["best_model_name"]
    steps_per_epoch = max(
        len(ordinary_loader),
        len(boundary_loader),
        len(dual_loader),
        len(dual_candidate_loader),
    )

    for epoch in range(1, epochs + 1):
        model.train()
        ordinary_iter = iter(ordinary_loader)
        boundary_iter = iter(boundary_loader)
        dual_iter = iter(dual_loader)
        dual_candidate_iter = iter(dual_candidate_loader)

        total_loss = 0.0
        total_ordinary_loss = 0.0
        total_boundary_loss = 0.0
        total_dual_loss = 0.0
        total_dual_zero_loss = 0.0

        for _ in range(steps_per_epoch):
            ordinary_batch, ordinary_iter = next_batch(ordinary_iter, ordinary_loader)
            boundary_batch, boundary_iter = next_batch(boundary_iter, boundary_loader)
            dual_batch, dual_iter = next_batch(dual_iter, dual_loader)
            dual_candidate_batch, dual_candidate_iter = next_batch(
                dual_candidate_iter,
                dual_candidate_loader,
            )

            x_ordinary, y_ordinary = ordinary_batch
            x_boundary, y_boundary = boundary_batch
            x_dual, y_dual = dual_batch
            if len(dual_candidate_batch) == 3:
                x_dual_candidate, y_candidate_minus, y_candidate_plus = (
                    dual_candidate_batch
                )
                y_candidate_minus = y_candidate_minus.to(device)
                y_candidate_plus = y_candidate_plus.to(device)
            else:
                (x_dual_candidate,) = dual_candidate_batch
                y_candidate_minus = None
                y_candidate_plus = None

            x_ordinary = x_ordinary.to(device)
            y_ordinary = y_ordinary.to(device)
            x_boundary = x_boundary.to(device)
            y_boundary = y_boundary.to(device)
            x_dual = x_dual.to(device)
            y_dual = y_dual.to(device)
            x_dual_candidate = x_dual_candidate.to(device)

            optimizer.zero_grad()

            ordinary_loss = criterion(model(x_ordinary), y_ordinary)
            boundary_loss = criterion(model(x_boundary), y_boundary)
            dual_loss = criterion(model(x_dual), y_dual)
            dual_zero_loss = decision_zero_loss(
                model,
                x_dual_candidate,
                y_candidate_minus,
                y_candidate_plus,
            )

            loss = (
                lambda_ordinary * ordinary_loss
                + lambda_boundary * boundary_loss
                + lambda_dual * dual_loss
                + lambda_dual_zero * dual_zero_loss
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ordinary_loss += ordinary_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_dual_loss += dual_loss.item()
            total_dual_zero_loss += dual_zero_loss.item()

        _, ordinary_val_acc = evaluate_with_labels(model, ordinary_val_loader, device)
        _, boundary_val_fidelity = evaluate_with_labels(model, boundary_val_loader, device)
        _, dual_val_fidelity = evaluate_with_labels(model, dual_val_loader, device)
        dual_zero_mse = evaluate_zero_mse(
            model,
            dual_candidate_val_loader,
            device,
        )

        score = (
            0.35 * ordinary_val_acc
            + 0.25 * boundary_val_fidelity
            + 0.40 * dual_val_fidelity
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": substitute_config,
                    "feature_columns": feature_columns,
                    "ordinary_data": str(ordinary_samples_path),
                    "boundary_data": boundary_source,
                    "dual_neighborhood_data": str(dual_neighborhood_path),
                    "dual_candidate_data": str(dual_candidates_path),
                    "ordinary_val_accuracy": ordinary_val_acc,
                    "boundary_val_fidelity": boundary_val_fidelity,
                    "dual_val_fidelity": dual_val_fidelity,
                    "dual_zero_mse": dual_zero_mse,
                },
                best_model_path,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {total_loss / steps_per_epoch:.4f} | "
            f"Ord CE: {total_ordinary_loss / steps_per_epoch:.4f} | "
            f"Boundary CE: {total_boundary_loss / steps_per_epoch:.4f} | "
            f"Dual CE: {total_dual_loss / steps_per_epoch:.4f} | "
            f"Dual Zero: {total_dual_zero_loss / steps_per_epoch:.4f} | "
            f"Ord Val Acc: {ordinary_val_acc:.4f} | "
            f"Boundary Val Fidelity: {boundary_val_fidelity:.4f} | "
            f"Dual Val Fidelity: {dual_val_fidelity:.4f} | "
            f"Dual Zero MSE: {dual_zero_mse:.4f}"
        )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    full_boundary_loader = DataLoader(boundary_dataset, batch_size=batch_size, shuffle=False)
    full_dual_loader = DataLoader(dual_dataset, batch_size=batch_size, shuffle=False)
    full_dual_candidate_loader = DataLoader(
        dual_candidate_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_metrics = evaluate_on_test_set(
        model=model,
        test_csv_path=Path(exp_config["paths"]["processed_data_dir"]) / "test.csv",
        oracle=oracle,
        device=device,
    )
    _, boundary_fidelity = evaluate_with_labels(model, full_boundary_loader, device)
    _, dual_region_fidelity = evaluate_with_labels(model, full_dual_loader, device)
    dual_zero_mse = evaluate_zero_mse(model, full_dual_candidate_loader, device)

    result = {
        "ordinary_samples": len(ordinary_dataset),
        "boundary_samples": len(boundary_dataset),
        "dual_neighborhood_samples": len(dual_dataset),
        "dual_candidates": len(dual_candidate_dataset),
        "best_score": best_score,
        "boundary_fidelity": boundary_fidelity,
        "dual_region_fidelity": dual_region_fidelity,
        "dual_zero_mse": dual_zero_mse,
        "best_model_path": str(best_model_path),
        **test_metrics,
    }

    print("Dual-point enhanced substitute model training finished.")
    print(result)

    return result
