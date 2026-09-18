from pathlib import Path

import torch

from src.models.target.model_factory import build_target_model


class HardLabelOracle:
    def __init__(self, checkpoint_path, device="cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        model_config = checkpoint["model_config"]
        self.model = build_target_model(model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.query_count = 0

    def query(self, x):
        """
        Return hard labels only.
        x can be a torch.Tensor with shape [num_features] or [batch_size, num_features].
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            labels = torch.argmax(logits, dim=1)

        self.query_count += x.size(0)

        return labels.cpu()

    def query_one(self, x):
        label = self.query(x)
        return int(label.item())

    def reset_query_count(self):
        self.query_count = 0

    def get_query_count(self):
        return self.query_count
