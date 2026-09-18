from pathlib import Path

import numpy as np
import pandas as pd


def _sorted_columns(df, prefix):
    columns = [column for column in df.columns if column.startswith(prefix)]
    return sorted(columns, key=lambda column: int(column.split("_")[1]))


class MultiScaleBoundarySampler:
    def __init__(self, oracle, epsilons=None):
        self.oracle = oracle
        self.epsilons = epsilons or [0.001, 0.005, 0.01]
        self.records = []
        self.pair_records = []

    def sample_from_decision_points(self, decision_points_path):
        decision_df = pd.read_csv(decision_points_path)
        feature_columns = _sorted_columns(decision_df, "feature_")
        direction_columns = _sorted_columns(decision_df, "direction_")

        query_start = self.oracle.get_query_count()

        for boundary_id, row in decision_df.iterrows():
            x_boundary = row[feature_columns].values.astype("float32")
            direction = row[direction_columns].values.astype("float32")

            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction = direction / norm

            for epsilon in self.epsilons:
                x_minus = x_boundary - float(epsilon) * direction
                x_plus = x_boundary + float(epsilon) * direction

                label_minus = self.oracle.query_one(x_minus)
                label_plus = self.oracle.query_one(x_plus)

                self.records.append(
                    self._sample_record(
                        x=x_minus,
                        label=label_minus,
                        boundary_id=boundary_id,
                        side="minus",
                        epsilon=epsilon,
                    )
                )
                self.records.append(
                    self._sample_record(
                        x=x_plus,
                        label=label_plus,
                        boundary_id=boundary_id,
                        side="plus",
                        epsilon=epsilon,
                    )
                )

                self.pair_records.append(
                    {
                        "boundary_id": int(boundary_id),
                        "epsilon": float(epsilon),
                        "label_minus": int(label_minus),
                        "label_plus": int(label_plus),
                        "is_crossing": int(label_minus != label_plus),
                    }
                )

        query_count = self.oracle.get_query_count() - query_start
        crossing_count = sum(record["is_crossing"] for record in self.pair_records)
        pair_count = len(self.pair_records)

        return {
            "decision_points": len(decision_df),
            "epsilons": ";".join(str(epsilon) for epsilon in self.epsilons),
            "sample_count": len(self.records),
            "pair_count": pair_count,
            "crossing_pair_count": crossing_count,
            "crossing_pair_ratio": crossing_count / pair_count if pair_count else 0.0,
            "query_count": query_count,
        }

    def _sample_record(self, x, label, boundary_id, side, epsilon):
        record = {f"feature_{i}": float(value) for i, value in enumerate(x)}
        record["label"] = int(label)
        record["boundary_id"] = int(boundary_id)
        record["side"] = side
        record["epsilon"] = float(epsilon)
        record["source"] = "multiscale_boundary"
        record["query_index"] = int(self.oracle.get_query_count())
        return record

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(self.records).to_csv(
            output_dir / "multiscale_boundary_samples.csv",
            index=False,
        )
        pd.DataFrame(self.pair_records).to_csv(
            output_dir / "multiscale_boundary_pairs.csv",
            index=False,
        )
