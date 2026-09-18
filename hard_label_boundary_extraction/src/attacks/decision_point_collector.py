from pathlib import Path
import time

import numpy as np
import pandas as pd


class DecisionPointCollector:
    def __init__(
        self,
        oracle,
        epsilon=0.001,
        binary_search_steps=20,
        max_expand_steps=20,
        initial_step=0.05,
        expand_factor=2.0,
        max_radius=8.0,
        random_state=42,
    ):
        self.oracle = oracle
        self.epsilon = epsilon
        self.binary_search_steps = binary_search_steps
        self.max_expand_steps = max_expand_steps
        self.initial_step = initial_step
        self.expand_factor = expand_factor
        self.max_radius = max_radius
        self.rng = np.random.default_rng(random_state)

        self.ordinary_records = []
        self.decision_records = []
        self.boundary_side_records = []
        self.max_query_budget = None

    def _has_budget(self, required_queries=1):
        if self.max_query_budget is None:
            return True
        return self.oracle.get_query_count() + required_queries <= self.max_query_budget

    def _query_one(self, x, source=None, store_ordinary=True):
        if not self._has_budget():
            raise RuntimeError("Query budget exhausted.")

        x = np.asarray(x, dtype=np.float32)
        label = self.oracle.query_one(x)

        if store_ordinary:
            self.ordinary_records.append(
                self._sample_record(
                    x=x,
                    label=label,
                    source=source,
                    query_index=self.oracle.get_query_count(),
                )
            )

        return label

    def _sample_record(self, x, label, source, query_index):
        record = {f"feature_{i}": float(value) for i, value in enumerate(x)}
        record["label"] = int(label)
        record["source"] = source
        record["query_index"] = int(query_index)
        return record

    def _decision_record(
        self,
        x_boundary,
        direction,
        label_minus,
        label_plus,
        source,
        queries_used,
    ):
        record = {
            f"feature_{i}": float(value)
            for i, value in enumerate(x_boundary)
        }
        record.update(
            {
                f"direction_{i}": float(value)
                for i, value in enumerate(direction)
            }
        )
        record["label_minus"] = int(label_minus)
        record["label_plus"] = int(label_plus)
        record["epsilon"] = float(self.epsilon)
        record["source"] = source
        record["queries_used"] = int(queries_used)
        return record

    def _boundary_side_record(self, x, label, boundary_id, side):
        record = {f"feature_{i}": float(value) for i, value in enumerate(x)}
        record["label"] = int(label)
        record["boundary_id"] = int(boundary_id)
        record["side"] = side
        record["epsilon"] = float(self.epsilon)
        return record

    def _normalize_direction(self, direction):
        direction = np.asarray(direction, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None
        return direction / norm

    def _binary_search_between(self, x_low, y_low, x_high, y_high):
        left = np.asarray(x_low, dtype=np.float32)
        right = np.asarray(x_high, dtype=np.float32)
        left_label = y_low
        right_label = y_high

        if left_label == right_label:
            return None

        for _ in range(self.binary_search_steps):
            mid = (left + right) / 2.0
            mid_label = self._query_one(
                mid,
                source="decision_binary_search",
                store_ordinary=False,
            )

            if mid_label == left_label:
                left = mid
                left_label = mid_label
            else:
                right = mid
                right_label = mid_label

        x_boundary = (left + right) / 2.0
        direction = self._normalize_direction(right - left)
        if direction is None:
            return None

        x_minus = x_boundary - self.epsilon * direction
        x_plus = x_boundary + self.epsilon * direction

        label_minus = self._query_one(
            x_minus,
            source="boundary_side_minus",
            store_ordinary=False,
        )
        label_plus = self._query_one(
            x_plus,
            source="boundary_side_plus",
            store_ordinary=False,
        )

        if label_minus == label_plus:
            return None

        return {
            "x_boundary": x_boundary,
            "direction": direction,
            "x_minus": x_minus,
            "label_minus": label_minus,
            "x_plus": x_plus,
            "label_plus": label_plus,
        }

    def search_along_direction(self, start, direction):
        if not self._has_budget(required_queries=self.binary_search_steps + 4):
            return None

        start = np.asarray(start, dtype=np.float32)
        direction = self._normalize_direction(direction)
        if direction is None:
            return None

        query_start = self.oracle.get_query_count()
        y_start = self._query_one(
            start,
            source="ray_start",
            store_ordinary=True,
        )

        step = self.initial_step
        x_low = start
        y_low = y_start

        for _ in range(self.max_expand_steps):
            if not self._has_budget(required_queries=self.binary_search_steps + 3):
                return None

            if step > self.max_radius:
                break

            x_high = start + step * direction
            y_high = self._query_one(
                x_high,
                source="ray_expand",
                store_ordinary=True,
            )

            if y_high != y_low:
                result = self._binary_search_between(
                    x_low=x_low,
                    y_low=y_low,
                    x_high=x_high,
                    y_high=y_high,
                )
                if result is None:
                    return None

                result["source"] = "random_ray"
                result["queries_used"] = self.oracle.get_query_count() - query_start
                return result

            x_low = x_high
            y_low = y_high
            step *= self.expand_factor

        return None

    def search_between_pair(self, x_a, y_a, x_b, y_b):
        if y_a == y_b:
            return None

        if not self._has_budget(required_queries=self.binary_search_steps + 2):
            return None

        query_start = self.oracle.get_query_count()
        result = self._binary_search_between(
            x_low=x_a,
            y_low=y_a,
            x_high=x_b,
            y_high=y_b,
        )
        if result is None:
            return None

        result["source"] = "opposite_label_pair"
        result["queries_used"] = self.oracle.get_query_count() - query_start
        return result

    def _add_decision_result(self, result):
        boundary_id = len(self.decision_records)

        self.decision_records.append(
            self._decision_record(
                x_boundary=result["x_boundary"],
                direction=result["direction"],
                label_minus=result["label_minus"],
                label_plus=result["label_plus"],
                source=result["source"],
                queries_used=result["queries_used"],
            )
        )

        self.boundary_side_records.append(
            self._boundary_side_record(
                x=result["x_minus"],
                label=result["label_minus"],
                boundary_id=boundary_id,
                side="minus",
            )
        )
        self.boundary_side_records.append(
            self._boundary_side_record(
                x=result["x_plus"],
                label=result["label_plus"],
                boundary_id=boundary_id,
                side="plus",
            )
        )

    def collect(
        self,
        seed_x,
        max_points=1000,
        max_trials=5000,
        max_query_budget=None,
        max_time_seconds=None,
    ):
        self.max_query_budget = max_query_budget
        start_time = time.perf_counter()

        def time_remaining():
            if max_time_seconds is None:
                return True
            return time.perf_counter() - start_time < max_time_seconds

        seed_x = np.asarray(seed_x, dtype=np.float32)
        seed_labels = []

        for x in seed_x:
            if not self._has_budget() or not time_remaining():
                break

            seed_labels.append(
                self._query_one(
                    x,
                    source="seed_natural",
                    store_ordinary=True,
                )
            )
        seed_labels = np.asarray(seed_labels)

        label_values = np.unique(seed_labels)
        if len(label_values) < 2:
            raise ValueError("At least two target labels are required to collect decision points.")

        label_to_indices = {
            label: np.flatnonzero(seed_labels == label)
            for label in label_values
        }

        trials = 0
        while (
            len(self.decision_records) < max_points
            and trials < max_trials
            and self._has_budget(required_queries=self.binary_search_steps + 2)
            and time_remaining()
        ):
            trials += 1

            if self.rng.random() < 0.7:
                label_a, label_b = self.rng.choice(label_values, size=2, replace=False)
                idx_a = self.rng.choice(label_to_indices[label_a])
                idx_b = self.rng.choice(label_to_indices[label_b])

                result = self.search_between_pair(
                    x_a=seed_x[idx_a],
                    y_a=int(label_a),
                    x_b=seed_x[idx_b],
                    y_b=int(label_b),
                )
            else:
                idx = self.rng.integers(0, len(seed_x))
                direction = self.rng.normal(size=seed_x.shape[1])
                result = self.search_along_direction(seed_x[idx], direction)

            if result is not None:
                self._add_decision_result(result)

        return {
            "ordinary_count": len(self.ordinary_records),
            "decision_point_count": len(self.decision_records),
            "boundary_side_count": len(self.boundary_side_records),
            "oracle_query_count": self.oracle.get_query_count(),
            "max_query_budget": max_query_budget,
            "trials": trials,
        }

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(self.ordinary_records).to_csv(
            output_dir / "ordinary_query_samples.csv",
            index=False,
        )
        pd.DataFrame(self.decision_records).to_csv(
            output_dir / "decision_points.csv",
            index=False,
        )
        pd.DataFrame(self.boundary_side_records).to_csv(
            output_dir / "boundary_side_samples.csv",
            index=False,
        )
