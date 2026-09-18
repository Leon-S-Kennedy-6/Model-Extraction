from pathlib import Path

import numpy as np
import pandas as pd


def _sorted_columns(df, prefix):
    columns = [column for column in df.columns if column.startswith(prefix)]
    return sorted(columns, key=lambda column: int(column.split("_")[1]))


class DualPointCollector:
    def __init__(
        self,
        oracle,
        epsilon=0.001,
        neighbor_radius=0.05,
        directions_per_point=8,
        max_candidates=300,
        binary_search_steps=20,
        max_expand_steps=12,
        expand_factor=1.5,
        max_query_budget=None,
        random_state=42,
    ):
        self.oracle = oracle
        self.epsilon = epsilon
        self.neighbor_radius = neighbor_radius
        self.directions_per_point = directions_per_point
        self.max_candidates = max_candidates
        self.binary_search_steps = binary_search_steps
        self.max_expand_steps = max_expand_steps
        self.expand_factor = expand_factor
        self.max_query_budget = max_query_budget
        self.rng = np.random.default_rng(random_state)

        self.ordinary_records = []
        self.aux_decision_records = []
        self.dual_candidate_records = []
        self.dual_neighborhood_records = []

    def _has_budget(self, required_queries=1):
        if self.max_query_budget is None:
            return True
        return self.oracle.get_query_count() + required_queries <= self.max_query_budget

    def _normalize(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return None
        return vector / norm

    def _query_one(
        self,
        x,
        source,
        parent_decision_id=None,
        search_direction_id=None,
        store=True,
    ):
        if not self._has_budget():
            raise RuntimeError("Query budget exhausted.")

        x = np.asarray(x, dtype=np.float32)
        label = self.oracle.query_one(x)

        if store:
            self.ordinary_records.append(
                self._ordinary_record(
                    x=x,
                    label=label,
                    source=source,
                    parent_decision_id=parent_decision_id,
                    search_direction_id=search_direction_id,
                )
            )

        return label

    def _ordinary_record(
        self,
        x,
        label,
        source,
        parent_decision_id=None,
        search_direction_id=None,
    ):
        record = {f"feature_{i}": float(value) for i, value in enumerate(x)}
        record["label"] = int(label)
        record["source"] = source
        record["query_index"] = int(self.oracle.get_query_count())
        record["parent_decision_id"] = (
            -1 if parent_decision_id is None else int(parent_decision_id)
        )
        record["search_direction_id"] = (
            -1 if search_direction_id is None else int(search_direction_id)
        )
        return record

    def _binary_search_decision(
        self,
        x_low,
        y_low,
        x_high,
        y_high,
        parent_decision_id,
        search_direction_id,
    ):
        if y_low == y_high:
            return None

        left = np.asarray(x_low, dtype=np.float32)
        right = np.asarray(x_high, dtype=np.float32)
        left_label = int(y_low)
        right_label = int(y_high)
        query_start = self.oracle.get_query_count()

        for _ in range(self.binary_search_steps):
            if not self._has_budget():
                return None

            mid = (left + right) / 2.0
            mid_label = self._query_one(
                mid,
                source="dual_aux_decision_binary_mid",
                parent_decision_id=parent_decision_id,
                search_direction_id=search_direction_id,
                store=True,
            )

            if mid_label == left_label:
                left = mid
                left_label = mid_label
            else:
                right = mid
                right_label = mid_label

        x_boundary = (left + right) / 2.0
        direction = self._normalize(right - left)
        if direction is None:
            return None

        x_minus = x_boundary - self.epsilon * direction
        x_plus = x_boundary + self.epsilon * direction
        label_minus = self._query_one(
            x_minus,
            source="dual_aux_decision_side_minus",
            parent_decision_id=parent_decision_id,
            search_direction_id=search_direction_id,
            store=True,
        )
        label_plus = self._query_one(
            x_plus,
            source="dual_aux_decision_side_plus",
            parent_decision_id=parent_decision_id,
            search_direction_id=search_direction_id,
            store=True,
        )

        if label_minus == label_plus:
            return None

        return {
            "x_boundary": x_boundary,
            "direction": direction,
            "label_minus": label_minus,
            "label_plus": label_plus,
            "queries_used": self.oracle.get_query_count() - query_start,
        }

    def _boundary_like(
        self,
        x,
        normal_direction,
        parent_decision_id,
        search_direction_id,
        source_prefix,
    ):
        x = np.asarray(x, dtype=np.float32)
        x_minus = x - self.epsilon * normal_direction
        x_plus = x + self.epsilon * normal_direction

        label_minus = self._query_one(
            x_minus,
            source=f"{source_prefix}_minus",
            parent_decision_id=parent_decision_id,
            search_direction_id=search_direction_id,
            store=True,
        )
        label_plus = self._query_one(
            x_plus,
            source=f"{source_prefix}_plus",
            parent_decision_id=parent_decision_id,
            search_direction_id=search_direction_id,
            store=True,
        )

        return label_minus != label_plus, label_minus, label_plus

    def _add_aux_decision_record(
        self,
        result,
        parent_decision_id,
        search_direction_id,
    ):
        aux_decision_id = len(self.aux_decision_records)
        x_boundary = result["x_boundary"]
        direction = result["direction"]

        record = {f"feature_{i}": float(value) for i, value in enumerate(x_boundary)}
        record.update(
            {f"direction_{i}": float(value) for i, value in enumerate(direction)}
        )
        record["label_minus"] = int(result["label_minus"])
        record["label_plus"] = int(result["label_plus"])
        record["epsilon"] = float(self.epsilon)
        record["source"] = "dual_search_aux_decision"
        record["parent_decision_id"] = int(parent_decision_id)
        record["search_direction_id"] = int(search_direction_id)
        record["queries_used"] = int(result["queries_used"])

        self.aux_decision_records.append(record)
        return aux_decision_id

    def _add_dual_candidate_record(
        self,
        x_dual,
        local_direction,
        normal_direction,
        label_minus,
        label_plus,
        parent_decision_id,
        search_direction_id,
        aux_decision_id,
        left_alpha,
        right_alpha,
        queries_used,
    ):
        dual_id = len(self.dual_candidate_records)
        record = {f"feature_{i}": float(value) for i, value in enumerate(x_dual)}
        record.update(
            {
                f"local_direction_{i}": float(value)
                for i, value in enumerate(local_direction)
            }
        )
        record.update(
            {
                f"normal_direction_{i}": float(value)
                for i, value in enumerate(normal_direction)
            }
        )
        record["dual_candidate_id"] = int(dual_id)
        record["label_minus"] = int(label_minus)
        record["label_plus"] = int(label_plus)
        record["parent_decision_id"] = int(parent_decision_id)
        record["search_direction_id"] = int(search_direction_id)
        record["aux_decision_id"] = int(aux_decision_id)
        record["epsilon"] = float(self.epsilon)
        record["left_boundary_alpha"] = float(left_alpha)
        record["right_non_boundary_alpha"] = float(right_alpha)
        record["queries_used"] = int(queries_used)
        record["source"] = "dual_candidate_boundary_departure"

        self.dual_candidate_records.append(record)
        return dual_id

    def _sample_dual_neighborhood(
        self,
        x_dual,
        local_direction,
        normal_direction,
        dual_candidate_id,
        parent_decision_id,
        search_direction_id,
    ):
        directions = [
            ("normal", normal_direction),
            ("local", local_direction),
        ]

        combo_plus = self._normalize(normal_direction + local_direction)
        combo_minus = self._normalize(normal_direction - local_direction)
        if combo_plus is not None:
            directions.append(("normal_plus_local", combo_plus))
        if combo_minus is not None:
            directions.append(("normal_minus_local", combo_minus))

        for direction_name, direction in directions:
            for side, sign in [("minus", -1.0), ("plus", 1.0)]:
                if not self._has_budget():
                    return

                x_sample = x_dual + sign * self.epsilon * direction
                label = self._query_one(
                    x_sample,
                    source=f"dual_neighborhood_{direction_name}_{side}",
                    parent_decision_id=parent_decision_id,
                    search_direction_id=search_direction_id,
                    store=True,
                )

                record = {
                    f"feature_{i}": float(value)
                    for i, value in enumerate(x_sample)
                }
                record["label"] = int(label)
                record["dual_candidate_id"] = int(dual_candidate_id)
                record["parent_decision_id"] = int(parent_decision_id)
                record["search_direction_id"] = int(search_direction_id)
                record["direction_name"] = direction_name
                record["side"] = side
                record["epsilon"] = float(self.epsilon)
                self.dual_neighborhood_records.append(record)

    def _search_departure_point(
        self,
        x2,
        normal_direction,
        local_direction,
        label_minus,
        label_plus,
        parent_decision_id,
        search_direction_id,
        aux_decision_id,
    ):
        query_start = self.oracle.get_query_count()
        left_alpha = 0.0
        left_point = np.asarray(x2, dtype=np.float32)
        right_alpha = None
        right_point = None

        step = self.neighbor_radius
        for _ in range(self.max_expand_steps):
            if not self._has_budget(required_queries=2):
                return None

            candidate = x2 + step * local_direction
            is_boundary, _, _ = self._boundary_like(
                candidate,
                normal_direction,
                parent_decision_id=parent_decision_id,
                search_direction_id=search_direction_id,
                source_prefix="dual_departure_probe",
            )

            if is_boundary:
                left_alpha = step
                left_point = candidate
                step *= self.expand_factor
            else:
                right_alpha = step
                right_point = candidate
                break

        if right_point is None:
            return None

        for _ in range(self.binary_search_steps):
            if not self._has_budget(required_queries=2):
                return None

            mid_alpha = (left_alpha + right_alpha) / 2.0
            mid_point = x2 + mid_alpha * local_direction
            is_boundary, _, _ = self._boundary_like(
                mid_point,
                normal_direction,
                parent_decision_id=parent_decision_id,
                search_direction_id=search_direction_id,
                source_prefix="dual_departure_binary_mid",
            )

            if is_boundary:
                left_alpha = mid_alpha
                left_point = mid_point
            else:
                right_alpha = mid_alpha
                right_point = mid_point

        x_dual = (left_point + right_point) / 2.0
        dual_id = self._add_dual_candidate_record(
            x_dual=x_dual,
            local_direction=local_direction,
            normal_direction=normal_direction,
            label_minus=label_minus,
            label_plus=label_plus,
            parent_decision_id=parent_decision_id,
            search_direction_id=search_direction_id,
            aux_decision_id=aux_decision_id,
            left_alpha=left_alpha,
            right_alpha=right_alpha,
            queries_used=self.oracle.get_query_count() - query_start,
        )
        self._sample_dual_neighborhood(
            x_dual=x_dual,
            local_direction=local_direction,
            normal_direction=normal_direction,
            dual_candidate_id=dual_id,
            parent_decision_id=parent_decision_id,
            search_direction_id=search_direction_id,
        )

        return dual_id

    def _search_one_direction(
        self,
        x2,
        normal_direction,
        label_minus,
        label_plus,
        parent_decision_id,
        search_direction_id,
    ):
        if not self._has_budget(required_queries=self.binary_search_steps + 8):
            return None

        random_direction = self._normalize(self.rng.normal(size=x2.shape[0]))
        if random_direction is None:
            return None

        x3 = x2 + self.neighbor_radius * random_direction
        label_x3 = self._query_one(
            x3,
            source="dual_random_probe_x3",
            parent_decision_id=parent_decision_id,
            search_direction_id=search_direction_id,
            store=True,
        )

        if label_x3 == label_minus:
            x_ref = x2 + self.epsilon * normal_direction
            y_ref = label_plus
        elif label_x3 == label_plus:
            x_ref = x2 - self.epsilon * normal_direction
            y_ref = label_minus
        else:
            return None

        aux_result = self._binary_search_decision(
            x_low=x3,
            y_low=label_x3,
            x_high=x_ref,
            y_high=y_ref,
            parent_decision_id=parent_decision_id,
            search_direction_id=search_direction_id,
        )
        if aux_result is None:
            return None

        aux_decision_id = self._add_aux_decision_record(
            aux_result,
            parent_decision_id=parent_decision_id,
            search_direction_id=search_direction_id,
        )

        local_direction = self._normalize(aux_result["x_boundary"] - x2)
        if local_direction is None:
            return None

        for signed_direction in [local_direction, -local_direction]:
            if len(self.dual_candidate_records) >= self.max_candidates:
                return None

            result = self._search_departure_point(
                x2=x2,
                normal_direction=normal_direction,
                local_direction=signed_direction,
                label_minus=label_minus,
                label_plus=label_plus,
                parent_decision_id=parent_decision_id,
                search_direction_id=search_direction_id,
                aux_decision_id=aux_decision_id,
            )
            if result is not None:
                return result

        return None

    def collect(self, decision_points_path):
        decision_df = pd.read_csv(decision_points_path)
        feature_columns = _sorted_columns(decision_df, "feature_")
        direction_columns = _sorted_columns(decision_df, "direction_")

        decision_indices = np.arange(len(decision_df))
        self.rng.shuffle(decision_indices)

        trials = 0
        for parent_decision_id in decision_indices:
            if len(self.dual_candidate_records) >= self.max_candidates:
                break
            if not self._has_budget(required_queries=self.binary_search_steps + 8):
                break

            row = decision_df.iloc[parent_decision_id]
            x2 = row[feature_columns].values.astype("float32")
            normal_direction = self._normalize(
                row[direction_columns].values.astype("float32")
            )
            if normal_direction is None:
                continue

            label_minus = int(row["label_minus"])
            label_plus = int(row["label_plus"])

            for search_direction_id in range(self.directions_per_point):
                if len(self.dual_candidate_records) >= self.max_candidates:
                    break
                if not self._has_budget(required_queries=self.binary_search_steps + 8):
                    break

                trials += 1
                self._search_one_direction(
                    x2=x2,
                    normal_direction=normal_direction,
                    label_minus=label_minus,
                    label_plus=label_plus,
                    parent_decision_id=parent_decision_id,
                    search_direction_id=search_direction_id,
                )

        return {
            "ordinary_count": len(self.ordinary_records),
            "aux_decision_point_count": len(self.aux_decision_records),
            "dual_candidate_count": len(self.dual_candidate_records),
            "dual_neighborhood_sample_count": len(self.dual_neighborhood_records),
            "oracle_query_count": self.oracle.get_query_count(),
            "max_query_budget": self.max_query_budget,
            "trials": trials,
        }

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(self.ordinary_records).to_csv(
            output_dir / "dual_search_ordinary_samples.csv",
            index=False,
        )
        pd.DataFrame(self.aux_decision_records).to_csv(
            output_dir / "dual_search_decision_points.csv",
            index=False,
        )
        pd.DataFrame(self.dual_candidate_records).to_csv(
            output_dir / "dual_point_candidates.csv",
            index=False,
        )
        pd.DataFrame(self.dual_neighborhood_records).to_csv(
            output_dir / "dual_neighborhood_samples.csv",
            index=False,
        )
