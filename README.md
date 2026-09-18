# hard_label_boundary_extraction


requirements: Python 3.13.9, PyTorch 2.12.1

`src/api/hard_label_oracle.py`:
provide the hard-label query interface and record query counts.

`src/attacks/decision_point_collector.py`:
collect ordinary query samples and decision-boundary points.

`src/attacks/dual_point_collector.py`:
search for dual-point candidates from collected decision points.

`src/attacks/boundary_sampler.py`:
sample local neighborhoods around decision-boundary points.

`src/training/train_substitute.py`:
train the initial substitute model with hard labels.

`src/training/train_boundary_enhanced.py`:
perform decision-boundary-enhanced training.

`src/training/train_dual_enhanced.py`:
perform dual-point-enhanced training.

`src/training/train_active_iteration.py`:
perform active iterative extraction and refinement.



