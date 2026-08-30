"""Muscle Memory — search hundreds of hours of skilled human work by motion.

Pipeline: ingest (windows -> video embeddings + IMU) -> imu_encoder (contrastive
IMU tower) -> index (search bank + 2D atlas) -> server (atlas UI, text search,
live phone-motion search) -> eval (honest retrieval metrics).

Every stage runs in `--source fixture` mode on synthetic data so the whole
system can be smoke-tested without the World Context flash drive.
"""

__version__ = "0.1.0"
