#!/bin/bash

# multi_train.sh
#
# Usage:    multi_train.sh
#

uv run train_autoencoder.py wandb.mode="offline" log_freq=10 device="cuda"
uv run train_autoencoder.py wandb.mode="offline" log_freq=10 device="cuda" max_pooling=True
uv run train_autoencoder.py wandb.mode="offline" log_freq=10 device="cuda" sigmoid=False
uv run train_autoencoder.py wandb.mode="offline" log_freq=10 device="cuda" sigmoid=False max_pooling=True
uv run train_autoencoder.py wandb.mode="offline" log_freq=10 device="cuda" latent_dim=64
uv run train_autoencoder.py wandb.mode="offline" log_freq=10 device="cuda" max_pooling=True latent_dim=64
uv run train_autoencoder.py wandb.mode="offline" log_freq=10 device="cuda" sigmoid=False latent_dim=64
uv run train_autoencoder.py wandb.mode="offline" log_freq=10 device="cuda" sigmoid=False max_pooling=True latent_dim=64