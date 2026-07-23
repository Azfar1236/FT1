#!/usr/bin/env python3
"""
train_classifier.py

Trains the classifier used by discover_hidden_objects.py. 3-class scheme
(Null, Transient, Continuous Variable) matching what Paz's actual
production pipeline uses (VarWISE, Paz et al. 2026) - the original paper's
4-class split (Null/Transient/Pulsating/Transit) was a proof-of-concept;
the real pipeline collapses Pulsating+Transit into "Continuous Variable"
and handles finer sub-typing with a separate step afterward.

HONESTY NOTE: FEFT's exact formula and VARnet's exact architecture are
not published anywhere I could find (no code release exists for either
paper). This uses a non-uniform DFT + standard wavelet transform as a
documented, working stand-in - see discover_hidden_objects.py's docstring
for the full explanation. This is a genuine, functional classifier built
on the same principles; it is not a byte-for-byte reproduction.

Usage
-----
    pip install torch numpy pywavelets --break-system-packages
    python3 train_classifier.py --epochs 30 --n-train 20000
"""

import argparse
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CLASSES = ["null", "transient", "continuous_variable"]
N_CLASSES = len(CLASSES)

GRID_LEN = 256
BASELINE_DAYS = 10.5 * 365.25
GROUP_SPACING_DAYS = 365.25 / 2
N_GROUPS = int(BASELINE_DAYS / GROUP_SPACING_DAYS)


def sample_observation_times(rng):
    times = []
    for g in range(N_GROUPS):
        group_center = g * GROUP_SPACING_DAYS + rng.uniform(-5, 5)
        n_in_group = rng.integers(12, 17)
        offsets = rng.uniform(-0.2, 0.2, size=n_in_group)
        times.extend(group_center + offsets)
    return np.sort(np.array(times))


def gen_null(rng, times, baseline_mag=14.0, noise=0.03):
    return baseline_mag + rng.normal(0, noise, size=len(times))


def gen_transient(rng, times, baseline_mag=14.0, noise=0.03):
    mags = baseline_mag + rng.normal(0, noise, size=len(times))
    t_event = rng.uniform(times.min(), times.max())
    amplitude = rng.uniform(1.0, 4.0)
    decay_days = rng.uniform(5, 60)
    dt = times - t_event
    decay = np.where(dt >= 0, np.exp(-dt / decay_days), 0.0)
    mags -= amplitude * decay
    return mags


def gen_continuous_variable(rng, times, baseline_mag=14.0, noise=0.03):
    """Merges the old Pulsating + Transit classes - variability that
    persists across the whole light curve rather than a single event."""
    mags = baseline_mag + rng.normal(0, noise, size=len(times))
    if rng.random() < 0.5:
        # smooth periodic (old "pulsating")
        period_days = rng.uniform(0.5, 300)
        amplitude = rng.uniform(0.1, 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        mags += amplitude * np.sin(2 * np.pi * times / period_days + phase)
    else:
        # sharp periodic dips (old "transit")
        period_days = rng.uniform(1, 400)
        depth = rng.uniform(0.3, 2.0)
        duty_cycle = rng.uniform(0.02, 0.08)
        phase = rng.uniform(0, period_days)
        t_mod = (times + phase) % period_days
        in_transit = t_mod < (duty_cycle * period_days)
        mags[in_transit] += depth
    return mags


GENERATORS = {
    "null": gen_null,
    "transient": gen_transient,
    "continuous_variable": gen_continuous_variable,
}


def resample_to_grid(times, mags):
    grid = np.linspace(times.min(), times.max(), GRID_LEN)
    return np.interp(grid, times, mags)


def ndft_magnitude(times, mags, n_freqs=GRID_LEN // 2):
    t = times - times.min()
    baseline = t.max() if t.max() > 0 else 1.0
    freqs = np.linspace(1.0 / baseline, n_freqs / baseline, n_freqs)
    y = mags - mags.mean()
    phase = -2j * np.pi * np.outer(freqs, t)
    spectrum = np.abs(np.exp(phase) @ y)
    if spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()
    return spectrum


def wavelet_features(grid_mags, wavelet="db4", level=4):
    coeffs = pywt.wavedec(grid_mags, wavelet, level=level)
    detail_channels = []
    for c in coeffs[1:]:
        resampled = np.interp(np.linspace(0, 1, GRID_LEN),
                               np.linspace(0, 1, len(c)), c)
        detail_channels.append(resampled)
    return np.array(detail_channels)


def build_sample(rng, class_name):
    times = sample_observation_times(rng)
    mags = GENERATORS[class_name](rng, times)

    grid_mags = resample_to_grid(times, mags)
    grid_mags_norm = (grid_mags - grid_mags.mean()) / (grid_mags.std() + 1e-6)

    ndft_mag = ndft_magnitude(times, mags)
    ndft_mag_padded = np.interp(np.linspace(0, 1, GRID_LEN),
                                 np.linspace(0, 1, len(ndft_mag)), ndft_mag)

    wave_feats = wavelet_features(grid_mags_norm)

    channels = np.vstack([
        grid_mags_norm[None, :],
        ndft_mag_padded[None, :],
        wave_feats,
    ])
    return channels.astype(np.float32)


class LightCurveDataset(Dataset):
    def __init__(self, n_samples, seed=0):
        rng = np.random.default_rng(seed)
        self.X, self.y = [], []
        for i in range(n_samples):
            class_idx = i % N_CLASSES
            self.X.append(build_sample(rng, CLASSES[class_idx]))
            self.y.append(class_idx)
        self.X = np.stack(self.X)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), self.y[idx]


class VARnetStyleCNN(nn.Module):
    def __init__(self, n_channels, n_classes=N_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train(n_train, n_val, epochs, batch_size, lr, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Generating {n_train} synthetic training light curves...")
    train_ds = LightCurveDataset(n_train, seed=1)
    print(f"Generating {n_val} synthetic validation light curves...")
    val_ds = LightCurveDataset(n_val, seed=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_channels = train_ds.X.shape[1]
    model = VARnetStyleCNN(n_channels=n_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        train_loss = total_loss / len(train_ds)

        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
        val_acc = correct / len(val_ds)

        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "n_channels": n_channels,
        "classes": CLASSES,
        "grid_len": GRID_LEN,
    }, out_path)
    print(f"Saved trained model to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="classifier_model.pt")
    args = ap.parse_args()

    train(args.n_train, args.n_val, args.epochs, args.batch_size,
          args.lr, args.out)


if __name__ == "__main__":
    main()
