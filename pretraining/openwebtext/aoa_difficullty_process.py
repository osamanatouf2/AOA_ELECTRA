import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# === 1. Load the AoA data ===
df = pd.read_csv("data/aoa.csv").dropna(subset=["Word", "AoA"])

# lowercase normalization
df["Word"] = df["Word"].str.lower()

# === 2. Create difficulty bins ===
# You can adjust the number of bins depending on how many stages you want.
n_bins = 7  # e.g., 5 stages of curriculum learning

# Use quantile-based binning (each bin has roughly equal number of words)
df["AoA_bin"], bins = pd.qcut(df["AoA"], q=n_bins, labels=False, retbins=True)

print("Bin thresholds:", bins)
print(df.head())

# === 3. Inspect the bin distribution ===
print(df["AoA_bin"].value_counts().sort_index())

# === 4. Save results ===
df.to_csv("data/aoa_with_bins.csv", index=False)

# Create lookup dictionaries
word2aoa = dict(zip(df["Word"], df["AoA"]))
word2bin = dict(zip(df["Word"], df["AoA_bin"]))

# Save as pickle files for quick loading in training scripts
with open("data/word2aoa.pkl", "wb") as f:
    pickle.dump(word2aoa, f)

with open("data/word2bin.pkl", "wb") as f:
    pickle.dump(word2bin, f)

# === 5. (Optional) Plot the AoA histogram ===
plt.figure(figsize=(8, 4))
plt.hist(df["AoA"], bins=30, color="skyblue", edgecolor="black")
plt.title("AoA Distribution")
plt.xlabel("Age of Acquisition")
plt.ylabel("Word Count")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/aoa_histogram.png")
plt.close()

print("✅ Preprocessing complete.")
print(f"Saved: aoa_with_bins.csv, word2aoa.pkl, word2bin.pkl, aoa_histogram.png")


