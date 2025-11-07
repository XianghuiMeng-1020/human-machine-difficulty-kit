import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("analysis/ednet_scale/ednet_scaling.csv")
plt.figure(figsize=(5,3))
plt.plot(df["n_users"], df["p_low"], marker="o", label="低曝光")
plt.plot(df["n_users"], df["p_hard"], marker="o", label="困难H_proxy")
plt.xlabel("number of users (slice)")
plt.ylabel("ratio")
plt.title("EdNet-KT1 coverage → difficulty buckets")
plt.legend()
plt.tight_layout()
os.makedirs("figs/ednet_scale", exist_ok=True)
plt.savefig("figs/ednet_scale/ednet_scaling.png", dpi=300)
print("✅ saved figs/ednet_scale/ednet_scaling.png")
