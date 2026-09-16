import pandas as pd
import matplotlib.pyplot as plt

df_iid = pd.read_csv("results_iid.csv")
df_non_iid = pd.read_csv("results_non_iid.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(df_iid["round"], df_iid["accuracy"], marker="o", label="IID")
axes[0].plot(df_non_iid["round"], df_non_iid["accuracy"], marker="s", label="Non-IID")
axes[0].set_xlabel("Round")
axes[0].set_ylabel("Accuracy (centralizzata)")
axes[0].set_title("Accuracy globale per round")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df_iid["round"], df_iid["loss"], marker="o", label="IID")
axes[1].plot(df_non_iid["round"], df_non_iid["loss"], marker="s", label="Non-IID")
axes[1].set_xlabel("Round")
axes[1].set_ylabel("Loss (centralizzata)")
axes[1].set_title("Loss globale per round")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("confronto_iid_non_iid.png", dpi=150)
plt.show()