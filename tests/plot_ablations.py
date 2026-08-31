import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# from damply import dirs
import sys




timing = pd.read_csv("compute_time.csv",index_col=0)

print(timing)
sns.lineplot(timing, x= 'Num. Obs', y = 'Computation Time', hue = 'Num. Neighbors')
plt.title("Compute Time (seconds) vs. Number of Observations")
plt.savefig("compute_time.png")
plt.close()

ablation = pd.read_csv("ablation.csv",index_col=0)
sns.barplot(ablation, x="AUC", y="Modalities", hue="Model",legend=False)
plt.tight_layout()
plt.savefig("modality_ablation.png")
plt.close()

print("PRINT NBR ABLATION")
neighbors = pd.read_csv("auc.csv",index_col=0)
neighbors = neighbors[neighbors['Model'].isin(['SNF'])]
sns.lineplot(data=neighbors, x="K", y="AUC",hue="Model")
plt.xlabel("Num. Neighbours")
plt.savefig("nbr_ablation.png")
plt.close()