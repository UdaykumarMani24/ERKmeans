import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# YOUR REAL RESULTS (from your experiment output)
data = {
    'Dataset': ['Iris', 'Wine', 'Soybean', 'Breast\nCancer', 'Heart\nDisease', 'MNIST'],
    'K-Means': [0.9986, 0.9952, 0.9251, 0.9217, 0.9563, 0.9787],
    'K-Means_std': [0.0009, 0.0001, 0.0197, 0.0044, 0.0007, 0.0089],
    'ER-KM': [0.9691, 0.9940, 0.9247, 0.9207, 0.9615, 0.9662],
    'ER-KM_std': [0.0732, 0.0014, 0.0242, 0.0051, 0.0109, 0.0132]
}

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df['Dataset']))
width = 0.35

bars1 = ax.bar(x - width/2, df['K-Means'], width, label='K-Means', 
               color='#2E86AB', yerr=df['K-Means_std'], capsize=5)
bars2 = ax.bar(x + width/2, df['ER-KM'], width, label='ER-K-Means', 
               color='#A23B72', yerr=df['ER-KM_std'], capsize=5)

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Normalized Entropy (H_norm)', fontsize=12)
ax.set_title('Figure 1: Cluster Balance Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df['Dataset'], fontsize=10)
ax.legend(loc='lower right')
ax.set_ylim(0.85, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_balance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("? Figure 1 saved with YOUR REAL experimental results!")