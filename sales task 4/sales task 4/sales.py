import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
data = {
    'TV': np.random.uniform(10, 200, 200),
    'Radio': np.random.uniform(5, 100, 200),
    'Social': np.random.uniform(20, 150, 200),
    'Audience': np.random.choice([0,1,2], 200)  # Encoded
}
data['Sales'] = 50 + 0.1*data['TV'] + 0.2*data['Radio'] + 0.15*data['Social'] + np.random.normal(0, 5, 200)

df = pd.DataFrame(data)
print(f"Shape: {df.shape}")
print(df.head(3))
print("Sales:", df['Sales'].describe()[['mean', 'min', 'max']])
plt.figure(figsize=(12, 4))
plt.subplot(1,3,1); df['Sales'].hist(bins=20); plt.title('Sales')
plt.subplot(1,3,2); plt.scatter(df['TV'], df['Sales']); plt.title('TV vs Sales')
plt.subplot(1,3,3); plt.scatter(df['Social'], df['Sales']); plt.title('Social vs Sales')
plt.tight_layout()  # Fixed typo: splt -> plt
plt.show()

print("✅ Ready!")

