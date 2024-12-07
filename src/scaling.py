import numpy as np
class MinMaxScaler:
    def fit(self, x):
        max_value = np.nanmax(x)
        min_value = np.nanmin(x)
        return min_value, max_value

    def transform(self, x, min_value, max_value):
        # Apply Min-Max scaling and centralize to [-0.5, 0.5]
        normalized = (x - min_value) / (max_value - min_value)
        return np.where(np.isnan(x), np.nan, normalized - 0.5)

    def inverse_transform(self, z, min_value, max_value):
        # Reverse centralization and scaling
        return (z + 0.5) * (max_value - min_value) + min_value

class StdScaler:
    def fit(self, x):
        # Calculate mean and stddev ignoring NANs
        x = np.array(x)
        if x.ndim == 1:
            # If 1D array, handle scalars
            mean = np.nanmean(x)
            std_dev = np.nanstd(x)
            std_dev = 1 if std_dev == 0 else std_dev
        else:
            mean = np.nanmean(x, axis=0)
            std_dev = np.nanstd(x, axis=0)
            std_dev[std_dev == 0] = 1
         
        return mean, std_dev

    def transform(self, x, mean, std_dev):
        # Apply the std scaling scaling, preserving NaNs
        return np.where(np.isnan(x), np.nan, (x - mean) / std_dev)

    def inverse_transform(self, z, mean, std_dev):
        return z * std_dev + mean
    
