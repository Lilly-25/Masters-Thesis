import numpy as np
class MinMaxScaler: # Not used anymore
    def fit(self, x, flatten):
        # Calculate maximum and minimum values, ignoring NaNs
        max_values = np.nanmax(x, axis=0)
        min_values = np.nanmin(x, axis=0)

        # Avoid division by 0, when max and min are same only 1 element in the column in that case after scaling value becomes 0
        range_values = max_values - min_values
        if not flatten:
            range_values[range_values == 0] = 1
        else:
            if range_values == 0:
                range_values = 1

        return min_values, range_values

    def transform(self, x, min_values, range_values):
        # Apply the Min-Max scaling, preserving NaNs
        return np.where(np.isnan(x), np.nan, (x - min_values) / range_values)

    def inverse_transform(self, z, min_values, range_values):
        return z * range_values + min_values

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
    
