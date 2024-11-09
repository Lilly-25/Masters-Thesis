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
    


# class StdScaler:
#     def __init__(self):
#         self.mean = None
#         self.std_dev = None

#     def fit(self, x, input):
#         x = np.array(x)
#         if x.ndim == 1:
#             # If 1D array, handle scalars
#             self.mean = np.nanmean(x)
#             self.std_dev = np.nanstd(x)
#             self.std_dev = 1 if self.std_dev == 0 else self.std_dev
#         elif input == 'x':
#             # For 2D arrays, calculate mean and std across columns
#             self.mean = np.nanmean(x, axis=0)
#             self.std_dev = np.nanstd(x, axis=0)
#             self.std_dev[self.std_dev == 0] = 1
#         elif input == 'y1':
#             # For 2D arrays, calculate mean and std across rows independently
#             self.mean = np.nanmean(x, axis=1, keepdims=True)
#             self.std_dev = np.nanstd(x, axis=1, keepdims=True)
#             self.std_dev[self.std_dev == 0] = 1
#         elif input == 'y2' and x.ndim == 3:
#             # For 3D arrays, calculate mean and std across the last two dimensions for each sample
#             self.mean = np.nanmean(x, axis=(1, 2), keepdims=True)
#             self.std_dev = np.nanstd(x, axis=(1, 2), keepdims=True)
#             self.std_dev[self.std_dev == 0] = 1

#     def transform(self, x):
#         x = np.array(x)
#         if self.mean is None or self.std_dev is None:
#             raise RuntimeError("The scaler has not been fitted yet.")
#         return np.where(np.isnan(x), np.nan, (x - self.mean) / self.std_dev)
        
#     def inverse_transform(self, z):
#         if self.mean is None or self.std_dev is None:
#             raise RuntimeError("The scaler has not been fitted yet.")
#         return z * self.std_dev + self.mean



# def graph_scaling_params(features_by_type):
#     """
#     Compute mean and standard deviation for each feature.
    
#     Args:
#         features_by_type (dict): Dictionary containing features by type
    
#     Returns:
#         dict: Dictionary containing mean and std for each feature
#     """
#     scaling_params = {}
    
#     for type_name, features_dict in features_by_type.items():
#         scaling_params[type_name] = {
#             'mean': {},
#             'std': {}
#         }
        
#         for feature_idx, feature_values in features_dict.items():
#             values = np.array(feature_values)
#             mean = np.mean(values)
#             std = np.std(values)
#             # Handle zero standard deviation
#             if std == 0:
#                 std = 1
                
#             scaling_params[type_name]['mean'][feature_idx] = mean
#             scaling_params[type_name]['std'][feature_idx] = std
    
#     return scaling_params


def graph_scaling_params(features_by_type):
    """
    Compute mean and standard deviation for each feature using StdScaler.
    
    Args:
        features_by_type (dict): Dictionary containing features by type
    
    Returns:
        dict: Dictionary containing mean and std for each feature
    """
    scaling_params = {}
    scaler = StdScaler()
    
    for type_name, features_dict in features_by_type.items():
        scaling_params[type_name] = {
            'mean': {},
            'std': {}
        }
        
        for feature_idx, feature_values in features_dict.items():
            values = np.array(feature_values)
            mean, std = scaler.fit(values)
            
            scaling_params[type_name]['mean'][feature_idx] = mean
            scaling_params[type_name]['std'][feature_idx] = std
    
    return scaling_params

def scale_graph(G, node_scaling_params, edge_scaling_params):
    """
    Scale features using complete edge type information.
    """
    scaled_G = G.copy()
    
    # Scale node features
    for node, data in scaled_G.nodes(data=True):
        node_type = data['type']
        features = np.array(data['features'])
        
        scaled_features = []
        for i, feature in enumerate(features):
            mean = node_scaling_params[node_type]['mean'][i]
            std = node_scaling_params[node_type]['std'][i]
            scaled_feature = (feature - mean) / std
            scaled_features.append(scaled_feature)
        
        scaled_G.nodes[node]['features'] = scaled_features
    
    # Scale edge features using complete edge type
    for u, v, k, data in scaled_G.edges(data=True, keys=True):
        # Get complete edge type descriptor
        src_type = G.nodes[u]['type']
        dst_type = G.nodes[v]['type']
        edge_type = data['type']
        full_edge_type = (src_type, edge_type, dst_type)
        
        features = np.array(data['features'])
        scaled_features = []
        
        try:
            for i, feature in enumerate(features):
                mean = edge_scaling_params[full_edge_type]['mean'][i]
                std = edge_scaling_params[full_edge_type]['std'][i]
                scaled_feature = (feature - mean) / std
                scaled_features.append(scaled_feature)
            
            scaled_G.edges[u, v, k]['features'] = scaled_features
            
        except KeyError as e:
            print(f"Error scaling edge {u}->{v} of type {full_edge_type}")
            print(f"Available edge types: {list(edge_scaling_params.keys())}")
            raise e
    
    return scaled_G