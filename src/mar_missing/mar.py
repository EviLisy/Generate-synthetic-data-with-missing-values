import numpy as np
import pandas as pd
from scipy.special import expit
from scipy import optimize


class MAR:
    def __init__(self,
                 dataset,
                 missing_rate,
                 same_miss_prob = True,
                 target_vars = None,
                 deter_vars = None,
                 d_deter = None,
                 split = None,
                 model = None,
                 weights = None,
                 weights_range = None,
                 seed = None):
        '''
        Generate Missing At Random (MAR) missing dataset given a complete dataset and missing rate.

        Case 1: If target_vars (or deter_vars) and a model are provided, we will use the model to compute probabilities and adjust them to match overall missing_rate.
        Case 2: If deter_vars is provided but model is not provided, we will use binary search to find the best coefficients for a logistic model to achieve the desired missing_rate.
        Case 3: If d_deter and split are provided, we will randomly split the dataset into provided number of splits, and randomly choose deter_vars and target_vars in each split.
                The coefficients for the logistic model will be randomly selected by binary search. After mask each split, the splits will be concatenated back to original order. 
                This allows for more complex missingness patterns where different subsets of data have different missingness mechanisms. 
                If user don't want to split the dataset but want to randomly select deter_vars and target_vars, they can set split=1.

        Parameters
        ----------
        dataset: pd.Dataframe
            complete dataset which determines the shape of final output.
            dataset should not contain any missing values, and all variables should be numeric.
        missing_rate: float
            Desired overall missing proportion of dataset.
        same_miss_prob: bool, optional
            Whether each target variable share the same missing probability. If False, we will randomly separate missingness 
            to target variables and compute different missing probabilities for each target variable. 
            If True, each target variable will share the same missing probability computed based on overall missing_rate and number of target variables.
        target_vars: list of str, optional
            Variable names to introduce missingness into. (e.g., 'X1' or ['X1','X2'])
            if none is provided, target_vars will be randomly selected from dataset columns.
        deter_vars: list of str, optional
            Determining features names that are fully observed columns driving missingness. (e.g., 'X3' or ['X3','X4'])
            if none is provided, deter_vars will be randomly selected from dataset columns excluding target_vars.
        d_deter: int, optional
            Number of determining features to use. If provided, will randomly select d_deter columns from dataset 
            and the other variable will be target_vars automatically.
        split: int, optional
            If provided, will split dataset into provided number of splits, and randomly choose deter_vars and
            target_vars in each split. This allows for more complex missingness patterns where different subsets 
            of data have different missingness mechanisms. After mask each split, the splits will be concatenated 
            back to original order.
        model: callable
            Function to compute P(missing at target_vars | deter_vars) = f(deter_vars), output can be shaped (n_samples,) or (n_samples, n_target_vars).
            If provided, it implies that the user agrees to adjust the probabilities to match the overall missing_rate, which will distort the original model probabilities. 
            If None, we will use binary search to select coefficients for a logistic model and use that to compute probabilities to make the final missing_rate close to the desired missing_rate.
        weights: list or np.array, optional
            If model is None, user can directly provide the weights coefficients to start binary search for a logistic model.
            If list or np.array is provided, it should have the same length as the number of deter_vars and will be used directly 
            as coefficients for the logistic model.
        weights_range: tuple, optional
            If model is None and weights is not provided, this parameter specifies the range for randomly selecting weights.
            Should be a tuple of two floats (min, max).
        seed: int, optional
            Random seed for reproducibility.
        '''
        # input attributes
        self.target_vars = None
        self.deter_vars = None
        self.model = None
        self.d_deter = None
        self.split = None
        self.weights = None
        self.weights_range = None
        self.seed = seed
        
        # lazy attributes to be computed later
        self.mask = None
        self.missing_prob = None
        self.missing_prob_adjusted = None
        self.split_datasets = None
        self.final_missing_rate = None
        self.col_missing_rates = None
        self.best_bias = None
        self.random_weights = None
       
        # 1. Initialize dataset
        # Check that dataset is 2-dimensional
        if dataset.ndim != 2:
            raise ValueError(f"dataset must be 2-dimensional, got {dataset.ndim} dimensions")
        
        # Check that dataset is a DataFrame or can be converted to a DataFrame
        if isinstance(dataset, np.ndarray):
            dataset = pd.DataFrame(dataset, columns=[f'X{i}' for i in range(dataset.shape[1])])
        if not isinstance(dataset, (pd.DataFrame)):
            raise ValueError(f"dataset must be a pandas DataFrame, got {type(dataset)}")
        
        # Check for missing values in the input dataset
        if dataset.isnull().sum().sum() > 0:
            raise ValueError("Dataset contains missing values. Please provide a complete dataset.")
        
        # Note: categorical variables are allowed. They will be one-hot encoded when needed for probability modeling.
        
        self.dataset = dataset.copy()
        self.n_samples, self.d_dimension = self.dataset.shape

        # 2. Validate overall missing_rate
        if not 0 <= missing_rate <= 1:
            raise ValueError(f"Overall missing_rate must be in [0,1], got {missing_rate}")
        
        self.missing_rate = missing_rate

        # 3. Validate same_miss_prob
        if not isinstance(same_miss_prob, bool):
            raise ValueError(f"same_miss_prob must be a boolean, got {same_miss_prob}")
        
        self.same_miss_prob = same_miss_prob
        
        # 4. Validate target_vars and deter_vars
        # Convert target_vars and deter_vars to numpy arrays and validate shapes
        if target_vars is not None:
            self.target_vars, self.target_names = self._validate_and_convert(target_vars)
            self.dim_target = self.target_vars.shape[1]
        if deter_vars is not None:
            self.deter_vars, self.deter_names = self._validate_and_convert(deter_vars)
            self.dim_deter = self.deter_vars.shape[1]

        # Check that target_vars and deter_vars are not overlapping
        if (target_vars is not None) and (deter_vars is not None):
            assert self.target_names is not None and self.deter_names is not None
            if set(self.target_names) & set(self.deter_names):
                raise ValueError(f"target_vars and deter_vars must not overlap.")
            
        # Determine target_vars when only deter_vars is provided
        if target_vars is None and deter_vars is not None:
            self.target_names = [col for col in self.dataset.columns if col not in self.deter_names]
            self.target_vars = self.dataset[self.target_names]
            self.dim_target = self.target_vars.shape[1]

        # Determine deter_vars when only target_vars is provided
        if target_vars is not None and deter_vars is None:
            self.deter_names = [col for col in self.dataset.columns if col not in self.target_names]
            self.deter_vars = self.dataset[self.deter_names]
            self.dim_deter = self.deter_vars.shape[1]
        
        # Calculate the max missing rate when all target_vars are missing
        if target_vars is not None or deter_vars is not None:
            max_missing_rate = self.dim_target / self.dataset.shape[1]
            if self.missing_rate > max_missing_rate:
                raise ValueError(f"Missing rate {self.missing_rate} is too high for the number of target variables {self.dim_target}."
                                 f"Max missing rate is {max_missing_rate} when all target variables are missing.")
        
        # 5. Validate d_deter
        if d_deter is not None:
            if isinstance(d_deter, int):
                if d_deter <= 0:
                    raise ValueError(f"d_deter must be a positive integer, got {d_deter}")
                if d_deter >= len(self.dataset.columns):
                    raise ValueError(f"d_deter must be less than number of columns in dataset, got {d_deter}")
                
        # Check that d_deter is not provided together with target_vars or deter_vars
        if d_deter is not None and (target_vars is not None or deter_vars is not None):
            raise ValueError(f"d_deter cannot be provided together with target_vars or deter_vars.")
        # If d_deter is not provided, and target_vars and deter_vars are also not provided
        elif d_deter is None and (target_vars is None and deter_vars is None):
            raise ValueError(f"Either d_deter or target_vars/deter_vars must be provided to determine the determining variables and target variables.")

        
        self.d_deter = d_deter
        
        # 6. Validate split
        if split is not None:
            if not isinstance(split, int) or split <= 0:
                raise ValueError(f"split must be a positive integer, got {split}")
            if split > len(self.dataset):
                raise ValueError(f"split cannot be greater than number of samples in dataset, got {split} and {len(self.dataset)}")
            if d_deter is None:
                raise ValueError(f"split should be provided together with d_deter.")
            if target_vars is not None or deter_vars is not None:
                raise ValueError(f"split cannot be provided together with target_vars or deter_vars.")
        
        self.split = split

        # 7. Initialize model
        # Check if deter_vars is acceptable for the given model
        if model is not None:
            if not callable(model):
                raise ValueError(f"model must be a callable function, got {type(model)}")
        
            self.model = model

        # 8. Validate weights
        if weights is not None:
            if model is not None:
                raise ValueError(f"weights cannot be provided when model is provided. ")
            
            # Convert weights to numpy array and allow 1D/2D shapes.
            if np.isscalar(weights):
                weights = np.array([weights], dtype=float)
            elif isinstance(weights, (list, np.ndarray)):
                weights = np.asarray(weights, dtype=float)
            else:
                raise TypeError(f"weights must be a scalar, or a list/array of numbers. Got {weights} of type {type(weights)}")

            if weights.ndim not in (1, 2):
                raise ValueError(f"weights must be 1D or 2D. Got shape {weights.shape}")

            # First dimension always corresponds to d_deter
            expected_deter = self.d_deter if self.d_deter is not None else (self.dataset[self.deter_names].shape[1] if self.deter_vars is not None else None)
            if expected_deter is not None and weights.shape[0] != expected_deter:
                raise ValueError(f"weights first dimension must match number of determining variables. Got {weights.shape[0]} and expected {expected_deter}")

            # If target variables are already known and weights is 2D, validate second dimension
            if weights.ndim == 2 and self.target_vars is not None and weights.shape[1] != self.dim_target:
                raise ValueError(f"weights second dimension must match number of target variables. Got {weights.shape[1]} and expected {self.dim_target}")
        
        self.weights = weights

        # 9. Validate weights_range
        if weights_range is not None:
            if model is not None:
                raise ValueError(f"weights_range cannot be provided when model is provided. Got weights_range={weights_range} and model={model}")
            if weights is not None:
                raise ValueError(f"weights_range cannot be provided when weights are provided. Got weights_range={weights_range} and weights={weights}")
            if not (isinstance(weights_range, tuple) and len(weights_range) == 2 and all(isinstance(w, float) for w in weights_range)):
                raise ValueError(f"weights_range must be a tuple of two numbers (min, max). Got {weights_range}")
            if weights_range[0] >= weights_range[1]:
                raise ValueError(f"weights_range min must be less than max. Got {weights_range}")
        
        self.weights_range = weights_range

        # Raise error if both model and weights (or weights_range) are not provided, since we cannot compute probabilities without either a model or weights for the logistic model.
        if model is None and weights is None and weights_range is None:
            raise ValueError(f"Either model or weights (or weights_range) must be provided to compute missing probabilities.")
    
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

    def _validate_and_convert(self, names):
        '''
        Extract and validate data from dataset using column names.
        Also extract column names if available.

        Parameters
        ----------
        names : str or list of str
            Column names to extract (e.g., 'X1' or ['X1', 'X2']).

        Returns
        -------
        data : pd.DataFrame
        Extracted values as a DataFrame preserving column names.
        names : list of str
        List of column names.
        '''
        # Ensure names is a list
        if isinstance(names, str):
            names = [names]
        elif not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"names must be a string or list of strings, got {names}")
        
        # Check if all names exist in dataset
        for name in names:
            if name not in self.dataset.columns:
                raise ValueError(f"Column '{name}' not found in dataset columns {self.dataset.columns.tolist()}")
            
        # Extract data
        data = self.dataset[names]

        if isinstance(data, pd.Series):
            data = data.to_frame()
        elif isinstance(data, pd.DataFrame):
            data = data.copy()
        else:
            raise ValueError(f"Extracted variable data must be a pandas Series or DataFrame, got {type(data)}")
        
        return data, names
    
    def select_variables(self, data):
        '''
        Given a dataset, randomly select target_vars and deter_vars based on provided d_deter and split.
        
        Parameters
        ----------
        data : pd.DataFrame
            The dataset from which to select variables.

        Returns
        -------
        deter_names : list of str
            List of names for the determinant variables.
        target_names : list of str
            List of names for the target variables.
        '''
        # Extract column names from the dataset
        column_names = data.columns.tolist()

        # Randomly select d_deter columns as deter_vars
        if self.d_deter is None:
            raise ValueError("d_deter must be set before calling select_variables().")
        d_deter = int(self.d_deter)
        deter_names = np.random.choice(column_names, size=d_deter, replace = False).tolist()

        # The remaining columns will be target_vars
        target_names = [col for col in column_names if col not in deter_names]

        return deter_names, target_names
    
    # Compute mean missing probability for each target variable based on the overall missing_rate and number of target variables
    def compute_col_mean_prob(self, n_target_vars, n_samples):
        '''
        Given number of target variables and number of samples, compute mean missing probability for each target variable based on the overall missing_rate of the given data.
        
        Parameters
        ----------
        n_target_vars: int
            Number of target variables.
        n_samples: int
            Number of samples in the dataset.  

        In the setting that each target variable share same missing probability, compute mean missing probability for each target variable to align with the overall missing_rate.
        In the setting that target variables share different missing probability in column. 
        To align with the overall missing_rate, we randomly separate missingness to target variables.

        If self.split is provided, we will use self.missing_rate as the overall missing_rate for each split. Users can set self.same_miss_prob to False 
        to achieve different missing probabilities for each target variable in each split, and the missing probabilities can also be different across splits 
        since we will randomly separate missingness to target variables in each split.

        Returns:
        --------
        col_mean_prob: np.ndarray
            An array of shape (n_target_vars,) containing the mean missing probability for each target variable.
        '''
        # Based on the overall missing rate, calculate a same mean probability for each target variable
        if self.same_miss_prob:
            # Compute mean missing probability for each target variable based on overall missing_rate and number of target variables
            col_mean_prob = self.missing_rate * self.d_dimension / n_target_vars

            # Convert col_mean_prob to an array of shape (n_target_vars,)
            col_mean_prob = np.full(n_target_vars, col_mean_prob)

        # Randomly separate missingness to target variables and compute different mean missing probability for each target variable based on the assigned missing counts.
        else:
            # Compute missingness number in the whole dataset based on overall missing_rate
            total_missing = int(self.missing_rate * n_samples * self.d_dimension)

            # Randomly assign missingness to target variables based on the total missing number and number of target variables
            col_missing_counts = np.random.multinomial(total_missing, [1/n_target_vars]*n_target_vars)
        
            # Compute mean missing probability for each target variable based on the assigned missing counts
            col_mean_prob = col_missing_counts / n_samples

        return col_mean_prob
    
    def normalize_vars(self, vars):
        '''
        Prepare variables for probability modeling:
        1) one-hot encode categorical columns,
        2) normalize only non-binary numeric columns.

        Parameters
        ----------
        vars: pd.DataFrame
            The variables to be normalized.
       
        Returns
        -------
        vars_norm: pd.DataFrame
            Numeric matrix ready for logistic probability modeling.
        '''
        if not isinstance(vars, pd.DataFrame):
            raise ValueError(f"Method normalize_vars must be passed a pandas DataFrame, got {type(vars)}")

        vars_df = vars.copy()

        # 1) One-hot encode categorical columns (including bool)
        self.cat_cols = vars_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if len(self.cat_cols) > 0:
            vars_df = pd.get_dummies(vars_df, columns=self.cat_cols, drop_first=False)

        # Ensure all columns are numeric after encoding
        vars_df = vars_df.apply(pd.to_numeric, errors='raise').astype(float)

        # 2) Normalize only non-binary columns
        vars_norm_df = vars_df.copy()
        for col in vars_norm_df.columns:
            col_values = vars_norm_df[col].dropna().unique()
            is_binary = len(col_values) <= 2 and set(np.round(col_values, 12)).issubset({0.0, 1.0})
            if is_binary:
                continue

            col_std = vars_norm_df[col].std()
            if col_std == 0 or np.isnan(col_std):
                vars_norm_df[col] = 0.0
            else:
                vars_norm_df[col] = (vars_norm_df[col] - vars_norm_df[col].mean()) / col_std

        return vars_norm_df
    
    def adjust_prob(self, deter_vars_norm, n_target_vars, n_samples):
        '''
        Given a missing probability model, adjust probabilities to align with column-wise missing rates, i.e col_mean_prob.
        Only model provided probabilities will be adjusted.

        Parameters
        ----------
        deter_vars_norm: np.ndarray or pd.DataFrame
            Normalized determining variables with shape (n_samples, n_deter_vars). 
        n_target_vars: int
            The number of target variables used to compute col_mean_prob, which will be used to adjust the model probabilities.
        n_samples: int
            The number of samples used to compute col_mean_prob, which will be used to adjust the model probabilities.

        Returns
        -------
        missing_prob_adjusted: np.ndarray
            Adjusted probabilities with shape (n_samples, n_target_vars).
        '''
        # Compute missing probabilities using the model
        if self.model is not None:
            try:
                missing_prob = self.model(deter_vars_norm)
            except Exception as e:
                raise ValueError(f"Provided model cannot be applied to normalized determining variables with shape {np.asarray(deter_vars_norm).shape}. Error: {e}")

            # Make sure missing_prob is a 1D array with length equal to number of samples
            missing_prob = np.asarray(missing_prob, dtype=float)
        else:
            raise ValueError(f"No model provided, cannot compute missing probabilities to adjust.")

        # Validate model output and convert to a 2D matrix (n_samples, n_target_vars)
        if missing_prob.ndim == 1:
            if len(missing_prob) != n_samples:
                raise ValueError(f"Given model output length {len(missing_prob)} doesn't match number of samples {n_samples}")
            base_prob = np.tile(missing_prob.reshape(-1, 1), (1, n_target_vars))
        elif missing_prob.ndim == 2:
            if missing_prob.shape[0] != n_samples:
                raise ValueError(f"Given model output first dimension {missing_prob.shape[0]} doesn't match number of samples {n_samples}")
            if missing_prob.shape[1] == 1:
                base_prob = np.tile(missing_prob, (1, n_target_vars))
            elif missing_prob.shape[1] == n_target_vars:
                base_prob = missing_prob
            else:
                raise ValueError(f"Given model output second dimension must be 1 or n_target_vars={n_target_vars}, got {missing_prob.shape[1]}")
        else:
            raise ValueError(f"Given model must return a 1D array or a 2D array, got shape {missing_prob.shape}")

        # Handle edge case where some target columns have mean probability 0
        mean_prob = base_prob.mean(axis=0)
        if np.any(mean_prob == 0):
            zero_cols = np.where(mean_prob == 0)[0].tolist()
            raise ValueError(f"Given model produces zero mean probabilities for target columns {zero_cols}, cannot adjust to achieve desired missing_rate")

        # Adjust probabilities to align with col-wise missing rates, i.e col_mean_prob.
        col_mean_prob = self.compute_col_mean_prob(n_target_vars = n_target_vars, n_samples = n_samples)
        adjust_factor = col_mean_prob / mean_prob

        # Apply per-target adjustment and clip to valid probability range [0,1]
        missing_prob_adjusted = base_prob * adjust_factor.reshape(1, -1)
        missing_prob_adjusted = np.clip(missing_prob_adjusted, 0, 1)

        return missing_prob_adjusted
    
    def expand_weights(self, deter_vars, weights):
        '''
        Given normalized determining variables and weights, expand weights to match the dimension of the deter_vars_norm if needed. 
        This is used when categorical variables are assigned to be determining variable. In this case, after one-hot encoding, the number of columns for the deter_vars_norm will be larger than the number of weights provided by user.
        
        Parameters
        ----------
        deter_vars_norm: pd.DataFrame
            Normalized determining variables with column names.
        weights: list or np.array
            Coefficients for the logistic model.

        Returns
        -------
        expanded_weights: np.array
            Expanded weights to match the dimension of the deter_vars_norm.
        '''
        # Initialize an empty list to store the expanded weights and a variable to keep track of the current index in the weights list
        expanded_weights = []
        weight_index = 0

        # Get all original column names before on-hot encoding
        original_cols = deter_vars.columns.tolist()

        for col in original_cols:
            if col in self.cat_cols:
                n_categories = deter_vars[col].nunique()
                weight = weights[weight_index]
                expanded_weights.extend([weight] * n_categories)
                weight_index += 1
            else:
                weight = weights[weight_index]
                expanded_weights.append(weight)
                weight_index += 1

        return np.array(expanded_weights, dtype=float)

 
    def _binary_search_bias(self, linear_combination, mean_prob):
        '''
        Given a linear combination, and a target mean probability, use binary search to find the bias coefficient for a logistic model to achieve the desired missing_rate.
        This function returns the best bias coefficient just for one target variable, and will be called in a loop for each target variable in the _binary_search_model function.
        '''
        def objective_function(bias):
            prob_missing = expit(linear_combination + bias)
            return prob_missing.mean() - mean_prob
        
        best_bias = optimize.bisect(objective_function, -100.0, 100.0)

        return best_bias
    
    def _binary_search_prob(self, deter_vars, deter_vars_norm, n_target_vars, n_samples):
        '''
        Use binary search to find the best coefficients for a logistic model to achieve the desired missing_rate.
        If weights are provided, we will start binary search from the provided weights, otherwise we will randomly select weights from the provided weights_range or from a default range.

        Parameters
        ----------
        deter_vars: df pd.DataFrame
            Determining variables with original values and column names.
        deter_vars_norm: pd.DataFrame
            Normalized determining variables with column names.
        n_target_vars: int
            Number of target variables used to compute col_mean_prob.
        n_samples: int
            Number of samples for computing col_mean_prob.

        Returns
        -------
        missing_prob: np.ndarray
            The missing probabilities for each target variable after binary search adjustment, with shape (n_samples, dim_target).
        '''
        # Calculate mean missing probability from the linear combination using the logistic function
        col_mean_prob = self.compute_col_mean_prob(n_target_vars, n_samples)

        # Initialize zero arrays to store missing probabilities and best bias for each target variable 
        missing_prob = np.zeros((n_samples, n_target_vars), dtype=float)
        self.best_bias = np.zeros(n_target_vars, dtype=float)

        # if weights are provided, use weights directly
        if self.weights is not None:
            weights_to_use = np.asarray(self.weights, dtype=float)
            if weights_to_use.ndim == 1:
                if weights_to_use.shape[0] != deter_vars.shape[1]:
                    raise ValueError(f"weights length must match number of determining variables. Got {weights_to_use.shape[0]} and expected {deter_vars_norm.shape[1]}")
                else:
                    if len(self.cat_cols) > 0:
                        weights_to_use = self.expand_weights(deter_vars = deter_vars, weights = weights_to_use)         
                
                # Reuse one weight vector for all target variables
                weights_to_use = np.tile(weights_to_use[:, None], (1, n_target_vars))
            
            elif weights_to_use.ndim == 2:
                if weights_to_use.shape != (deter_vars.shape[1], n_target_vars):
                    raise ValueError(f"weights shape must be ({deter_vars.shape[1]}, {n_target_vars}). Got {weights_to_use.shape}")
                else:
                    if len(self.cat_cols) > 0:
                        expanded_weights_list = []
                        for i in range(weights_to_use.shape[1]):
                            expanded_weights_i = self.expand_weights(deter_vars = deter_vars, weights = weights_to_use[:, i])
                            expanded_weights_list.append(expanded_weights_i)
                        weights_to_use = np.column_stack(expanded_weights_list)
            else:
                raise ValueError(f"weights must be 1D or 2D. Got shape {weights_to_use.shape}")
        
        else:
            # if weights_range is provided, use the provided range to randomly select weights
            if self.weights_range is not None:
                weight_min, weight_max = self.weights_range
            else:
                weight_min, weight_max = -2.0, 2.0
            
            # Randomly select one weight vector per target variable
            weights_to_use = np.random.uniform(weight_min, weight_max, size=(deter_vars.shape[1], n_target_vars))
            
            # Check if there are categorical variables and expand weights if needed
            if len(self.cat_cols) > 0:
                expanded_weights_list = []
                for i in range(weights_to_use.shape[1]):
                    expanded_weights_i = self.expand_weights(deter_vars = deter_vars, weights = weights_to_use[:, i])
                    expanded_weights_list.append(expanded_weights_i)
                weights_to_use = np.column_stack(expanded_weights_list)
    

        # Generate missing probabilities for target variables column by column
        for i in range(n_target_vars):
            linear_combination_i = np.dot(deter_vars_norm, weights_to_use[:, i])
            # Generate best bias using binary search for i-th target variable to align with the overall missing_rate
            best_bias_i = self._binary_search_bias(linear_combination_i, col_mean_prob[i])
            self.best_bias[i] = best_bias_i
            # Compute missing probabilities for the i-th target variable using the best bias
            missing_prob[:, i] = expit(linear_combination_i + best_bias_i)
        
        return missing_prob
            
    def _generate_mask_(self, missing_prob):
        '''
        Given missing probabilities of each target variable, generate mask with same shape.
        Return a mask with shape (n_samples, dim_target) where each entry is a boolean indicating whether the value is missing (True) or observed (False).
        '''
        # Convert missing probabilities into boolean mask (True = missing)
        self.mask = np.random.binomial(1, missing_prob).astype(bool)
        
        return self.mask
    
    def _split_dataset(self, data):
        '''
        Randomly split the given dataset into self.split parts and return a list of split datasets.
        If self.split is None or self.split == 1, return a list containing the original dataset.
        Returns a list of split datasets.
        '''
        if self.split is None or self.split == 1:
            return data
        else:
            # shuffle the dataset indices
            shuffled_indices = np.random.permutation(data.index)

            # split the shuffled indices into self.split parts
            split_indices = np.array_split(shuffled_indices, self.split)
            
            # create split datasets
            split_datasets = [data.loc[indices] for indices in split_indices]

            return split_datasets


    def apply(self, get_statistics = True):
        '''
        Apply mask to target variables and output dataset under MAR assumption.
        
        Returns
        -------
        X_missing: np.ndarray
            Target variables with missing values (NaN), same shapes as target_vars
        '''
        # =============================================================
        # Case 1: deter_vars (or target_vars) and a model are provided.
        # =============================================================
        # If model is provided, and deter_vars (or target_vars) is provided
        if self.model is not None and (self.deter_vars is not None or self.target_vars is not None):
            print("Applying MAR missingness with provided missing probability model and variables...")
             
            # Compute normalized deter_vars
            deter_vars_norm = self.normalize_vars(vars = self.deter_vars)

            # Compute adjusted missing probabilities for each target variable.
            self.missing_prob_adjusted = self.adjust_prob(deter_vars_norm = deter_vars_norm,
                                                          n_target_vars = self.dim_target,
                                                          n_samples = self.n_samples)

            # Generate mask based on the adjusted probabilities.
            self.mask = self._generate_mask_(self.missing_prob_adjusted)

            # Apply mask to target_vars column by column in the dataset and get the final missing dataset
            for i, target_name in enumerate(self.target_names):
                self.dataset.loc[self.mask[:, i], target_name] = np.nan
            

            # Calculate descriptive statistics of the missingness pattern:
            # 1. Final overall missing rate
            self.final_missing_rate = self.dataset.isnull().sum().sum() / (self.n_samples * self.d_dimension)
            # 2. Column-wise missing rates
            self.col_missing_rates = self.dataset.isnull().mean()
            # 3. Convert self.mask to a DataFrame 
            self.mask_df = pd.DataFrame(self.mask, columns=self.target_names)
            # 4. Convert self.missing_prob_adjusted to a DataFrame
            self.missing_prob_adjusted_df = pd.DataFrame(self.missing_prob_adjusted, columns=self.target_names)

            if get_statistics:
                print(f"Missing probabilities of target variables:\n {self.target_names} are generated based on:"
                      f"\n the provided model: {self.model}"
                      f"\ndetermining variables:\n {self.deter_names}.")
                print(f"Final overall missing rate: {self.final_missing_rate:.4f}")
                print(f"Column-wise missing rates:\n{self.col_missing_rates}")
                print(f"Missingness pattern (first 5 rows of mask):\n{self.mask_df.head()}")
                print(f"Adjusted missing probabilities (first 5 rows):\n{self.missing_prob_adjusted_df.head()}")

        # =========================================================
        # Case 2: deter_vars is provided but model is not provided.
        # =========================================================
        # In this case, we will use binary search to find the best coefficients for a logistic model to achieve the desired missing_rate.
        elif self.model is None and (self.deter_vars is not None or self.target_vars is not None):
            print("Applying MAR missingness with binary searched logistic model and provided variables...")

            # Normalize deter_vars
            deter_vars_norm = self.normalize_vars(vars = self.deter_vars)

            # Compute missing probabilities using binary search for a logistic model
            self.missing_prob = self._binary_search_prob(deter_vars=self.deter_vars,
                                                         deter_vars_norm = deter_vars_norm, 
                                                         n_target_vars = self.dim_target, 
                                                         n_samples = self.n_samples)

            # Generate mask based on the computed probabilities.
            self.mask = self._generate_mask_(self.missing_prob)

            # Apply mask to target_vars column by column in the dataset and get the final missing dataset
            for i, target_name in enumerate(self.target_names):
                self.dataset.loc[self.mask[:, i], target_name] = np.nan

            # Calculate descriptive statistics of the missingness pattern:
            # 1. Final overall missing rate
            self.final_missing_rate = self.dataset.isnull().sum().sum() / (self.n_samples * self.d_dimension)
            # 2. Column-wise missing rates
            self.col_missing_rates = self.dataset.isnull().mean()
            # 3. Convert self.mask to a DataFrame
            self.mask_df = pd.DataFrame(self.mask, columns=self.target_names)
            # 4. Convert self.missing_prob to a DataFrame
            self.missing_prob_df = pd.DataFrame(self.missing_prob, columns=self.target_names)
            # 5. Convert self.best_bias to a DataFrame
            self.best_bias_df = pd.DataFrame(self.best_bias, index=self.target_names, columns=['best_bias'])
            # 6. Convert self.random_weights to a DataFrame if random_weights is not None
            if self.random_weights is not None:
                if np.asarray(self.random_weights).ndim == 1:
                    weight_index = self.deter_names if len(self.deter_names) == len(self.random_weights) else [f'deter_{j}' for j in range(len(self.random_weights))]
                    self.random_weights_df = pd.DataFrame(self.random_weights, index=weight_index, columns=['random_weight'])
                else:
                    w = np.asarray(self.random_weights)
                    weight_index = self.deter_names if len(self.deter_names) == w.shape[0] else [f'deter_{j}' for j in range(w.shape[0])]
                    self.random_weights_df = pd.DataFrame(w, index=weight_index, columns=self.target_names)

            if get_statistics:
                if self.weights is not None:
                    print(f"Missing probabilities of target variables:\n {self.target_names} \nare generated based on:"
                          f"provided determining variables:"
                          f"\n{self.deter_names}"
                          f"\nmodel weights:"
                          f"\n{self.weights}")
                elif self.weights_range is not None:
                    print(f"Missing probabilities of target variables:\n {self.target_names} \nare generated based on:"
                          f"provided determining variables:"
                          f"\n{self.deter_names}"
                          f"\nmodel weights (randomly selected from the provided weights_range: {self.weights_range}):"
                          f"\n{self.random_weights}")

                print(f"Best bias for each target variable found by binary search to align with the overall missing_rate:\n{self.best_bias_df}")
                print(f"Missing probabilities of target variables:\n {self.target_names} \nare generated based on binary searched logistic model with determining variables:\n {self.deter_names}.")
                print(f"Final overall missing rate: {self.final_missing_rate:.4f}")
                print(f"Column-wise missing rates:\n{self.col_missing_rates}")
                print(f"Missingness pattern (first 5 rows of mask):\n{self.mask_df.head()}")
                print(f"Missing probabilities from binary search (first 5 rows):\n{self.missing_prob_df.head()}")


        # ====================================================================================
        # Case 3: deter_vars and target_vars are not provided, model and d_deter are provided.
        # ====================================================================================
        # In this case, we will randomly split the dataset into provided number of splits, and randomly choose deter_vars and target_vars in each split.
        elif self.model is not None and (self.deter_vars is None and self.target_vars is None):

            # If user don't want to split the dataset but want to randomly select deter_vars and target_vars, they can set split=1.
            if self.split is None or self.split == 1:
                print("Applying MAR missingness with the given model and selected variables without splitting...")
                
                # Randomly select deter_vars and target_vars for the whole dataset
                deter_names, target_names = self.select_variables(self.dataset)

                # Initialize target_vars and deter_vars based on the selected variable names
                self.deter_vars = self.dataset[deter_names]
                self.target_vars = self.dataset[target_names]
                self.dim_target = self.target_vars.shape[1]

                # Compute normalized deter_vars
                deter_vars_norm = self.normalize_vars(vars = self.deter_vars)

                # Compute adjusted missing probabilities for each target variable.
                self.missing_prob_adjusted = self.adjust_prob(deter_vars_norm = deter_vars_norm,
                                                              n_target_vars = self.dim_target,
                                                              n_samples = self.n_samples)

                # Generate mask based on the adjusted probabilities.
                self.mask = self._generate_mask_(self.missing_prob_adjusted)

                # Apply mask to target_vars column by column in the dataset
                for i, target_name in enumerate(target_names):
                    self.dataset.loc[self.mask[:, i], target_name] = np.nan
                # Calculate descriptive statistics of the missingness pattern:
                # 1. Final overall missing rate
                self.final_missing_rate = self.dataset.isnull().sum().sum() / (self.n_samples * self.d_dimension)
                # 2. Column-wise missing rates
                self.col_missing_rates = self.dataset.isnull().mean()
                # 3. Convert mask to a DataFrame
                self.mask_df = pd.DataFrame(self.mask, columns=target_names)
                # 4. Convert self.missing_prob_adjusted to a DataFrame
                self.missing_prob_adjusted_df = pd.DataFrame(self.missing_prob_adjusted, columns=target_names)

                if get_statistics:
                    print(f"Missing probabilities of randomly selected target variables:\n {target_names}\nare generated based on:"
                          f"\n the provided model: {self.model}"
                          f"\nrandomly selected determining variables:\n {deter_names}.")
                    print(f"Final overall missing rate: {self.final_missing_rate:.4f}")
                    print(f"Column-wise missing rates:\n{self.col_missing_rates}")
                    print(f"Missingness pattern (first 5 rows of mask):\n{self.mask_df.head()}")
                    print(f"Adjusted missing probabilities (first 5 rows):\n{self.missing_prob_adjusted_df.head()}")

            # If user want to split the dataset into provided number of splits, and randomly choose deter_vars and target_vars in each split.
            else:
                print(f"Applying MAR missingness with the given model and randomly selected variables in {self.split} splits...")

                # Split the dataset into self.split parts
                self.split_datasets = self._split_dataset(data = self.dataset)

                # Initialize an empty list to store the processed split datasets and selected variable names for each split
                self.processed_splits = []

                # Process each split dataset separately
                for split_data in self.split_datasets:
                    # Randomly select deter_vars and target_vars for the split dataset
                    deter_names, target_names = self.select_variables(split_data)

                    # Extract deter_vars and target_vars based on the selected variable names for the split dataset
                    deter_vars_split = split_data[deter_names]
                    target_vars_split = split_data[target_names]
                    dim_target_split = target_vars_split.shape[1]

                    # Normalize the selected deter_vars for the split dataset
                    deter_vars_split_norm = self.normalize_vars(vars = deter_vars_split)

                    # Compute adjusted missing probabilities for each target variable in the split dataset.
                    missing_prob_adjusted_split = self.adjust_prob(deter_vars_norm = deter_vars_split_norm,
                                                                   n_target_vars = dim_target_split,
                                                                   n_samples = target_vars_split.shape[0])
                    
                    # Generate mask based on the adjusted probabilities for the split dataset.
                    mask_split = self._generate_mask_(missing_prob_adjusted_split)

                    # Apply mask to target_vars column by column in the split dataset
                    for i, target_name in enumerate(target_names):
                        split_data.loc[mask_split[:, i], target_name] = np.nan

                    # Store the processed split dataset and selected variable names for the split dataset
                    self.processed_splits.append((split_data, deter_names, target_names))
                
                # Concatenate the processed split datasets back to a single dataset
                self.dataset = pd.concat([split_data for split_data, _, _ in self.processed_splits], ignore_index=False)

                # Order the dataset by index to make sure the order is the same as the original dataset
                self.dataset = self.dataset.sort_index()

                # Calculate descriptive statistics of the missingness pattern:
                # 1. Final overall missing rate
                self.final_missing_rate = self.dataset.isnull().sum().sum() / (self.n_samples * self.d_dimension)
                # 2. Column-wise missing rates
                self.col_missing_rates = self.dataset.isnull().mean()

                if get_statistics:
                    print(f"Final overall missing rate: {self.final_missing_rate:.4f}")
                    print(f"Column-wise missing rates:\n{self.col_missing_rates}")
                    print(f"Randomly selected target variables in each split:")
                    for i, (split_data, deter_names, target_names) in enumerate(self.processed_splits):
                        print(f"  Split {i+1}: {target_names}")
                
            
        # ================================================================================
        # Case 4: deter_vars and model are not provided, we will randomly generate missing
        # probabilities and apply missingness to the dataset.
        # ================================================================================
        elif self.model is None and (self.deter_vars is None and self.target_vars is None):
            print("Applying MAR missingness with randomly generated variables using binary search model...")

            # If user don't want to split the dataset but want to randomly generate missingness using binary search with complete dataset.
            if self.split is None or self.split == 1:
                # Randomly select deter_vars and target_vars for the whole dataset
                deter_names, target_names = self.select_variables(self.dataset)

                # Initialize target_vars and deter_vars based on the selected variable names
                self.deter_vars = self.dataset[deter_names]
                self.target_vars = self.dataset[target_names]
                self.dim_target = self.target_vars.shape[1]

                # Compute normalized deter_vars
                deter_vars_norm = self.normalize_vars(vars = self.deter_vars)

                # Compute missing probabilities using binary search for a logistic model
                self.missing_prob = self._binary_search_prob(deter_vars=self.deter_vars,
                                                             deter_vars_norm = deter_vars_norm,
                                                             n_target_vars = self.dim_target,
                                                             n_samples = self.n_samples)
                
                # Generate mask based on the computed probabilities.
                self.mask = self._generate_mask_(self.missing_prob)

                # Apply mask to target_vars column by column in the dataset
                for i, target_name in enumerate(target_names):
                    self.dataset.loc[self.mask[:, i], target_name] = np.nan
                
                # Calculate descriptive statistics of the missingness pattern:
                # 1. Final overall missing rate
                self.final_missing_rate = self.dataset.isnull().sum().sum() / (self.n_samples * self.d_dimension)
                # 2. Column-wise missing rates
                self.col_missing_rates = self.dataset.isnull().mean()
                # 3. Convert mask to a DataFrame
                self.mask_df = pd.DataFrame(self.mask, columns=target_names)
                # 4. Convert self.missing_prob to a DataFrame
                self.missing_prob_df = pd.DataFrame(self.missing_prob, columns=target_names)
                # 5. Convert self.best_bias to a DataFrame
                self.best_bias_df = pd.DataFrame(self.best_bias, index=target_names, columns=['best_bias'])
                # 6. Convert self.random_weights to a DataFrame if random_weights is not None
                if self.random_weights is not None:
                    if np.asarray(self.random_weights).ndim == 1:
                        weight_index = deter_names if len(deter_names) == len(self.random_weights) else [f'deter_{j}' for j in range(len(self.random_weights))]
                        self.random_weights_df = pd.DataFrame(self.random_weights, index=weight_index, columns=['random_weight'])
                    else:
                        w = np.asarray(self.random_weights)
                        weight_index = deter_names if len(deter_names) == w.shape[0] else [f'deter_{j}' for j in range(w.shape[0])]
                        self.random_weights_df = pd.DataFrame(w, index=weight_index, columns=target_names)
                
                if get_statistics:
                    if self.weights is not None:
                        print(f"Missing probabilities of randomly selected target variables:\n{target_names}\nare generated based on:"
                              f"randomly selected determining variables:\n{deter_names}"
                              f"\nprovided model weights:"
                              f"\n{self.weights}")
                    elif self.weights_range is not None:
                        print(f"Missing probabilities of randomly selected target variables:\n {target_names} \nare generated based on:"
                              f"provided determining variables:\n{deter_names}"
                              f"\nmodel weights (randomly selected from the provided weights_range: {self.weights_range}):"
                              f"\n{self.random_weights}")
                    
                    print(f"Final overall missing rate: {self.final_missing_rate:.4f}")
                    print(f"Column-wise missing rates:\n{self.col_missing_rates}")
                    print(f"Missingness pattern (first 5 rows of mask):\n{self.mask_df.head()}")
                    print(f"Missing probabilities from binary search (first 5 rows):\n{self.missing_prob_df.head()}")


            else:
                # Split the dataset into self.split parts
                self.split_datasets = self._split_dataset(data = self.dataset)

                # Initialize an empty list to store the processed split datasets and selected variable names for each split
                processed_splits = []

                # Process each split dataset separately
                for split_data in self.split_datasets:
                    # Randomly select deter_vars and target_vars for the split dataset
                    deter_names, target_names = self.select_variables(split_data)

                    # Extract deter_vars and target_vars based on the selected variable names for the split dataset
                    deter_vars_split = split_data[deter_names]
                    target_vars_split = split_data[target_names]
                    dim_target_split = target_vars_split.shape[1]

                    # Normalize the selected deter_vars for the split dataset
                    deter_vars_split_norm = self.normalize_vars(vars = deter_vars_split)

                    # Compute missing probabilities using binary search for a logistic model for the split dataset
                    missing_prob_split = self._binary_search_prob(deter_vars=deter_vars_split,
                                                                  deter_vars_norm = deter_vars_split_norm,
                                                                  n_target_vars = dim_target_split,
                                                                  n_samples = target_vars_split.shape[0])
                    
                    # Generate mask based on the computed probabilities for the split dataset.
                    mask_split = self._generate_mask_(missing_prob_split)

                    # Apply mask to target_vars column by column in the split dataset
                    for i, target_name in enumerate(target_names):
                        split_data.loc[mask_split[:, i], target_name] = np.nan

                    # Store the processed split dataset and selected variable names for the split dataset
                    processed_splits.append((split_data, deter_names, target_names))
                
                # Concatenate the processed split datasets back to a single dataset
                self.dataset = pd.concat([split_data for split_data, _, _ in processed_splits], ignore_index=False)

                # Order the dataset by index to make sure the order is the same as the original dataset
                self.dataset = self.dataset.sort_index()

                # Calculate descriptive statistics of the missingness pattern:
                # 1. Final overall missing rate
                self.final_missing_rate = self.dataset.isnull().sum().sum() / (self.n_samples * self.d_dimension)
                # 2. Column-wise missing rates
                self.col_missing_rates = self.dataset.isnull().mean()

                if get_statistics:
                    print(f"Final overall missing rate: {self.final_missing_rate:.4f}")
                    print(f"Column-wise missing rates:\n{self.col_missing_rates}")
                    print(f"Randomly selected target variables in each split:")
                    for i, (split_data, deter_names, target_names) in enumerate(processed_splits):
                        print(f"  Split {i+1}: {target_names}")


        return self.dataset