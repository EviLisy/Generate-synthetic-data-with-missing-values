import numpy as np
import pandas as pd
from scipy.special import expit
import warnings

class MAR:
    def __init__(self,
                 dataset,
                 target_vars,
                 deter_vars,
                 model,
                 missing_rate,
                 seed=None):
        '''
        Generate Missing At Random (MAR) missingness mechanism.

        Parameters
        ----------
        dataset: pd.Dataframe
            complete dataset which determines the shape of final output.
        target_vars: pd.Series or pd.DataFrame
            Variable to introduce missingness into. (e.g., df['X1] or df['X1','X2])
        deter_vars: pd.Series or pd.DataFrame
            Determining features that are fully observed columns driving missingness
        model: callable
            Function to compute P(missing at target_vars | deter_vars) = f(deter_vars)
            Should accept standardized deter_vars and return probabilities
        missing_rate: float
            Desired proportion of missing values.
        seed: int, optional
            Random seed for reproducibility.
        '''
        # Validate missing_rate
        if not 0 <= missing_rate <= 1:
            raise ValueError(f"missing_rate must be in [0,1], got {missing_rate}")
        
        # Convert to numpy arrays and validate shapes
        self.target_vars, self.target_names = self._validate_and_convert(target_vars, "target_vars")
        self.deter_vars, self.deter_names = self._validate_and_convert(deter_vars, "deter_vars")

        # Ensure same number of samples
        if len(self.target_vars) != len(self.deter_vars):
            raise ValueError(
                f"target_vars and deter_vars must have same number of samples."
                f"Got {len(self.target_vars)} and {len(self.deter_vars)}"
            )
        
        self.dataset = dataset.copy()
        self.model = model
        self.missing_rate = missing_rate
        self.seed = seed
        self.missing_prob_adjusted = None   # will be computed in generate_mask
        self.n_samples = len(self.target_vars)

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

    def _validate_and_convert(self, data, name):
        '''Convert input to numpy array and validate.'''

        names = []

        if isinstance(data, pd.Series):
            names = [data.name if data.name is not None else 'col_0']
            data = data.values
        elif isinstance(data, pd.DataFrame):
            names = data.colums.tolist()
            data = data.values
        elif isinstance(data, np.ndarray):
            # If it's a raw array, we generate generic names
            values = data
            if values.ndim == 1:
                names = ['col_0']
            else:
                names = [f"col_{i}" for i in range(values.shape[1])]
        else:
            raise TypeError(
                f"{name} must be numpy array, pandas Series, or Dataframe"
            )
        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            raise ValueError(f"{name} must be 1D or 2D, got {data.ndim}D")
        
        return data, names
    

    def generate_mask(self, return_prob=False):
        '''
        Generate mask and missingness probability for each observation in target_vars.
        
        return_prob: bool, default=False
            If True, also return the missing probabilities 
        
        Returns
        -------
        self.mask: np.array
            Boolean mask indicating missing positions (True = missing)
        self.missing_prob_adjusted: np.ndarray
            Probability of missingness for each sample adjusted by missing rate, shape(n_samples,) (only if return_prob=True)
        '''
        # Handle case where deter_vars has zero variance
        deter_vars_std = self.deter_vars.std(axis=0)
        deter_vars_mean = self.deter_vars.mean(axis=0)

        # Avoid division by zero
        deter_vars_std[deter_vars_std == 0] =1.0

        # Normalize deter_vars: (X - mean) / std
        deter_vars_norm = (self.deter_vars - deter_vars_mean) / deter_vars_std

        # Generate missing probability according to model
        # Model should return shape (n_samples,)
        missing_prob = self.model(deter_vars_norm)

        # Validate model output
        if missing_prob.ndim > 1:
            if missing_prob.shape[1] == 1:
                missing_prob = missing_prob.ravel()
            else:
                raise ValueError(
                    f"model must return 1D array, got shape {missing_prob.shape}"
                )
            
        if len(missing_prob) != self.n_samples:
            raise ValueError(
                f"model output length {len(missing_prob)} doesn't match number of samples {self.n_samples}"
            )

        # Adjust probabilities to align with overall missing_rate
        # Handle edge case where all probabilities in [0,1]
        if missing_prob.mean() == 0:
            # If model produces all zeros, use uniform probabilities
            missing_prob = np.ones(self.n_samples) / self.n_samples

        self.missing_prob_adjusted = missing_prob * (self.missing_rate / missing_prob.mean())

        # Clip to valid probability range [0,1]
        self.missing_prob_adjusted = np.clip(self.missing_prob_adjusted, 0, 1)

        # Warn if we couldn't achieve exact missing rate
        achievable_rate = self.missing_prob_adjusted.mean()
        if abs(achievable_rate - self.missing_rate) > 0.05:
            import warnings
            warnings.warn(
                f"Could not achieve exact missing_rate={self.missing_rate:.3f}."
                f"Achievable rate is {achievable_rate:.3f}."
                f"Consider using a different model function."
            )
        
        # Convert missing probabilities into boolean mask (True = missing)
        self.mask = np.random.binomial(1, self.missing_prob_adjusted, size=self.n_samples).astype(bool)

        if return_prob:
            return self.mask, self.missing_prob_adjusted
        else:
            return self.mask

    def apply(self):
        '''
        Apply missingness to target variables and output dataset under MAR assumption.
        
        Returns
        -------
        X_missing: np.ndarray
            Target variables with missing values (NaN), same shapes as target_vars
        
        '''
        # Generate probabilities
        mask = self.generate_mask(return_prob=False)

        # Apply mask to target_vars in the dataset
        self.dataset.loc[mask, self.target_names] = np.nan

        return self.dataset
        
    def get_statistics(self):
        '''
        Get statistics about the missingness mechanism.

        Returns
        -------
        stats: dict
            Dictionary containing:
            - n_samples: number of samples
            - missing_rate_target: target missing rate
            - missing_rate_achievable: achievable missing rate after clipping
            - prob_min: minimum missing probability
            - prob_max: maximum missing probability
            - prob_mean: mean missing probability
        '''
        if self.missing_prob_adjusted is None:
            self.generate_mask()

        return {
            'n_samples': self.n_samples,
            'n_target_vars': self.target_vars.shape[1],
            'n_deter_vars': self.deter_vars.shape[1],
            'missing_rate_target': self.missing_rate,
            'missing_rate_achievable': self.missing_prob_adjusted.mean(),
            'prob_min': self.missing_prob_adjusted.min(),
            'prob_max': self.missing_prob_adjusted.max(),
            'prob_mean': self.missing_prob_adjusted.mean(),
            'prob_std': self.missing_prob_adjusted.std(),
        }
    
# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from scipy.special import expit
    
    print("="*70)
    print("MAR Missingness Generation - Example Usage")
    print("="*70)
    
    # ========================================
    # Example 1: Single determining variable
    # ========================================
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Determining Variable (Age → Income Missing)")
    print("="*70)
    
    # Generate complete dataset
    np.random.seed(42)
    n = 1000
    
    age = np.random.normal(40, 15, n)
    age = np.clip(age, 18, 80)
    income = 30000 + 800 * age + np.random.normal(0, 15000, n)
    
    df = pd.DataFrame({
        'age': age,
        'income': income
    })
    
    print("\nOriginal dataset (first 10 rows):")
    print(df.head(10))
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    
    # Define model: younger people more likely to have missing income
    def model_age(age_normalized):
        """
        P(Income missing | Age)
        Younger people (lower age) → higher missing probability
        
        Parameters
        ----------
        age_normalized : np.ndarray
            Standardized age values (mean=0, std=1)
        """
        # Negative coefficient: lower age → higher probability
        return expit(-0.5 - 0.8 * age_normalized.ravel())
    
    # Create MAR instance
    mar_income = MAR(
        dataset=df,
        target_vars=df['income'],
        deter_vars=df['age'],
        model=model_age,
        missing_rate=0.30,
        seed=42
    )
    
    # Apply missingness
    df_missing = mar_income.apply()
    mask_income, missing_prob_income = mar_income.generate_mask(return_prob=True)
    
    # Get statistics
    stats = mar_income.get_statistics()
    print("\nMAR Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nDataset with MAR missingness (first 15 rows):")
    print(df_missing.head(15))
    
    print(f"\nActual missing rate: {df_missing['income'].isna().sum()/(df_missing.shape[0] * df_missing.shape[1])}")
    print(f"Mean age (income observed): {df_missing['age'][~mask_income].mean():.2f}")
    print(f"Mean age (income missing): {df_missing['age'][mask_income].mean():.2f}")