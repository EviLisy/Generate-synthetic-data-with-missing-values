import numpy as np
import pandas as pd
from scipy.special import expit
import warnings

class MAR:
    def __init__(self,
                 target_vars,
                 deter_vars,
                 model,
                 missing_rate,
                 seed=None):
        '''
        Generate Missing At Random (MAR) missingness mechanism.

        Parameters
        ----------
        target_vars: np.ndarray or pd.Series
            Variable to introduce missingness into.
        deter_vars: list[int]
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
        self.target_vars = self._validate_and_convert(target_vars, "target_vars")
        self.deter_vars = self._validate_and_convert(deter_vars, "deter_vars")

        # Ensure same number of samples
        if len(self.target_vars) != len(self.deter_vars):
            raise ValueError(
                f"target_vars and deter_vars must have same number of samples."
                f"Got {len(self.target_vars)} and {len(self.deter_vars)}"
            )

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
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, np.ndarray):
            pass
        else:
            raise TypeError(
                f"{name} must be numpy array, pandas Series, or Dataframe"
            )
        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            raise ValueError(f"{name} must be 1D or 2D, got {data.ndim}D")
        
        return data
    

    def generate_mask(self):
        '''
        Generate missingness probability for each observation in target_vars.
        
        Returns
        -------
        missing_prob_adjusted: np.ndarray
            Probability of missingness for each sample, shape(n_samples,)
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

        return self.missing_prob_adjusted

    def apply(self, return_mask=False):
        '''
        Apply missingness to target variables.

        Parameters
        ----------
        return_mask: bool, default=False
            If True, also return the boolean mask (True = missing)

        Returns
        -------
        X_missing: np.ndarray
            Target variables with missing values (NaN), same shapes as target_vars
        mask: np.array, optional
            Boolean mask indicating missing positions (only if return_mask=True)
        '''
        # Generate probabilities
        missing_prob = self.generate_mask()

        # Convert probabilities to binary mask
        # mask=True means missing
        mask = np.random.binomial(1, missing_prob, size=self.n_samples).astype(bool)

        # Apply mask to target_vars
        X_missing = self.target_vars.astype(float).copy()

        # Handle multi-column target_vars
        if X_missing.shape[1] == 1:
            # Single target variable
            X_missing[mask, 0] = np.nan
        else:
            # Multiple target variables - apply same mask to all
            X_missing[mask, :] = np.nan

        # Squeeze if single column
        if X_missing.shape[1] == 1:
            X_missing = X_missing.ravel()
        
        if return_mask:
            return X_missing, mask
        else:
            return X_missing
        
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
        target_vars=df['income'],
        deter_vars=df['age'],
        model=model_age,
        missing_rate=0.30,
        seed=42
    )
    
    # Apply missingness
    income_missing, mask = mar_income.apply(return_mask=True)
    
    # Get statistics
    stats = mar_income.get_statistics()
    print("\nMAR Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Add to dataframe
    df['income_mar'] = income_missing
    
    print(f"\nDataset with MAR missingness (first 15 rows):")
    print(df.head(15))
    
    print(f"\nActual missing rate: {df['income_mar'].isna().mean():.3f}")
    print(f"Mean age (income observed): {df[df['income_mar'].notna()]['age'].mean():.2f}")
    print(f"Mean age (income missing): {df[df['income_mar'].isna()]['age'].mean():.2f}")
    
    # ========================================
    # Example 2: Multiple determining variables
    # ========================================
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Determining Variables (Age + Education → Income)")
    print("="*70)
    
    # Add education variable
    education = np.random.choice([1, 2, 3, 4], size=n, p=[0.3, 0.4, 0.2, 0.1])
    df['education'] = education
    
    # Regenerate income based on both age and education
    df['income'] = 20000 + 10000 * education + 500 * age + np.random.normal(0, 10000, n)
    
    # Model with multiple predictors
    def model_multi(X_norm):
        """
        P(Income missing | Age, Education)
        Lower age AND lower education → higher missing probability
        
        Parameters
        ----------
        X_norm : np.ndarray, shape (n_samples, 2)
            Standardized [age, education]
        """
        return expit(-0.3 - 0.6 * X_norm[:, 0] - 0.4 * X_norm[:, 1])
    
    # Stack age and education
    deter_multi = np.column_stack([age, education])
    
    mar_multi = MAR(
        target_vars=df['income'],
        deter_vars=deter_multi,
        model=model_multi,
        missing_rate=0.25,
        seed=42
    )
    
    income_missing_multi = mar_multi.apply()
    df['income_mar_multi'] = income_missing_multi
    
    stats_multi = mar_multi.get_statistics()
    print("\nMulti-variable MAR Statistics:")
    for key, value in stats_multi.items():
        print(f"  {key}: {value}")
    
    print(f"\nMissing rate by education level:")
    for edu_level in [1, 2, 3, 4]:
        mask_edu = df['education'] == edu_level
        miss_rate = df.loc[mask_edu, 'income_mar_multi'].isna().mean()
        mean_age = df.loc[mask_edu, 'age'].mean()
        print(f"  Education {edu_level} (mean age={mean_age:.1f}): {miss_rate:.1%} missing")
    
    # ========================================
    # Example 3: Using train_test_split pattern
    # ========================================
    print("\n" + "="*70)
    print("EXAMPLE 3: Split-Apply-Combine Pattern with train_test_split")
    print("="*70)
    
    from sklearn.model_selection import train_test_split
    
    # Create fresh dataset
    df_complete = pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'income': np.random.normal(50000, 20000, n)
    })
    
    print("Original complete dataset:")
    print(df_complete.head(10))
    
    # Split into two groups
    group1, group2 = train_test_split(df_complete, test_size=0.5, random_state=42)
    
    print(f"\nGroup 1 size: {len(group1)}")
    print(f"Group 2 size: {len(group2)}")
    
    # Define models
    def model_income_age(age_std):
        return expit(0 + 1 * age_std.ravel())
    
    def model_age_income(income_std):
        return expit(0 - 1 * income_std.ravel())
    
    # Group 1: Mask income based on age
    mar1 = MAR(
        target_vars=group1['income'],
        deter_vars=group1['age'],
        model=model_income_age,
        missing_rate=0.30,
        seed=42
    )
    group1['income'] = mar1.apply()
    
    # Group 2: Mask age based on income
    mar2 = MAR(
        target_vars=group2['age'].astype(float),  # Convert to float
        deter_vars=group2['income'],
        model=model_age_income,
        missing_rate=0.30,
        seed=42
    )
    group2['age'] = mar2.apply()
    
    # Combine back and sort by index
    df_mar = pd.concat([group1, group2], axis=0).sort_index()
    
    print("\nCombined dataset with MAR missingness:")
    print(df_mar.head(15))
    
    print(f"\nMissing pattern:")
    both_obs = (~df_mar['age'].isna() & ~df_mar['income'].isna()).sum()
    age_miss = (df_mar['age'].isna() & ~df_mar['income'].isna()).sum()
    income_miss = (~df_mar['age'].isna() & df_mar['income'].isna()).sum()
    both_miss = (df_mar['age'].isna() & df_mar['income'].isna()).sum()
    
    print(f"  Both observed: {both_obs} ({both_obs/len(df_mar):.1%})")
    print(f"  Only age missing: {age_miss} ({age_miss/len(df_mar):.1%})")
    print(f"  Only income missing: {income_miss} ({income_miss/len(df_mar):.1%})")
    print(f"  Both missing: {both_miss} ({both_miss/len(df_mar):.1%})")
    
    # ========================================
    # Example 4: Multiple target variables
    # ========================================
    print("\n" + "="*70)
    print("EXAMPLE 4: Multiple Target Variables (same mask for all)")
    print("="*70)
    
    # Create dataset with multiple targets
    savings = 5000 + 0.1 * df['income'] + np.random.normal(0, 3000, n)
    targets_multi = np.column_stack([df['income'].values, savings])
    
    mar_multi_target = MAR(
        target_vars=targets_multi,
        deter_vars=df['age'],
        model=model_age,
        missing_rate=0.20,
        seed=42
    )
    
    targets_missing = mar_multi_target.apply()
    
    print(f"Shape of output: {targets_missing.shape}")
    print(f"Income missing rate: {np.isnan(targets_missing[:, 0]).mean():.3f}")
    print(f"Savings missing rate: {np.isnan(targets_missing[:, 1]).mean():.3f}")
    print("\nNote: Same mask is applied to both target variables")
    
    print("\n" + "="*70)
    print("Examples completed successfully!")
    print("="*70)