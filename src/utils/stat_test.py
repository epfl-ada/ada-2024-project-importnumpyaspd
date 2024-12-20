from statsmodels.stats import diagnostic
from scipy import stats
import pandas as pd
import numpy as np

def ttest(df1, df2, feature, alpha=0.05):
    """
    Perform an independent t-test on a specified feature between two dataframes, and say it's the test is fine !
    """
    df1_copy=df1.copy()
    df2_copy=df2.copy()
    
    df1_copy = pd.to_numeric(df1_copy[feature], errors='coerce').dropna()
    df2_copy = pd.to_numeric(df2_copy[feature], errors='coerce').dropna()
    # Drop rows with NaN values in the specified feature
    df1_copy = df1_copy.dropna()
    df2_copy = df2_copy.dropna()
    
    # Perform the t-test
    t_stat, p_value = stats.ttest_ind(df1_copy, df2_copy)
    
    # Check if we reject the null hypothesis
    reject_null = p_value < alpha

    if reject_null :
        print(f"The null hypothesis can be rejected, there is a significance difference between the 2 distributioms. (p_value : {p_value})")
    else :
        print(f"The null hypothesis can't be rejected, there is no significance differences. (p_value : {p_value})")
    
    return t_stat, p_value

