"""
Regime Detection using Gaussian Mixture Models
Classifies market into 4 regimes: Growth, Stagnation, Crisis, Transition.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """
    GMM-based market regime detector.
    
    Classifies market states into 4 regimes based on:
    - Returns
    - Realized Volatility
    - VIX normalization  
    - Momentum
    
    Regime Mapping:
    - Growth: High momentum, positive returns, low volatility
    - Stagnation: Low volatility, low returns/momentum
    - Crisis: High volatility, negative returns
    - Transition: Elevated volatility, uncertain direction
    """
    
    def __init__(self, n_components=4, random_state=42):
        self.n_components = n_components
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
            n_init=5  # Multiple initializations for stability
        )
        self.scaler = StandardScaler()
        self.regime_map = {}
        self.is_fitted = False
        
    def fit(self, df, verbose=True):
        """
        Fit the GMM on regime features and create regime mapping.
        
        Args:
            df: DataFrame with columns: returns, realized_vol_20d, vix_norm, momentum_50d
            verbose: Print regime mapping details
        """
        X = df.copy()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        
        # Predict cluster assignments
        preds = self.model.predict(X_scaled)
        df_with_cluster = df.copy()
        df_with_cluster['cluster'] = preds
        
        # Calculate cluster statistics
        stats = df_with_cluster.groupby('cluster').mean()
        
        # Map clusters to regimes based on statistical properties
        self._map_regimes(stats, verbose)
        self.is_fitted = True
        
    def _map_regimes(self, stats, verbose=True):
        """
        Map GMM clusters to regime labels based on characteristics.
        
        Mapping Logic:
        1. Crisis = Highest volatility cluster
        2. Growth = Highest momentum (excluding crisis)
        3. Stagnation = Lowest volatility (excluding crisis, growth)
        4. Transition = Remaining cluster
        """
        vol_col = 'realized_vol_20d'
        mom_col = 'momentum_50d'
        ret_col = 'returns'
        
        remaining = list(stats.index)
        
        # 1. Crisis = Highest volatility
        crisis_id = stats[vol_col].idxmax()
        self.regime_map[crisis_id] = 'Crisis'
        remaining.remove(crisis_id)
        
        # 2. Growth = Highest momentum among remaining
        if remaining:
            growth_id = stats.loc[remaining, mom_col].idxmax()
            self.regime_map[growth_id] = 'Growth'
            remaining.remove(growth_id)
        
        # 3 & 4. Remaining two: Stagnation (lower vol) vs Transition (higher vol)
        if len(remaining) >= 2:
            stag_id = stats.loc[remaining, vol_col].idxmin()
            self.regime_map[stag_id] = 'Stagnation'
            remaining.remove(stag_id)
            
            trans_id = remaining[0]
            self.regime_map[trans_id] = 'Transition'
        elif len(remaining) == 1:
            self.regime_map[remaining[0]] = 'Stagnation'
        
        if verbose:
            print("\n=== REGIME MAPPING ===")
            print("Cluster -> Regime:")
            for cluster_id, regime_name in sorted(self.regime_map.items()):
                print(f"  Cluster {cluster_id} -> {regime_name}")
            print("\nCluster Statistics:")
            print(stats[[ret_col, vol_col, mom_col]].round(4))
            print("=" * 30)
            
    def predict_proba(self, df):
        """
        Get probability distribution over regimes.
        
        Args:
            df: DataFrame with regime features (single row or multiple rows)
            
        Returns:
            Array of shape (n_samples, n_components) with probabilities
        """
        if not self.is_fitted:
            raise ValueError("RegimeDetector must be fitted before predicting")
            
        X_scaled = self.scaler.transform(df)
        return self.model.predict_proba(X_scaled)
    
    def get_current_regime(self, df):
        """
        Get the most likely regime for given features.
        
        Args:
            df: DataFrame with single row of regime features
            
        Returns:
            Tuple of (regime_name, probability)
        """
        probs = self.predict_proba(df)[0]
        max_idx = np.argmax(probs)
        return self.regime_map.get(max_idx, 'Unknown'), probs[max_idx]
    
    def get_agent_weights(self, probs):
        """
        Convert regime probabilities to agent weights.
        
        Mapping:
        - Growth -> Trend Agent
        - Stagnation -> Mean Reversion Agent
        - Crisis -> Defensive Agent
        - Transition -> Recovery Agent
        
        Args:
            probs: Array of probabilities for each cluster
            
        Returns:
            Dictionary with agent weights
        """
        weights = {
            'trend': 0.0,
            'mean_rev': 0.0,
            'defensive': 0.0,
            'recovery': 0.0
        }
        
        for cluster_id in range(self.n_components):
            regime = self.regime_map.get(cluster_id, 'Stagnation')
            prob = probs[cluster_id]
            
            if regime == 'Growth':
                weights['trend'] += prob
            elif regime == 'Stagnation':
                weights['mean_rev'] += prob
            elif regime == 'Crisis':
                weights['defensive'] += prob
            elif regime == 'Transition':
                # Transition: Split between recovery and defensive
                weights['recovery'] += prob * 0.7
                weights['defensive'] += prob * 0.3
        
        return weights
    
    def get_regime_name(self, cluster_id):
        """Get regime name for a cluster ID."""
        return self.regime_map.get(cluster_id, 'Unknown')


def create_regime_features_df(returns, vol, vix_norm, momentum):
    """
    Helper to create a single-row DataFrame for regime prediction.
    
    Args:
        returns: Current return
        vol: Current realized volatility (20d)
        vix_norm: Current normalized VIX
        momentum: Current 50d momentum
        
    Returns:
        DataFrame with single row
    """
    return pd.DataFrame([{
        'returns': returns,
        'realized_vol_20d': vol,
        'vix_norm': vix_norm,
        'momentum_50d': momentum
    }])
