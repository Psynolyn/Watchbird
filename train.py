"""
Bird Movement Anomaly Detection using Isolation Forest
=======================================================

RATIONALE:
----------
Isolation Forest is particularly well-suited for bird movement anomaly detection because:

1. **Unsupervised Nature**: Bird behavior data rarely comes with labeled anomalies. 
   Isolation Forest doesn't require labels and works by isolating outliers that are 
   "few and different" - exactly what we need for detecting unusual behaviors like 
   sudden altitude drops, unexpected stops, or erratic speed patterns.

2. **Efficiency with High-Dimensional Data**: With multiple engineered features 
   (speed statistics, acceleration patterns, vertical movement), IF scales well 
   (O(n log n)) and handles feature interactions without explicit specification.

3. **Robustness to Normal Variation**: Unlike distance-based methods, IF doesn't 
   assume spherical clusters. Birds naturally exhibit varied behaviors (soaring, 
   flapping flight, gliding), and IF adapts by focusing on isolation rather than 
   density, making it robust to multi-modal normal behavior.

WHY EXCLUDE BEARING:
-------------------
Raw bearing (compass direction) is excluded because:
- Birds frequently change direction during normal flight (circling thermals, 
  searching for food, course corrections)
- Bearing is circular (0° = 360°), requiring special handling
- Sudden bearing changes don't necessarily indicate anomalies
Instead, we use bearing-invariant features: speed magnitude, acceleration, 
turn rate derived from speed changes, and movement statistics.

BEHAVIORAL FEATURES:
-------------------
Our engineered features capture:
- **Hovering**: Low speed sustained over time (possible GPS errors or unusual stops)
- **Burst Movement**: High acceleration or speed changes (escape responses, prey pursuit)
- **Altitude Dives**: Rapid vertical speed changes (potential distress or hunting)
- **Migration Patterns**: Rolling statistics capture sustained vs erratic movement
- **Movement Consistency**: Standard deviation features detect behavioral volatility

"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import joblib
from datetime import datetime, timedelta
import db_operations

# Core ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class Logger:
    """Centralized logging system that stores messages in a list."""
    
    _logs: List[str] = []
    _initialized = False
    
    @classmethod
    def reset(cls):
        """Reset logs at the start of training."""
        cls._logs = []
        cls._initialized = True
    
    @classmethod
    def add(cls, message: str):
        """Add a message to the log."""
        if not cls._initialized:
            cls.reset()
        cls._logs.append(message)
    
    @classmethod
    def get_logs(cls) -> List[str]:
        """Get all logged messages."""
        return cls._logs.copy()
    
    @classmethod
    def get_logs_string(cls) -> str:
        """Get all logged messages as a single string."""
        return '\n'.join(cls._logs)
    
    @classmethod
    def print_logs(cls):
        """Print all logs to console."""
        for log in cls._logs:
            print(log)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the anomaly detection pipeline."""
    
    # File paths
    DATA_PATH = 'dataset/derived.csv'
    MODEL_SAVE_PATH = 'models/isolation_forest_model.pkl'
    SCALER_SAVE_PATH = 'models/scaler.pkl'
    RESULTS_PATH = 'results/anomalies.csv'
    
    # Feature engineering
    ROLLING_WINDOWS = [3, 5, 9]  # Multiple window sizes for robustness
    HOVER_SPEED_THRESHOLD = 0.5  # m/s - below this is hovering
    HOVER_DURATION_THRESHOLD = 600  # seconds - sustained hover time
    BURST_ACCELERATION_THRESHOLD = 2.0  # m/s² - sudden acceleration
    BURST_SPEED_CHANGE_THRESHOLD = 1.5  # m/s² - rapid speed change
    
    # GPS quality filters
    MIN_LATITUDE = -90
    MAX_LATITUDE = 90
    MIN_LONGITUDE = -180
    MAX_LONGITUDE = 180
    MAX_SPEED = 40  # m/s - ~144 km/h, reasonable max for most birds
    MIN_DT = 1  # seconds - minimum time between fixes
    
    # Model parameters
    N_ESTIMATORS = 200
    MAX_SAMPLES = 'auto'
    CONTAMINATION = 0.1  # Expected proportion of anomalies (5%)
    RANDOM_STATE = 42
    N_JOBS = -1
    
    # Training parameters
    TEST_SIZE_RATIO = 0.2  # Last 20% of data for testing
    MIN_SAMPLES_FOR_TRAINING = 100
    
    # Anomaly detection
    ANOMALY_THRESHOLD_QUANTILE = 0.05  # Bottom 5% of scores
    EVENT_GAP_THRESHOLD = 900  # seconds - gap to separate anomaly events
    MIN_EVENT_SIZE = 2  # minimum points to consider an event (filters isolated anomalies)
    SLIDING_WINDOW_SIZE = 10  # points for online detection
    ANOMALY_CLUSTER_THRESHOLD = 3  # min anomalies in window to trigger alert
    
    # Retraining
    RETRAIN_PERIOD_DAYS = 7
    MIN_NEW_SAMPLES_FOR_RETRAIN = 500


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(device_id: Optional[int] = None,
              device_ids: Optional[List[int]] = None,
              path: str = Config.DATA_PATH,
              validate: bool = True) -> pd.DataFrame:
    """
    Load bird movement data from CSV with optional device filtering.
    
    Parameters:
    -----------
    device_id : int, optional
        Specific device to load. If None, loads all devices.
    device_ids : list of int, optional
        List of device IDs to load. Takes precedence over device_id.
    path : str
        Path to the CSV file
    validate : bool
        Whether to perform data quality validation
        
    Returns:
    --------
    pd.DataFrame
        Loaded and preprocessed data
    """
    Logger.add(f"Loading data from {path}...")
    
    # Load with timezone-aware timestamp parsing
    df = pd.read_csv(path)
    
    # Parse timestamp as timezone-aware
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    
    # Filter by device(s) if specified
    if device_ids is not None:
        df = df[df['Device_id'].isin(device_ids)].copy()
        Logger.add(f"Filtered to Device_ids={device_ids}: {len(df)} records")
        Logger.add(f"  Records per device: {df.groupby('Device_id').size().to_dict()}")
    elif device_id is not None:
        df = df[df['Device_id'] == device_id].copy()
        Logger.add(f"Filtered to Device_id={device_id}: {len(df)} records")
    else:
        Logger.add(f"Loaded all devices: {len(df)} records")
        Logger.add(f"  Unique devices: {df['Device_id'].nunique()}")
    
    if len(df) == 0:
        filter_desc = f"device_ids={device_ids}" if device_ids else f"device_id={device_id}"
        raise ValueError(f"No data found for {filter_desc}")
    
    # Sort by timestamp (critical for time-series features)
    df = df.sort_values(['Device_id', 'Timestamp']).reset_index(drop=True)
    
    if validate:
        df = validate_and_clean_data(df)
    
    return df


def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate GPS data quality and clean obvious errors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    initial_count = len(df)
    Logger.add(f"\nData validation starting with {initial_count} records...")
    
    # Remove records with null coordinates
    df = df.dropna(subset=['Latitude', 'Longitude', 'Altitude'])
    Logger.add(f"  Removed {initial_count - len(df)} records with null coordinates")
    
    # Remove invalid GPS coordinates
    df = df[
        (df['Latitude'] >= Config.MIN_LATITUDE) & 
        (df['Latitude'] <= Config.MAX_LATITUDE) &
        (df['Longitude'] >= Config.MIN_LONGITUDE) & 
        (df['Longitude'] <= Config.MAX_LONGITUDE)
    ]
    Logger.add(f"  Removed records with invalid GPS bounds: {initial_count - len(df)} total removed")
    
    # Compute dt if missing or zero
    if 'dt' not in df.columns or df['dt'].isna().any():
        df = compute_time_deltas(df)
    
    # Remove records with impossible dt values
    initial_count = len(df)
    df = df[df['dt'] >= Config.MIN_DT]
    if initial_count - len(df) > 0:
        Logger.add(f"  Removed {initial_count - len(df)} records with dt < {Config.MIN_DT}s")
    
    # Recompute speed if missing, or validate existing
    df = compute_or_validate_speed(df)
    
    # Remove impossible speeds
    initial_count = len(df)
    df = df[df['speed_m_s'] <= Config.MAX_SPEED]
    if initial_count - len(df) > 0:
        Logger.add(f"  Removed {initial_count - len(df)} records with speed > {Config.MAX_SPEED} m/s")
    
    Logger.add(f"Data validation complete: {len(df)} valid records\n")
    
    return df.reset_index(drop=True)


def compute_time_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute time differences between consecutive records."""
    df = df.copy()
    df['dt'] = df.groupby('Device_id')['Timestamp'].diff().dt.total_seconds()
    df['dt'] = df['dt'].fillna(0)
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS coordinates using Haversine formula.
    
    Returns distance in meters.
    """
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def compute_or_validate_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Compute or validate speed_m_s from GPS coordinates."""
    df = df.copy()
    
    # Compute distance if not present
    if 'distance_m' not in df.columns or df['distance_m'].isna().any():
        df['delta_lat'] = df.groupby('Device_id')['Latitude'].diff()
        df['delta_lon'] = df.groupby('Device_id')['Longitude'].diff()
        
        df['distance_m'] = haversine_distance(
            df['Latitude'].shift(1), 
            df['Longitude'].shift(1),
            df['Latitude'], 
            df['Longitude']
        )
        df['distance_m'] = df['distance_m'].fillna(0)
    
    # Compute speed
    df['speed_m_s'] = np.where(
        df['dt'] > 0,
        df['distance_m'] / df['dt'],
        0
    )
    
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_features(df: pd.DataFrame, 
                   rolling_windows: List[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Engineer features for anomaly detection, avoiding raw bearing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe with basic movement metrics
    rolling_windows : list of int
        Window sizes for rolling statistics
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with engineered features
    feature_columns : list of str
        Names of feature columns for modeling
    """
    Logger.add("Engineering features...")
    
    if rolling_windows is None:
        rolling_windows = Config.ROLLING_WINDOWS
    
    df = df.copy()
    feature_columns = []
    
    # === Basic movement features ===
    
    # Vertical speed (m/s)
    df['vertical_speed'] = np.where(
        df['dt'] > 0,
        df['delta_alt'] / df['dt'],
        0
    )
    feature_columns.extend(['dt', 'distance_m', 'speed_m_s', 'vertical_speed'])
    
    # Absolute acceleration
    if 'acceleration' in df.columns:
        df['acceleration_abs'] = df['acceleration'].abs()
        feature_columns.append('acceleration_abs')
    
    # Speed change (jerk approximation)
    df['speed_change'] = df.groupby('Device_id')['speed_m_s'].diff()
    df['speed_change_rate'] = np.where(
        df['dt'] > 0,
        df['speed_change'] / df['dt'],
        0
    )
    df['speed_change_abs'] = df['speed_change_rate'].abs()
    feature_columns.append('speed_change_abs')
    
    # Altitude change magnitude
    df['alt_change_abs'] = df['delta_alt'].abs()
    feature_columns.append('alt_change_abs')
    
    # === Rolling window statistics ===
    
    for window in rolling_windows:
        # Speed statistics
        df[f'speed_mean_{window}'] = df.groupby('Device_id')['speed_m_s'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'speed_std_{window}'] = df.groupby('Device_id')['speed_m_s'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        ).fillna(0)
        
        # Acceleration statistics
        if 'acceleration' in df.columns:
            df[f'acc_mean_{window}'] = df.groupby('Device_id')['acceleration'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'acc_std_{window}'] = df.groupby('Device_id')['acceleration'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            ).fillna(0)
            feature_columns.extend([f'acc_mean_{window}', f'acc_std_{window}'])
        
        # Vertical speed statistics
        df[f'vert_speed_mean_{window}'] = df.groupby('Device_id')['vertical_speed'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'vert_speed_std_{window}'] = df.groupby('Device_id')['vertical_speed'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        ).fillna(0)
        
        # Distance sum (total movement in window)
        df[f'distance_sum_{window}'] = df.groupby('Device_id')['distance_m'].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
        
        feature_columns.extend([
            f'speed_mean_{window}', f'speed_std_{window}',
            f'vert_speed_mean_{window}', f'vert_speed_std_{window}',
            f'distance_sum_{window}'
        ])
    
    # === Z-score of speed (deviation from recent behavior) ===
    
    # Compute rolling mean and std for z-score calculation
    df['speed_rolling_mean'] = df.groupby('Device_id')['speed_m_s'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )
    df['speed_rolling_std'] = df.groupby('Device_id')['speed_m_s'].transform(
        lambda x: x.rolling(10, min_periods=1).std()
    )
    
    # Calculate z-score
    df['speed_zscore'] = (df['speed_m_s'] - df['speed_rolling_mean']) / (df['speed_rolling_std'] + 1e-6)
    df['speed_zscore'] = df['speed_zscore'].fillna(0)
    
    # Drop intermediate columns
    df.drop(['speed_rolling_mean', 'speed_rolling_std'], axis=1, inplace=True)
    
    feature_columns.append('speed_zscore')
    
    # === Behavioral flags ===
    
    # Hover detection (low speed sustained)
    df['is_hovering'] = (df['speed_m_s'] < Config.HOVER_SPEED_THRESHOLD).astype(int)
    df['hover_duration'] = df.groupby('Device_id')['is_hovering'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    ) * df['dt']
    df['hover_flag'] = (df['hover_duration'] > Config.HOVER_DURATION_THRESHOLD).astype(int)
    feature_columns.append('hover_flag')
    
    # Burst movement detection
    df['burst_flag'] = (
        (df['acceleration_abs'] > Config.BURST_ACCELERATION_THRESHOLD) |
        (df['speed_change_abs'] > Config.BURST_SPEED_CHANGE_THRESHOLD)
    ).astype(int)
    feature_columns.append('burst_flag')
    
    # === Temporal features (optional) ===
    
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    # Normalize to [0, 1]
    df['hour_norm'] = df['hour'] / 24.0
    df['day_of_week_norm'] = df['day_of_week'] / 7.0
    feature_columns.extend(['hour_norm', 'day_of_week_norm'])
    
    # === Log transforms for heavy-tailed features ===
    
    df['log_speed'] = np.log1p(df['speed_m_s'])
    df['log_distance'] = np.log1p(df['distance_m'])
    feature_columns.extend(['log_speed', 'log_distance'])
    
    # Fill any remaining NaNs
    df[feature_columns] = df[feature_columns].fillna(0)
    
    # Replace infinities
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], 0)
    
    Logger.add(f"Engineered {len(feature_columns)} features")
    
    return df, feature_columns


# ============================================================================
# MODEL TRAINING
# ============================================================================

def split_train_test(df: pd.DataFrame, 
                     test_ratio: float = Config.TEST_SIZE_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-aware train/test split (no shuffling).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data sorted by timestamp
    test_ratio : float
        Proportion of data for testing (latest records)
        
    Returns:
    --------
    train_df, test_df : tuple of DataFrames
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    Logger.add(f"\nTrain/Test split:")
    Logger.add(f"  Training: {len(train_df)} records ({train_df['Timestamp'].min()} to {train_df['Timestamp'].max()})")
    Logger.add(f"  Testing:  {len(test_df)} records ({test_df['Timestamp'].min()} to {test_df['Timestamp'].max()})")
    
    return train_df, test_df


def train_isolation_forest(X_train: np.ndarray,
                           contamination: float = Config.CONTAMINATION,
                           n_estimators: int = Config.N_ESTIMATORS,
                           max_samples: str = Config.MAX_SAMPLES,
                           random_state: int = Config.RANDOM_STATE,
                           verbose: bool = True) -> IsolationForest:
    """
    Train Isolation Forest model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix
    contamination : float
        Expected proportion of anomalies
    n_estimators : int
        Number of trees
    max_samples : str or int
        Samples per tree
    random_state : int
        Random seed
    verbose : bool
        Log training info
        
    Returns:
    --------
    IsolationForest
        Trained model
    """
    if verbose:
        Logger.add(f"\nTraining Isolation Forest...")
        Logger.add(f"  Samples: {X_train.shape[0]}")
        Logger.add(f"  Features: {X_train.shape[1]}")
        Logger.add(f"  Contamination: {contamination}")
        Logger.add(f"  n_estimators: {n_estimators}")
    
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=Config.N_JOBS,
        verbose=0
    )
    
    model.fit(X_train)
    
    if verbose:
        Logger.add(f"  Training complete!")
    
    return model


def hyperparameter_search(X_train: np.ndarray,
                          param_grid: Dict = None,
                          n_splits: int = 3,
                          verbose: bool = True) -> Dict:
    """
    Perform time-aware hyperparameter search for Isolation Forest.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training data
    param_grid : dict
        Parameter grid to search
    n_splits : int
        Number of CV splits
    verbose : bool
        Log progress
        
    Returns:
    --------
    dict
        Best parameters
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_samples': ['auto', 256, 512],
            'contamination': [0.01, 0.05, 0.1]
        }
    
    Logger.add(f"\nHyperparameter search with {n_splits}-fold time series CV...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = -np.inf
    best_params = None
    
    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    Logger.add(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train[train_idx]
            X_val = X_train[val_idx]
            
            model = IsolationForest(
                n_estimators=params['n_estimators'],
                max_samples=params['max_samples'],
                contamination=params['contamination'],
                random_state=Config.RANDOM_STATE,
                n_jobs=Config.N_JOBS
            )
            
            model.fit(X_tr)
            # Use mean anomaly score as metric (higher is better for IF)
            score = model.score_samples(X_val).mean()
            scores.append(score)
        
        mean_score = np.mean(scores)
        
        if verbose and (i % 5 == 0 or mean_score > best_score):
            Logger.add(f"  Params {params}: score = {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    Logger.add(f"\nBest parameters: {best_params}")
    Logger.add(f"Best CV score: {best_score:.4f}")
    
    return best_params


# ============================================================================
# SCORING AND EVALUATION
# ============================================================================

def score_and_label(model: IsolationForest,
                    scaler: StandardScaler,
                    X: np.ndarray,
                    threshold: Optional[float] = None,
                    contamination: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score samples and assign anomaly labels.
    
    Parameters:
    -----------
    model : IsolationForest
        Trained model
    scaler : StandardScaler
        Fitted scaler
    X : np.ndarray
        Feature matrix
    threshold : float, optional
        Score threshold for anomalies (lower scores = anomalies)
    contamination : float, optional
        If threshold not provided, use this quantile
        
    Returns:
    --------
    scores : np.ndarray
        Anomaly scores
    labels : np.ndarray
        Binary labels (1 = anomaly, -1 = normal)
    """
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get anomaly scores
    scores = model.score_samples(X_scaled)
    
    # Determine threshold
    if threshold is None:
        if contamination is None:
            contamination = Config.ANOMALY_THRESHOLD_QUANTILE
        threshold = np.quantile(scores, contamination)
    
    # Assign labels (IF uses -1 for anomalies, 1 for normal)
    labels = np.where(scores < threshold, -1, 1)
    
    return scores, labels


def evaluate_anomalies(df: pd.DataFrame,
                       scores: np.ndarray,
                       labels: np.ndarray) -> Dict:
    """
    Compute evaluation metrics and statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe
    scores : np.ndarray
        Anomaly scores
    labels : np.ndarray
        Anomaly labels
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    n_anomalies = (labels == -1).sum()
    anomaly_rate = n_anomalies / len(labels)
    
    # Get anomaly indices
    anomaly_idx = np.where(labels == -1)[0]
    
    metrics = {
        'total_samples': len(labels),
        'n_anomalies': n_anomalies,
        'anomaly_rate': anomaly_rate,
        'score_mean': scores.mean(),
        'score_std': scores.std(),
        'score_min': scores.min(),
        'score_max': scores.max(),
        'anomaly_score_mean': scores[anomaly_idx].mean() if n_anomalies > 0 else np.nan,
        'normal_score_mean': scores[labels == 1].mean() if (labels == 1).sum() > 0 else np.nan
    }
    
    Logger.add(f"\n{'='*60}")
    Logger.add(f"ANOMALY DETECTION RESULTS")
    Logger.add(f"{'='*60}")
    Logger.add(f"Total samples:        {metrics['total_samples']:,}")
    Logger.add(f"Anomalies detected:   {metrics['n_anomalies']:,} ({metrics['anomaly_rate']*100:.2f}%)")
    Logger.add(f"Score range:          [{metrics['score_min']:.4f}, {metrics['score_max']:.4f}]")
    Logger.add(f"Mean anomaly score:   {metrics['anomaly_score_mean']:.4f}")
    Logger.add(f"Mean normal score:    {metrics['normal_score_mean']:.4f}")
    Logger.add(f"{'='*60}\n")
    
    return metrics


def detect_anomaly_events(df: pd.DataFrame,
                          gap_threshold: int = Config.EVENT_GAP_THRESHOLD,
                          min_points_per_event: int = 2) -> pd.DataFrame:
    """
    Group consecutive anomaly points into events, filtering isolated anomalies.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with 'Anomaly' column
    gap_threshold : int
        Maximum seconds between anomalies in same event
    min_points_per_event : int
        Minimum anomaly points required to consider it a valid event
        (default=2 filters out isolated single anomalies)
        
    Returns:
    --------
    pd.DataFrame
        Events with start, end, duration, and point count
    """
    anomalies = df[df['Anomaly'] == -1].copy()
    
    if len(anomalies) == 0:
        return pd.DataFrame()
    
    # Calculate time gaps
    anomalies['time_gap'] = anomalies['Timestamp'].diff().dt.total_seconds()
    
    # Mark event boundaries
    anomalies['new_event'] = (anomalies['time_gap'] > gap_threshold) | (anomalies['time_gap'].isna())
    anomalies['event_id'] = anomalies['new_event'].cumsum()
    
    # Aggregate events
    events = anomalies.groupby('event_id').agg({
        'Timestamp': ['min', 'max', 'count'],
        'Latitude': 'mean',
        'Longitude': 'mean',
        'speed_m_s': 'mean',
        'anomaly_score': 'mean'
    }).reset_index()
    
    events.columns = ['event_id', 'start_time', 'end_time', 'n_points', 
                     'mean_lat', 'mean_lon', 'mean_speed', 'mean_score']
    events['duration_seconds'] = (events['end_time'] - events['start_time']).dt.total_seconds()
    
    # Filter out events with too few points (isolated anomalies)
    total_events = len(events)
    events = events[events['n_points'] >= min_points_per_event].reset_index(drop=True)
    filtered_count = total_events - len(events)
    
    Logger.add(f"\nDetected {total_events} total anomaly events")
    Logger.add(f"Filtered out {filtered_count} isolated anomalies (< {min_points_per_event} points)")
    Logger.add(f"Retained {len(events)} significant anomaly events:")
    if len(events) > 0:
        Logger.add(events[['event_id', 'start_time', 'duration_seconds', 'n_points', 'mean_speed']].to_string())
    
    return events


def filter_isolated_anomalies(df: pd.DataFrame,
                             gap_threshold: int = Config.EVENT_GAP_THRESHOLD,
                             min_event_size: int = Config.MIN_EVENT_SIZE) -> pd.DataFrame:
    """
    Filter out isolated anomalies, keeping only clustered anomalies.
    
    This function identifies anomaly events and removes isolated single-point
    anomalies that are likely due to normal bird movement randomness rather
    than true anomalous behavior.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with 'Anomaly' column (-1 for anomaly, 1 for normal)
    gap_threshold : int
        Maximum seconds between anomalies to be considered same event
    min_event_size : int
        Minimum number of anomaly points to keep as a valid event
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with isolated anomalies reclassified as normal (Anomaly=1)
    """
    df_filtered = df.copy()
    
    # Get only anomalies
    anomalies = df_filtered[df_filtered['Anomaly'] == -1].copy()
    
    if len(anomalies) == 0:
        Logger.add("No anomalies found to filter")
        return df_filtered
    
    # Calculate time gaps between consecutive anomalies
    anomalies['time_gap'] = anomalies['Timestamp'].diff().dt.total_seconds()
    
    # Identify event boundaries
    anomalies['new_event'] = (anomalies['time_gap'] > gap_threshold) | (anomalies['time_gap'].isna())
    anomalies['event_id'] = anomalies['new_event'].cumsum()
    
    # Count points per event
    event_sizes = anomalies.groupby('event_id').size()
    
    # Find events that are too small (isolated anomalies)
    small_events = event_sizes[event_sizes < min_event_size].index
    isolated_indices = anomalies[anomalies['event_id'].isin(small_events)].index
    
    # Reclassify isolated anomalies as normal
    initial_anomaly_count = (df_filtered['Anomaly'] == -1).sum()
    df_filtered.loc[isolated_indices, 'Anomaly'] = 1
    final_anomaly_count = (df_filtered['Anomaly'] == -1).sum()
    
    filtered_count = initial_anomaly_count - final_anomaly_count
    
    Logger.add(f"\nFiltered isolated anomalies:")
    Logger.add(f"  Initial anomalies: {initial_anomaly_count}")
    Logger.add(f"  Isolated anomalies removed: {filtered_count}")
    Logger.add(f"  Remaining clustered anomalies: {final_anomaly_count}")
    Logger.add(f"  Kept anomalies in events with ≥{min_event_size} points")
    
    return df_filtered


# ============================================================================
# ONLINE DETECTION (SLIDING WINDOW)
# ============================================================================

def sliding_window_detection(df: pd.DataFrame,
                             model: IsolationForest,
                             scaler: StandardScaler,
                             feature_columns: List[str],
                             window_size: int = Config.SLIDING_WINDOW_SIZE,
                             cluster_threshold: int = Config.ANOMALY_CLUSTER_THRESHOLD) -> List[Dict]:
    """
    Perform sliding window anomaly detection for online monitoring.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features
    model : IsolationForest
        Trained model
    scaler : StandardScaler
        Fitted scaler
    feature_columns : list
        Feature column names
    window_size : int
        Size of sliding window
    cluster_threshold : int
        Minimum anomalies in window to trigger alert
        
    Returns:
    --------
    list of dict
        Alerts with timestamp and details
    """
    alerts = []
    
    for i in range(window_size, len(df)):
        window_df = df.iloc[i-window_size:i]
        X_window = window_df[feature_columns].values
        
        scores, labels = score_and_label(model, scaler, X_window)
        
        n_anomalies = (labels == -1).sum()
        
        if n_anomalies >= cluster_threshold:
            alert = {
                'timestamp': df.iloc[i]['Timestamp'],
                'window_start': df.iloc[i-window_size]['Timestamp'],
                'n_anomalies': n_anomalies,
                'mean_score': scores.mean(),
                'min_score': scores.min()
            }
            alerts.append(alert)
    
    return alerts


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(model: IsolationForest, 
               scaler: StandardScaler,
               feature_columns: List[str],
               model_path: str = Config.MODEL_SAVE_PATH,
               scaler_path: str = Config.SCALER_SAVE_PATH):
    """Save trained model and scaler."""
    # Create directory if needed
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_data = {
        'model': model,
        'feature_columns': feature_columns,
        'trained_date': datetime.now()
    }
    joblib.dump(model_data, model_path)
    joblib.dump(scaler, scaler_path)
    
    Logger.add(f"\nModel saved to: {model_path}")
    Logger.add(f"Scaler saved to: {scaler_path}")


def load_model(model_path: str = Config.MODEL_SAVE_PATH,
               scaler_path: str = Config.SCALER_SAVE_PATH) -> Tuple[IsolationForest, StandardScaler, List[str]]:
    """Load trained model and scaler."""
    model_data = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    Logger.add(f"\nModel loaded from: {model_path}")
    Logger.add(f"  Trained: {model_data['trained_date']}")
    Logger.add(f"  Features: {len(model_data['feature_columns'])}")
    
    return model_data['model'], scaler, model_data['feature_columns']


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_timeseries_anomalies(df: pd.DataFrame,
                              save_path: Optional[str] = None):
    """
    Plot time series of speed with anomalies highlighted.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with 'Anomaly' column
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    normal = df[df['Anomaly'] == 1]
    anomaly = df[df['Anomaly'] == -1]
    
    # Speed plot
    axes[0].plot(normal['Timestamp'], normal['speed_m_s'], 
                'b.', alpha=0.5, markersize=2, label='Normal')
    axes[0].plot(anomaly['Timestamp'], anomaly['speed_m_s'], 
                'r.', markersize=5, label='Anomaly')
    axes[0].set_ylabel('Speed (m/s)')
    axes[0].set_title('Bird Movement: Speed Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Altitude plot
    axes[1].plot(normal['Timestamp'], normal['Altitude'], 
                'b.', alpha=0.5, markersize=2, label='Normal')
    axes[1].plot(anomaly['Timestamp'], anomaly['Altitude'], 
                'r.', markersize=5, label='Anomaly')
    axes[1].set_ylabel('Altitude (m)')
    axes[1].set_title('Altitude Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Anomaly score plot
    axes[2].plot(df['Timestamp'], df['anomaly_score'], 
                'k-', alpha=0.3, linewidth=0.5)
    axes[2].axhline(y=df[df['Anomaly'] == -1]['anomaly_score'].max(), 
                   color='r', linestyle='--', label='Anomaly threshold')
    axes[2].set_ylabel('Anomaly Score')
    axes[2].set_xlabel('Time')
    axes[2].set_title('Anomaly Scores Over Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        Logger.add(f"Time series plot saved to: {save_path}")
    
    plt.show()


def plot_spatial_anomalies(df: pd.DataFrame,
                           save_path: Optional[str] = None):
    """
    Plot spatial distribution with anomalies highlighted.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with 'Anomaly' column
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    normal = df[df['Anomaly'] == 1]
    anomaly = df[df['Anomaly'] == -1]
    
    # Scatter plot
    axes[0].scatter(normal['Longitude'], normal['Latitude'], 
                   c='blue', alpha=0.3, s=10, label='Normal')
    axes[0].scatter(anomaly['Longitude'], anomaly['Latitude'], 
                   c='red', alpha=0.8, s=50, marker='x', label='Anomaly')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Spatial Distribution of Anomalies')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Speed colored by anomaly score
    scatter = axes[1].scatter(df['Longitude'], df['Latitude'], 
                             c=df['anomaly_score'], cmap='RdYlBu',
                             alpha=0.6, s=20)
    axes[1].scatter(anomaly['Longitude'], anomaly['Latitude'], 
                   c='red', s=100, marker='x', linewidths=2, 
                   label='Detected Anomaly')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Movement Track Colored by Anomaly Score')
    axes[1].legend()
    plt.colorbar(scatter, ax=axes[1], label='Anomaly Score')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        Logger.add(f"Spatial plot saved to: {save_path}")
    
    plt.show()


def plot_feature_distributions(df: pd.DataFrame,
                               feature_columns: List[str],
                               save_path: Optional[str] = None):
    """
    Plot feature distributions comparing normal vs anomalous points.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features and 'Anomaly' column
    feature_columns : list
        Features to plot
    save_path : str, optional
        Path to save figure
    """
    # Select most important features for visualization
    key_features = ['speed_m_s', 'vertical_speed', 'acceleration_abs', 
                   'speed_change_abs', 'speed_std_5', 'log_speed']
    key_features = [f for f in key_features if f in feature_columns][:6]
    
    n_features = len(key_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    normal = df[df['Anomaly'] == 1]
    anomaly = df[df['Anomaly'] == -1]
    
    for i, feature in enumerate(key_features):
        ax = axes[i]
        
        # Plot histograms
        ax.hist(normal[feature], bins=50, alpha=0.6, label='Normal', 
               color='blue', density=True)
        ax.hist(anomaly[feature], bins=30, alpha=0.6, label='Anomaly', 
               color='red', density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution: {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        Logger.add(f"Feature distribution plot saved to: {save_path}")
    
    plt.show()


def create_folium_map(df: pd.DataFrame,
                      save_path: str = 'results/anomaly_map.html'):
    """
    Create interactive Folium map with anomalies.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with 'Anomaly' column
    save_path : str
        Path to save HTML map
    """
    if not FOLIUM_AVAILABLE:
        Logger.add("Folium not available. Skipping map creation.")
        return
    
    # Calculate map center
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add normal points as a line
    normal = df[df['Anomaly'] == 1]
    coords = normal[['Latitude', 'Longitude']].values.tolist()
    
    folium.PolyLine(
        coords,
        color='blue',
        weight=2,
        opacity=0.5,
        popup='Normal track'
    ).add_to(m)
    
    # Add anomaly markers
    anomaly = df[df['Anomaly'] == -1]
    
    for _, row in anomaly.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8,
            popup=f"Anomaly<br>Time: {row['Timestamp']}<br>"
                  f"Speed: {row['speed_m_s']:.2f} m/s<br>"
                  f"Score: {row['anomaly_score']:.4f}",
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.7
        ).add_to(m)
    
    # Add heatmap layer for anomalies
    if len(anomaly) > 0:
        heat_data = anomaly[['Latitude', 'Longitude']].values.tolist()
        HeatMap(heat_data, name='Anomaly Heatmap', 
               min_opacity=0.3, radius=15).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(save_path)
    Logger.add(f"\nInteractive map saved to: {save_path}")
    Logger.add(f"Open in browser to view")


def compute_feature_importance_manual(model: IsolationForest,
                                     X: np.ndarray,
                                     feature_columns: List[str],
                                     n_repeats: int = 5) -> pd.DataFrame:
    """
    Manually compute permutation feature importance for Isolation Forest.
    
    Parameters:
    -----------
    model : IsolationForest
        Trained model
    X : np.ndarray
        Feature matrix (scaled)
    feature_columns : list
        Feature names
    n_repeats : int
        Number of permutation repeats
        
    Returns:
    --------
    pd.DataFrame
        Feature importance scores
    """
    Logger.add("\nComputing feature importance (this may take a moment)...")
    Logger.add(f"  Testing {len(feature_columns)} features with {n_repeats} repeats...")
    
    # Calculate baseline score
    baseline_scores = model.score_samples(X)
    baseline_metric = np.mean(baseline_scores)
    
    # Manually compute permutation importance
    importances = []
    
    for i, feature_name in enumerate(feature_columns):
        feature_importance_scores = []
        
        for repeat in range(n_repeats):
            # Copy data and permute one feature
            X_permuted = X.copy()
            np.random.seed(Config.RANDOM_STATE + repeat)
            np.random.shuffle(X_permuted[:, i])
            
            # Score with permuted feature
            permuted_scores = model.score_samples(X_permuted)
            permuted_metric = np.mean(permuted_scores)
            
            # Importance is the decrease in performance (baseline - permuted)
            # Larger decrease means more important feature
            importance = baseline_metric - permuted_metric
            feature_importance_scores.append(importance)
        
        importances.append({
            'mean': np.mean(feature_importance_scores),
            'std': np.std(feature_importance_scores)
        })
        
        if (i + 1) % 10 == 0:
            Logger.add(f"  Progress: {i + 1}/{len(feature_columns)} features")
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance_mean': [imp['mean'] for imp in importances],
        'importance_std': [imp['std'] for imp in importances]
    }).sort_values('importance_mean', ascending=False)
    
    Logger.add("\n  Top 10 Most Important Features:")
    Logger.add(importance_df.head(10).to_string(index=False))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(15)
    ax.barh(range(len(top_features)), top_features['importance_mean'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Permutation Importance')
    ax.set_title('Top 15 Feature Importances')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    return importance_df


# ============================================================================
# RETRAINING STRATEGY
# ============================================================================

def should_retrain(last_train_date: datetime,
                  n_new_samples: int,
                  retrain_days: int = Config.RETRAIN_PERIOD_DAYS,
                  min_samples: int = Config.MIN_NEW_SAMPLES_FOR_RETRAIN) -> bool:
    """
    Determine if model should be retrained.
    
    Parameters:
    -----------
    last_train_date : datetime
        Date of last training
    n_new_samples : int
        Number of new samples since last training
    retrain_days : int
        Days between retrains
    min_samples : int
        Minimum new samples needed
        
    Returns:
    --------
    bool
        Whether to retrain
    """
    days_since_train = (datetime.now() - last_train_date).days
    
    return (days_since_train >= retrain_days) or (n_new_samples >= min_samples)


def incremental_retrain(old_data: pd.DataFrame,
                       new_data: pd.DataFrame,
                       feature_columns: List[str],
                       window_days: int = 30) -> Tuple[IsolationForest, StandardScaler]:
    """
    Retrain model using sliding window approach.
    
    Parameters:
    -----------
    old_data : pd.DataFrame
        Historical data
    new_data : pd.DataFrame
        New data
    feature_columns : list
        Feature columns
    window_days : int
        Days of data to use for training
        
    Returns:
    --------
    model, scaler : tuple
        Retrained model and scaler
    """
    Logger.add(f"\nIncremental retraining with {window_days}-day window...")
    
    # Combine and get most recent data
    combined = pd.concat([old_data, new_data], ignore_index=True)
    combined = combined.sort_values('Timestamp')
    
    cutoff_date = combined['Timestamp'].max() - timedelta(days=window_days)
    recent_data = combined[combined['Timestamp'] >= cutoff_date]
    
    Logger.add(f"  Using {len(recent_data)} recent samples for retraining")
    
    # Prepare features
    X = recent_data[feature_columns].values
    
    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train new model
    model = train_isolation_forest(X_scaled, verbose=True)
    
    return model, scaler


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(device_id: Optional[int] = None,
                device_ids: Optional[List[int]] = None,
                data_path: str = Config.DATA_PATH,
                perform_hp_search: bool = False,
                save_results: bool = True,
                create_visualizations: bool = True,
                model_name: Optional[str] = "IF_Model") -> Dict:

    """
    Run complete anomaly detection pipeline.
    
    Parameters:
    -----------
    device_id : int, optional
        Device to analyze (None for all)
    device_ids : list of int, optional
        List of device IDs to analyze. Takes precedence over device_id.
    data_path : str
        Path to data CSV
    perform_hp_search : bool
        Whether to perform hyperparameter search
    save_results : bool
        Whether to save model and results
    create_visualizations : bool
        Whether to create plots
        
    Returns:
    --------
    dict
        Pipeline results including model, metrics, and predictions
    """
    # Reset logs at the start of training
    Logger.reset()
    
    Logger.add("="*80)
    Logger.add("BIRD MOVEMENT ANOMALY DETECTION PIPELINE")
    Logger.add("="*80)
    
    # Step 1: Load data
    df = load_data(device_id=device_id, device_ids=device_ids, path=data_path)
    
    if len(df) < Config.MIN_SAMPLES_FOR_TRAINING:
        raise ValueError(f"Insufficient data: {len(df)} samples "
                        f"(minimum {Config.MIN_SAMPLES_FOR_TRAINING} required)")
    
    # Step 2: Engineer features
    df, feature_columns = build_features(df)
    
    # Step 3: Train/test split (time-aware)
    train_df, test_df = split_train_test(df)
    
    X_train = train_df[feature_columns].values
    X_test = test_df[feature_columns].values
    
    # Step 4: Scale features
    Logger.add("\nScaling features...")
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 5: Hyperparameter search (optional)
    if perform_hp_search:
        best_params = hyperparameter_search(X_train_scaled)
        model = train_isolation_forest(
            X_train_scaled,
            contamination=best_params['contamination'],
            n_estimators=best_params['n_estimators'],
            max_samples=best_params['max_samples']
        )
    else:
        model = train_isolation_forest(X_train_scaled)
    
    # Step 6: Score and evaluate on test set
    Logger.add("\n" + "="*60)
    Logger.add("EVALUATING ON TEST SET")
    Logger.add("="*60)
    
    test_scores, test_labels = score_and_label(model, scaler, X_test)
    test_df['anomaly_score'] = test_scores
    test_df['Anomaly'] = test_labels
    
    test_metrics = evaluate_anomalies(test_df, test_scores, test_labels)
    
    # Step 6.5: Filter isolated anomalies
    test_df = filter_isolated_anomalies(
        test_df, 
        gap_threshold=Config.EVENT_GAP_THRESHOLD,
        min_event_size=Config.MIN_EVENT_SIZE
    )
    
    # Re-evaluate after filtering
    test_scores_filtered = test_df['anomaly_score'].values
    test_labels_filtered = test_df['Anomaly'].values
    test_metrics_filtered = evaluate_anomalies(test_df, test_scores_filtered, test_labels_filtered)
    
    # Step 7: Detect anomaly events (now only clustered events)
    events = detect_anomaly_events(
        test_df, 
        gap_threshold=Config.EVENT_GAP_THRESHOLD,
        min_points_per_event=Config.MIN_EVENT_SIZE
    )
    
    # Step 8: Score full dataset for visualization
    Logger.add("\nScoring full dataset...")
    full_scores, full_labels = score_and_label(model, scaler, df[feature_columns].values)
    df['anomaly_score'] = full_scores
    df['Anomaly'] = full_labels
    
    # Filter isolated anomalies from full dataset
    df = filter_isolated_anomalies(
        df,
        gap_threshold=Config.EVENT_GAP_THRESHOLD,
        min_event_size=Config.MIN_EVENT_SIZE
    )
    
    full_scores = df['anomaly_score'].values
    full_labels = df['Anomaly'].values
    full_metrics = evaluate_anomalies(df, full_scores, full_labels)
    
    # Step 9: Feature importance
    if len(test_df) > 100:  # Only if we have enough test samples
        importance_df = compute_feature_importance_manual(
            model, X_test_scaled, feature_columns
        )
    else:
        importance_df = None
    
    # Step 10: Save model and results
    if save_results:
        save_model(model, scaler, feature_columns, model_path='models/'+model_name+'.pkl', scaler_path='models/'+model_name+'_Scaler.pkl')
        
        # Save results CSV
        output_df = df[['Timestamp', 'Device_id', 'Latitude', 'Longitude', 
                       'Altitude', 'speed_m_s', 'anomaly_score', 'Anomaly']].copy()
        
        Path(Config.RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(Config.RESULTS_PATH, index=False)
        Logger.add(f"\nResults saved to: {Config.RESULTS_PATH}")

        df_to_upsert = df[["Id", "Anomaly"]].copy()
        df_to_upsert["Id"] = df_to_upsert["Id"].astype(int)
        df_to_upsert["Anomaly"] = df_to_upsert["Anomaly"].astype(int)
       
        Logger.add(db_operations.save_outlier_result(df_to_upsert.to_dict(orient="records")))

    
    # Step 11: Visualizations
    if create_visualizations:
        Logger.add("\nGenerating visualizations...")
        
        plot_timeseries_anomalies(test_df, 
                                 save_path='results/timeseries_anomalies.png')
        
        plot_spatial_anomalies(test_df,
                              save_path='results/spatial_anomalies.png')
        
        plot_feature_distributions(test_df, feature_columns,
                                  save_path='results/feature_distributions.png')
        
        create_folium_map(test_df)
    
    # Return results
    results = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'test_metrics': test_metrics,
        'test_metrics_filtered': test_metrics_filtered,
        'full_metrics': full_metrics,
        'events': events,
        'test_df': test_df,
        'full_df': df,
        'importance': importance_df
    }
    
    return results


def run_inference(new_data_path: str,
                 model_path: str = Config.MODEL_SAVE_PATH,
                 scaler_path: str = Config.SCALER_SAVE_PATH,
                 device_id: Optional[int] = None,
                 device_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Run inference on new data using saved model.
    
    Parameters:
    -----------
    new_data_path : str
        Path to new data CSV
    model_path : str
        Path to saved model
    scaler_path : str
        Path to saved scaler
    device_id : int, optional
        Device to analyze
    device_ids : list of int, optional
        List of device IDs to analyze. Takes precedence over device_id.
        
    Returns:
    --------
    pd.DataFrame
        Predictions with anomaly scores and labels
    """
    Logger.add("="*80)
    Logger.add("RUNNING INFERENCE ON NEW DATA")
    Logger.add("="*80)
    
    # Load model
    model, scaler, feature_columns = load_model(model_path, scaler_path)
    
    # Load new data
    df = load_data(device_id=device_id, device_ids=device_ids, path=new_data_path)
    
    # Engineer features
    df, _ = build_features(df)
    
    # Score
    X = df[feature_columns].values
    scores, labels = score_and_label(model, scaler, X)
    
    df['anomaly_score'] = scores
    df['Anomaly'] = labels
    
    # Filter isolated anomalies
    df = filter_isolated_anomalies(
        df,
        gap_threshold=Config.EVENT_GAP_THRESHOLD,
        min_event_size=Config.MIN_EVENT_SIZE
    )
    
    # Evaluate
    scores = df['anomaly_score'].values
    labels = df['Anomaly'].values
    metrics = evaluate_anomalies(df, scores, labels)
    
    # Detect events (only clustered anomalies)
    events = detect_anomaly_events(
        df,
        gap_threshold=Config.EVENT_GAP_THRESHOLD,
        min_points_per_event=Config.MIN_EVENT_SIZE
    )
    
    return df, events, metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
'''
if __name__ == "__main__":
    """
    Example usage of the anomaly detection pipeline.
    
    This module can be imported and used in other scripts:
    
    EXAMPLE 1: Train on multiple devices
    -------------------------------------
    from train import run_pipeline, Logger, Config
    
    # Train on devices 1, 2, and 3
    results = run_pipeline(
        device_ids=[1, 2, 3],
        perform_hp_search=False,
        create_visualizations=True
    )
    
    # Access logs
    logs = Logger.get_logs()
    for log in logs:
        print(log)
    
    EXAMPLE 2: Train on all devices
    --------------------------------
    from train import run_pipeline
    
    # Train on all devices (don't specify device_id or device_ids)
    results = run_pipeline(
        create_visualizations=True
    )
    
    EXAMPLE 3: Train with custom filtering on multiple devices
    -----------------------------------------------------------
    from train import run_pipeline, Config
    
    # Adjust filtering to keep only events with 3+ anomaly points
    Config.MIN_EVENT_SIZE = 3
    Config.EVENT_GAP_THRESHOLD = 600  # 10 minutes
    
    results = run_pipeline(
        device_ids=[1, 2, 5, 10],
        perform_hp_search=False,
        create_visualizations=True
    )
    
    EXAMPLE 4: Run inference on multiple devices
    ---------------------------------------------
    from train import run_inference, Logger
    
    df, events, metrics = run_inference(
        new_data_path='new_data.csv',
        device_ids=[1, 2, 3]
    )
    
    # Events now only contain clustered anomalies
    print(f"Detected {len(events)} significant anomaly events")
    
    logs = Logger.get_logs_string()
    print(logs)
    
    EXAMPLE 5: Load and score specific devices
    -------------------------------------------
    from train import load_model, load_data, build_features, score_and_label
    
    model, scaler, feature_columns = load_model()
    df = load_data(device_ids=[1, 3, 5], path='new_data.csv')
    df, _ = build_features(df)
    scores, labels = score_and_label(model, scaler, df[feature_columns].values)
    
    EXAMPLE 6: Manual filtering with multiple devices
    --------------------------------------------------
    from train import (
        load_data, build_features, load_model, 
        score_and_label, filter_isolated_anomalies
    )
    
    # Load and score data from multiple devices
    df = load_data(device_ids=[1, 2, 3])
    df, feature_columns = build_features(df)
    model, scaler, _ = load_model()
    scores, labels = score_and_label(model, scaler, df[feature_columns].values)
    df['anomaly_score'] = scores
    df['Anomaly'] = labels
    
    # Filter with custom parameters
    df_filtered = filter_isolated_anomalies(
        df,
        gap_threshold=900,     # 15 minutes
        min_event_size=2       # Keep events with 2+ points
    )
    
    # Check results per device
    print(df_filtered.groupby('Device_id')['Anomaly'].value_counts())
    
    
    FILTERING PARAMETERS GUIDE:
    ---------------------------
    MIN_EVENT_SIZE: Controls how many consecutive anomaly points are needed
                    to consider it a "real" anomaly event
    
    - MIN_EVENT_SIZE = 1: Keep all anomalies (no filtering)
    - MIN_EVENT_SIZE = 2: Remove isolated single-point anomalies (recommended)
    - MIN_EVENT_SIZE = 3: Only keep events with 3+ consecutive anomalies
    - MIN_EVENT_SIZE = 5: Very strict, only significant anomaly clusters
    
    EVENT_GAP_THRESHOLD: Maximum time gap (seconds) between anomalies to be
                         considered part of the same event
    
    - 300s (5 min): Tight grouping, bird must stay anomalous continuously
    - 600s (10 min): Moderate grouping (good for most cases)
    - 900s (15 min): Default - allows some normal behavior between anomalies
    - 1800s (30 min): Loose grouping, treats broader patterns as single events
    
    DEVICE SELECTION:
    -----------------
    - device_id=1: Train on single device
    - device_ids=[1, 2, 3]: Train on multiple specific devices
    - No parameter: Train on all devices in dataset
    """
    
    # Demo: Run pipeline on multiple devices
    Logger.add("\n" + "="*80)
    Logger.add("DEMO MODE: Training on Multiple Devices")
    Logger.add("="*80 + "\n")
    
    # Example: Train on devices 1 and 2
    results = run_pipeline(
        device_ids=[1, 2],
        perform_hp_search=False,
        create_visualizations=True
    )
    
    Logger.add("\n" + "="*80)
    Logger.add("PIPELINE COMPLETE!")
    Logger.add("="*80)
    Logger.add(f"\nAnomalies detected (after filtering): {results['test_metrics_filtered']['n_anomalies']}")
    Logger.add(f"Anomaly rate: {results['test_metrics_filtered']['anomaly_rate']*100:.2f}%")
    Logger.add(f"Events identified: {len(results['events'])}")
'''