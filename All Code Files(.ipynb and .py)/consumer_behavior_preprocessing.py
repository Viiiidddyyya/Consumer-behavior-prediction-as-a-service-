"""
Consumer Behavior Prediction Service - Data Preprocessing Pipeline
===================================================================

This notebook handles:
1. Data Collection (Google Trends, Retail Sales, Simulated Social Media)
2. Data Cleaning and Alignment
3. Feature Engineering
4. Export processed datasets

Author: Data Engineering Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CONSUMER BEHAVIOR PREDICTION SERVICE - PREPROCESSING PIPELINE")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: CONFIGURATION AND SETUP
# ============================================================================

print("[1/7] Configuration Setup")
print("-" * 80)

# Date range configuration
START_DATE = '2018-01-01'
END_DATE = '2023-12-31'

# UK retail keywords for Google Trends
RETAIL_KEYWORDS = [
    'christmas gifts',
    'black friday',
    'boxing day sale',
    'summer sale',
    'discount',
    'online shopping',
    'gift ideas',
    'shopping deals',
    'clearance sale',
    'holiday shopping'
]

# Holiday dates for UK (major retail holidays)
UK_HOLIDAYS = {
    '2018-12-25': 'Christmas', '2018-12-26': 'Boxing Day', '2018-11-23': 'Black Friday',
    '2019-12-25': 'Christmas', '2019-12-26': 'Boxing Day', '2019-11-29': 'Black Friday',
    '2020-12-25': 'Christmas', '2020-12-26': 'Boxing Day', '2020-11-27': 'Black Friday',
    '2021-12-25': 'Christmas', '2021-12-26': 'Boxing Day', '2021-11-26': 'Black Friday',
    '2022-12-25': 'Christmas', '2022-12-26': 'Boxing Day', '2022-11-25': 'Black Friday',
    '2023-12-25': 'Christmas', '2023-12-26': 'Boxing Day', '2023-11-24': 'Black Friday',
}

print(f"Date Range: {START_DATE} to {END_DATE}")
print(f"Keywords: {len(RETAIL_KEYWORDS)} retail-related terms")
print(f"Holidays Tracked: {len(set(UK_HOLIDAYS.values()))} major UK retail holidays")
print()

# ============================================================================
# SECTION 2: GOOGLE TRENDS DATA COLLECTION
# ============================================================================

print("[2/7] Google Trends Data Collection")
print("-" * 80)

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
    print("PyTrends library detected - will fetch live data")
except ImportError:
    PYTRENDS_AVAILABLE = False
    print("PyTrends not available - will generate synthetic trend data")

def fetch_google_trends_data():
    """Fetch Google Trends data for UK retail keywords"""
    
    if PYTRENDS_AVAILABLE:
        print("Fetching Google Trends data from API...")
        try:
            pytrends = TrendReq(hl='en-GB', tz=0)
            all_trends = []
            
            for keyword in RETAIL_KEYWORDS:
                print(f"  - Fetching: {keyword}")
                pytrends.build_payload([keyword], cat=0, timeframe=f'{START_DATE} {END_DATE}', geo='GB')
                trend_data = pytrends.interest_over_time()
                
                if not trend_data.empty:
                    trend_data = trend_data[[keyword]].reset_index()
                    trend_data.columns = ['date', keyword]
                    all_trends.append(trend_data)
            
            # Merge all keyword trends
            trends_df = all_trends[0]
            for df in all_trends[1:]:
                trends_df = pd.merge(trends_df, df, on='date', how='outer')
            
            print(f"Successfully fetched data for {len(RETAIL_KEYWORDS)} keywords")
            return trends_df
            
        except Exception as e:
            print(f"Error fetching from API: {e}")
            print("Falling back to synthetic data generation")
            return generate_synthetic_trends()
    else:
        return generate_synthetic_trends()

def generate_synthetic_trends():
    """Generate realistic synthetic Google Trends data"""
    print("Generating synthetic Google Trends data...")
    
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='W')
    trends_data = {'date': date_range}
    
    np.random.seed(42)
    
    for keyword in RETAIL_KEYWORDS:
        base_trend = 30 + np.random.randn(len(date_range)) * 5
        
        seasonal_component = 20 * np.sin(2 * np.pi * np.arange(len(date_range)) / 52)
        
        holiday_spikes = np.zeros(len(date_range))
        for i, date in enumerate(date_range):
            month = date.month
            if month == 11:
                holiday_spikes[i] = 30
            elif month == 12:
                holiday_spikes[i] = 40
            elif month in [1, 7]:
                holiday_spikes[i] = 15
        
        trends_data[keyword] = np.clip(base_trend + seasonal_component + holiday_spikes, 0, 100)
    
    trends_df = pd.DataFrame(trends_data)
    print(f"Generated synthetic data for {len(RETAIL_KEYWORDS)} keywords")
    print(f"Date range: {trends_df['date'].min()} to {trends_df['date'].max()}")
    return trends_df

google_trends_df = fetch_google_trends_data()
print(f"Google Trends shape: {google_trends_df.shape}")
print()

# ============================================================================
# SECTION 3: RETAIL SALES DATA
# ============================================================================

print("[3/7] Retail Sales Data Collection")
print("-" * 80)

def generate_uk_retail_sales():
    """Generate realistic UK retail sales index data"""
    print("Generating UK retail sales index data...")
    
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    
    np.random.seed(42)
    
    base_sales = 100
    growth_rate = 0.002
    sales_values = [base_sales]
    
    for i in range(1, len(date_range)):
        month = date_range[i].month
        
        seasonal_factor = 1.0
        if month == 12:
            seasonal_factor = 1.3
        elif month == 11:
            seasonal_factor = 1.15
        elif month in [1, 7]:
            seasonal_factor = 1.08
        elif month in [2, 8]:
            seasonal_factor = 0.95
        
        new_value = sales_values[-1] * (1 + growth_rate) * seasonal_factor
        new_value += np.random.randn() * 2
        sales_values.append(new_value)
    
    retail_df = pd.DataFrame({
        'date': date_range,
        'retail_sales_index': sales_values,
        'retail_sales_value_gbp_millions': np.array(sales_values) * 300
    })
    
    print(f"Generated retail sales data: {len(retail_df)} months")
    print(f"Date range: {retail_df['date'].min()} to {retail_df['date'].max()}")
    return retail_df

retail_sales_df = generate_uk_retail_sales()
print(f"Retail Sales shape: {retail_sales_df.shape}")
print()

# ============================================================================
# SECTION 4: SIMULATED SOCIAL MEDIA DATA
# ============================================================================

print("[4/7] Social Media Data Simulation")
print("-" * 80)

def generate_social_media_data(trends_df):
    """Generate simulated social media engagement data correlated with trends"""
    print("Generating simulated social media engagement data...")
    
    social_df = trends_df[['date']].copy()
    
    np.random.seed(42)
    
    avg_search_volume = trends_df[RETAIL_KEYWORDS].mean(axis=1)
    
    social_df['posts_count'] = (avg_search_volume * 1000 + np.random.randn(len(social_df)) * 500).clip(0)
    social_df['engagement_score'] = (avg_search_volume * 50 + np.random.randn(len(social_df)) * 200).clip(0)
    social_df['sentiment_score'] = 0.6 + np.random.randn(len(social_df)) * 0.15
    social_df['sentiment_score'] = social_df['sentiment_score'].clip(0, 1)
    social_df['share_count'] = (social_df['posts_count'] * 0.3 + np.random.randn(len(social_df)) * 100).clip(0)
    social_df['comment_count'] = (social_df['posts_count'] * 0.5 + np.random.randn(len(social_df)) * 200).clip(0)
    
    print(f"Generated social media data: {len(social_df)} records")
    print(f"Features: posts, engagement, sentiment, shares, comments")
    return social_df

social_media_df = generate_social_media_data(google_trends_df)
print(f"Social Media shape: {social_media_df.shape}")
print()

# ============================================================================
# SECTION 5: DATA ALIGNMENT AND MERGING
# ============================================================================

print("[5/7] Data Alignment and Integration")
print("-" * 80)

print("Aligning all data sources to monthly frequency...")

google_trends_df['date'] = pd.to_datetime(google_trends_df['date'])
google_trends_df['year_month'] = google_trends_df['date'].dt.to_period('M')
google_trends_monthly = google_trends_df.groupby('year_month')[RETAIL_KEYWORDS].mean().reset_index()
google_trends_monthly['date'] = google_trends_monthly['year_month'].dt.to_timestamp()
google_trends_monthly = google_trends_monthly.drop('year_month', axis=1)

social_media_df['date'] = pd.to_datetime(social_media_df['date'])
social_media_df['year_month'] = social_media_df['date'].dt.to_period('M')
social_media_monthly = social_media_df.groupby('year_month')[
    ['posts_count', 'engagement_score', 'sentiment_score', 'share_count', 'comment_count']
].mean().reset_index()
social_media_monthly['date'] = social_media_monthly['year_month'].dt.to_timestamp()
social_media_monthly = social_media_monthly.drop('year_month', axis=1)

retail_sales_df['date'] = pd.to_datetime(retail_sales_df['date'])

print("Merging all datasets...")
merged_df = retail_sales_df.copy()
merged_df = pd.merge(merged_df, google_trends_monthly, on='date', how='left')
merged_df = pd.merge(merged_df, social_media_monthly, on='date', how='left')

merged_df = merged_df.sort_values('date').reset_index(drop=True)

print(f"Merged dataset shape: {merged_df.shape}")
print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
print()

# ============================================================================
# SECTION 6: FEATURE ENGINEERING
# ============================================================================

print("[6/7] Feature Engineering")
print("-" * 80)

print("Creating temporal features...")
merged_df['year'] = merged_df['date'].dt.year
merged_df['month'] = merged_df['date'].dt.month
merged_df['quarter'] = merged_df['date'].dt.quarter
merged_df['day_of_year'] = merged_df['date'].dt.dayofyear

print("Creating cyclical features for seasonality...")
merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)

print("Creating holiday proximity features...")
def days_to_nearest_holiday(date, holidays_dict):
    min_distance = float('inf')
    for holiday_date in holidays_dict.keys():
        holiday_dt = pd.to_datetime(holiday_date)
        distance = abs((date - holiday_dt).days)
        if distance < min_distance:
            min_distance = distance
    return min_distance

merged_df['days_to_holiday'] = merged_df['date'].apply(
    lambda x: days_to_nearest_holiday(x, UK_HOLIDAYS)
)

merged_df['is_holiday_month'] = merged_df['date'].apply(
    lambda x: any(pd.to_datetime(h).month == x.month and pd.to_datetime(h).year == x.year 
                  for h in UK_HOLIDAYS.keys())
).astype(int)

print("Creating lag features for retail sales (1-6 months)...")
for lag in range(1, 7):
    merged_df[f'retail_sales_lag_{lag}'] = merged_df['retail_sales_index'].shift(lag)

print("Creating moving averages (3, 6, 12 months)...")
for window in [3, 6, 12]:
    merged_df[f'retail_sales_ma_{window}'] = merged_df['retail_sales_index'].rolling(
        window=window, min_periods=1
    ).mean()

print("Creating volatility indicators...")
for window in [3, 6]:
    merged_df[f'retail_sales_volatility_{window}'] = merged_df['retail_sales_index'].rolling(
        window=window, min_periods=1
    ).std()

print("Creating aggregate trend features...")
trend_columns = [col for col in merged_df.columns if col in RETAIL_KEYWORDS]
if trend_columns:
    merged_df['avg_search_volume'] = merged_df[trend_columns].mean(axis=1)
    merged_df['max_search_volume'] = merged_df[trend_columns].max(axis=1)
    merged_df['search_volume_std'] = merged_df[trend_columns].std(axis=1)

print("Creating year-over-year growth rate...")
merged_df['yoy_growth_rate'] = merged_df['retail_sales_index'].pct_change(periods=12) * 100

print("Creating momentum indicators...")
merged_df['momentum_3m'] = merged_df['retail_sales_index'] - merged_df['retail_sales_lag_3']
merged_df['momentum_6m'] = merged_df['retail_sales_index'] - merged_df['retail_sales_lag_6']

print(f"Total features created: {merged_df.shape[1]}")
print()

# ============================================================================
# SECTION 7: DATA EXPORT
# ============================================================================

print("[7/7] Data Export")
print("-" * 80)

print("Saving processed datasets...")

google_trends_monthly.to_csv('google_trends_monthly.csv', index=False)
print("  - google_trends_monthly.csv")

retail_sales_df.to_csv('retail_sales_monthly.csv', index=False)
print("  - retail_sales_monthly.csv")

social_media_monthly.to_csv('social_media_monthly.csv', index=False)
print("  - social_media_monthly.csv")

merged_df.to_csv('consumer_behavior_dataset_processed.csv', index=False)
print("  - consumer_behavior_dataset_processed.csv (MAIN DATASET)")

print()
print("=" * 80)
print("DATA SUMMARY")
print("=" * 80)
print(f"Total records: {len(merged_df)}")
print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
print(f"Total features: {merged_df.shape[1]}")
print(f"Missing values: {merged_df.isnull().sum().sum()}")
print()

print("Feature Categories:")
print(f"  - Temporal features: 4 (year, month, quarter, day_of_year)")
print(f"  - Cyclical features: 2 (month_sin, month_cos)")
print(f"  - Holiday features: 2 (days_to_holiday, is_holiday_month)")
print(f"  - Lag features: 6 (1-6 months)")
print(f"  - Moving averages: 3 (3, 6, 12 months)")
print(f"  - Volatility features: 2 (3, 6 months)")
print(f"  - Search trend features: {len(RETAIL_KEYWORDS)} keywords + 3 aggregates")
print(f"  - Social media features: 5 (posts, engagement, sentiment, shares, comments)")
print(f"  - Growth indicators: 3 (YoY growth, 3m momentum, 6m momentum)")
print()

print("Dataset Preview:")
print(merged_df.head(10))
print()

print("Dataset Info:")
print(merged_df.info())
print()

print("Statistical Summary:")
print(merged_df.describe())
print()

print("=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)
print()
print("Next Steps:")
print("1. Load 'consumer_behavior_dataset_processed.csv' for modeling")
print("2. Split data into train/test sets (80/20 or time-based)")
print("3. Implement forecasting models (Prophet, SARIMAX, LightGBM)")
print("4. Evaluate using sMAPE, RMSE, and Direction-of-Change metrics")
print()
print("Ready for model development!")
