"""Explicit Load -> Validate -> Clean -> Explore -> Visualize -> Transform -> Export -> Summary pipeline."""
import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from src.shared.config import load_config, setup_logging

LOG = logging.getLogger(__name__)

def load(path):
    """Load source CSV with useful failure messages."""
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError) as exc:
        LOG.exception('Cannot load source CSV')
        raise RuntimeError(f'Could not load {path}: {exc}') from exc

def validate(df, cfg):
    """Check required columns and nonempty source."""
    required = set(cfg['features']['numeric'] + cfg['features']['categorical'] + [cfg['features']['target']])
    missing = sorted(required - set(df.columns))
    if missing or df.empty:
        raise ValueError(f'Dataset invalid; missing columns={missing}; empty={df.empty}')
    return df

def clean(df, cfg):
    """Remove implausible rows and normalize selected categorical fields."""
    cols = cfg['features']['numeric'] + cfg['features']['categorical'] + [cfg['features']['target']]
    out = df.loc[:, cols].copy()
    for col in cfg['features']['numeric'] + [cfg['features']['target']]:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    for col in cfg['features']['categorical']:
        out[col] = out[col].fillna('unknown').astype(str).str.strip().str.lower().replace('', 'unknown')
    out = out.dropna(subset=cfg['features']['numeric'] + [cfg['features']['target']]).drop_duplicates()
    c = cfg['cleaning']
    out = out.loc[out.price.between(c['min_price'], c['max_price']) & out.year.between(c['min_year'], c['max_year']) & out.odometer.between(0, c['max_odometer'])]
    if out.empty:
        raise ValueError('No rows remain after cleaning; inspect thresholds and input CSV')
    return out

def explore(df):
    """Compute descriptive statistics for cleaned cars."""
    return {'rows': int(len(df)), 'median_price': float(df.price.median()), 'mean_price': float(df.price.mean()), 'manufacturers': int(df.manufacturer.nunique())}

def visualize(df, path):
    """Write a price distribution PNG."""
    dest = Path(path); dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.hist(df.price, bins=40)
    ax.set(xlabel='Listing price (USD)', ylabel='Number of listings', title='Used car listing prices')
    fig.tight_layout(); fig.savefig(dest, dpi=140); plt.close(fig)

def transform(df, cfg):
    """Select consistent model columns; categorical encoding occurs inside sklearn Pipeline."""
    return df[cfg['features']['numeric'] + cfg['features']['categorical'] + [cfg['features']['target']]].copy()

def export(df, path):
    """Persist prepared rows for reproducible model training."""
    dest = Path(path); dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)

def summary(stats, path):
    """Persist EDA summary JSON."""
    dest = Path(path); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(stats, indent=2), encoding='utf-8')

def main():
    """Run the full data preparation pipeline."""
    setup_logging(); cfg = load_config(); paths = cfg['paths']
    df = load(paths['raw_data']); LOG.info('Loaded %d rows', len(df))
    df = validate(df, cfg)
    df = clean(df, cfg); LOG.info('Retained %d cleaned rows', len(df))
    stats = explore(df)
    visualize(df, paths['chart'])
    df = transform(df, cfg)
    export(df, paths['cleaned_data'])
    summary(stats, paths['summary'])
    LOG.info('Finished pipeline: %s', stats)

if __name__ == '__main__':
    main()
