#!/usr/bin/env python3
"""
Part 7: S&P 500 Cross-Sectional Image Pipeline (Local Execution)
Two phases: (1) Check existing price data pickles  (2) Parallel image generation
Usage: python3 run_part7_local.py
"""

import os, json, time, warnings, sys, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count
import requests
from io import StringIO

# === CONFIGURATION ===
CONFIG = {
    'start_date': '2010-01-01',
    'window_size': 20,
    'predict_horizon': 5,
    'embargo_days': 10,
    'test_start_date': '2023-01-01',
    'fig_size': (4, 4),
    'fig_dpi': 120,
    'img_pixels': 480,
    'n_workers': max(1, cpu_count() - 2),
}

CLASS_NAMES_6 = ['down_3plus', 'down_2_3', 'down_1_2', 'up_1_2', 'up_2_3', 'up_3plus']

BASE_DIR = Path(__file__).parent / 'cross_sectional_data'
IMG_DIR = BASE_DIR / 'images'
META_DIR = BASE_DIR / 'metadata'
DATA_DIR = Path(__file__).parent / 'colab_upload' / 'price_data'
for d in [IMG_DIR, META_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f'CPU cores: {cpu_count()}, Workers: {CONFIG["n_workers"]}')
print(f'Output: {BASE_DIR}')


# === S&P 500 Constituents ===
def get_sp500_constituents():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; SP500-Chart-Dataset/1.0)'}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    constituents = []
    for _, row in df.iterrows():
        constituents.append({
            'ticker': row['Symbol'],
            'company': row['Security'],
            'sector': row['GICS Sector'],
            'sub_industry': row['GICS Sub-Industry'],
            'date_added': str(row.get('Date added', '')),
        })
    return constituents


# === Data Download ===
def download_stock_data(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df is None or len(df) < 200:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA120'] = df['Close'].rolling(120).mean()
        ma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['BB_upper'] = ma20 + 2 * std20
        df['BB_lower'] = ma20 - 2 * std20
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))
        df = df.dropna()
        return df
    except Exception:
        return None


# === Sample Generation ===
def classify_return_6(pct):
    if pct < -3: return 'down_3plus'
    elif -3 <= pct < -2: return 'down_2_3'
    elif -2 <= pct < -1: return 'down_1_2'
    elif 1 < pct <= 2: return 'up_1_2'
    elif 2 < pct <= 3: return 'up_2_3'
    elif pct > 3: return 'up_3plus'
    return None

def generate_samples(df, ticker):
    w = CONFIG['window_size']
    h = CONFIG['predict_horizon']
    samples = []
    for i in range(0, len(df) - w - h + 1):
        fc = df['Close'].iloc[i + w + h - 1]
        cc = df['Close'].iloc[i + w - 1]
        pct = (fc - cc) / cc * 100
        label = classify_return_6(pct)
        if label is None:
            continue
        samples.append({
            'index': i,
            'end_date': df.index[i + w - 1].strftime('%Y-%m-%d'),
            'label': label,
            'label_idx': CLASS_NAMES_6.index(label),
            'pct_return': round(pct, 4),
            'ticker': ticker,
        })
    return samples

def temporal_split_samples(samples):
    ts = pd.Timestamp(CONFIG['test_start_date'])
    emb = pd.Timedelta(days=CONFIG['embargo_days'])
    train = [s for s in samples if pd.Timestamp(s['end_date']) < ts - emb]
    test = [s for s in samples if pd.Timestamp(s['end_date']) >= ts]
    return train, test


# === Image Generation (for multiprocessing worker) ===
def generate_images_for_stock_worker(args):
    """Worker: generate images for a ticker (loads data from pickle)"""
    ticker, samples = args
    matplotlib.use('Agg')
    import mplfinance as mpf
    import matplotlib.pyplot as plt

    # Load price data from pickle
    pkl_path = DATA_DIR / f'{ticker}.pkl'
    if not pkl_path.exists():
        return {'ticker': ticker, 'status': 'no_pickle', 'generated': 0, 'skipped': 0, 'errors': 0}

    df = pd.read_pickle(pkl_path)
    w = CONFIG['window_size']
    ticker_dir = IMG_DIR / ticker
    generated, skipped, errors = 0, 0, 0

    for cls in CLASS_NAMES_6:
        (ticker_dir / cls).mkdir(parents=True, exist_ok=True)

    for s in samples:
        fname = f"{ticker}_{s['index']}_{s['end_date'].replace('-','')}.png"
        fpath = ticker_dir / s['label'] / fname

        if fpath.exists():
            skipped += 1
            continue

        try:
            wd = df.iloc[s['index']:s['index'] + w].copy()
            if len(wd) < w:
                errors += 1
                continue

            for col in ['MA5', 'MA60', 'MA120', 'BB_upper', 'BB_lower']:
                if col in wd.columns:
                    wd[col] = wd[col].ffill().bfill()

            # Percentage normalization: all prices relative to first Close
            base_price = wd['Close'].iloc[0]
            if base_price != 0 and not np.isnan(base_price):
                for col in ['Open', 'High', 'Low', 'Close', 'MA5', 'MA60', 'MA120', 'BB_upper', 'BB_lower']:
                    if col in wd.columns:
                        wd[col] = (wd[col] / base_price - 1) * 100

            indicator_cols = [c for c in ['MA5', 'MA60', 'MA120', 'BB_upper', 'BB_lower'] if c in wd.columns]
            all_vals = pd.concat([wd['High'], wd['Low']] + [wd[c] for c in indicator_cols])
            y_min, y_max = all_vals.min(), all_vals.max()
            margin = (y_max - y_min) * 0.05
            ylim = (y_min - margin, y_max + margin)

            apds = [
                mpf.make_addplot(wd['MA5'], color='blue', width=2.5, linestyle='dotted'),
                mpf.make_addplot(wd['MA60'], color='red', width=2.5, linestyle='--'),
                mpf.make_addplot(wd['MA120'], color='green', width=2.5),
            ]
            fig, axes = mpf.plot(
                wd, type='candle', style='charles', volume=True,
                addplot=apds, returnfig=True,
                figsize=CONFIG['fig_size'],
                tight_layout=True, axisoff=True,
                ylim=ylim
            )
            ax_price = axes[0]
            if 'BB_lower' in wd.columns and 'BB_upper' in wd.columns:
                ax_price.fill_between(range(len(wd)), wd['BB_lower'].values, wd['BB_upper'].values, alpha=0.15, color='grey')
            fig.savefig(str(fpath), bbox_inches='tight', pad_inches=0,
                        dpi=CONFIG['fig_dpi'])
            plt.close(fig)
            generated += 1
        except Exception:
            plt.close('all')
            errors += 1

    return {'ticker': ticker, 'status': 'ok', 'generated': generated, 'skipped': skipped, 'errors': errors}


# === MAIN ===
if __name__ == '__main__':
    # 1. Fetch constituents
    constituents_path = META_DIR / 'sp500_constituents.json'
    if constituents_path.exists():
        with open(constituents_path) as f:
            constituents = json.load(f)
        print(f'Loaded {len(constituents)} constituents from cache')
    else:
        print('Fetching S&P 500 constituents from Wikipedia...')
        constituents = get_sp500_constituents()
        with open(constituents_path, 'w') as f:
            json.dump(constituents, f, indent=2)
        print(f'Fetched {len(constituents)} constituents')

    all_tickers = [c['ticker'] for c in constituents]
    ticker_sector = {c['ticker']: c['sector'] for c in constituents}

    # ========================================
    # PHASE 1: Check existing pickle files
    # ========================================
    print(f'\n{"="*50}')
    print('PHASE 1: Checking price data pickles')
    print(f'{"="*50}')

    ok_tickers = [t for t in all_tickers if (DATA_DIR / f'{t}.pkl').exists()]
    missing = [t for t in all_tickers if not (DATA_DIR / f'{t}.pkl').exists()]
    print(f'Pickle files found: {len(ok_tickers)} / {len(all_tickers)}')
    if missing:
        print(f'Missing: {len(missing)} — {missing[:10]}...')

    # ========================================
    # PHASE 2: Generate samples + parallel image generation
    # ========================================
    print(f'\n{"="*50}')
    print('PHASE 2: Generating samples & images (parallel)')
    print(f'{"="*50}')

    progress_path = META_DIR / 'progress.json'
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
    else:
        progress = {}

    # Generate samples for each stock (fast, sequential)
    image_tasks = []  # (ticker, samples) for parallel processing
    for ticker in ok_tickers:
        if ticker in progress and progress[ticker].get('status') == 'ok':
            continue  # Already done

        pkl_path = DATA_DIR / f'{ticker}.pkl'
        if not pkl_path.exists():
            continue

        df = pd.read_pickle(pkl_path)
        samples = generate_samples(df, ticker)
        if len(samples) < 50:
            progress[ticker] = {'ticker': ticker, 'status': 'too_few_samples', 'n_samples': len(samples)}
            continue

        train_samples, test_samples = temporal_split_samples(samples)

        # Save sample metadata
        meta = {
            'ticker': ticker,
            'n_total': len(samples),
            'n_train': len(train_samples),
            'n_test': len(test_samples),
            'class_dist_train': dict(Counter(s['label'] for s in train_samples)),
            'class_dist_test': dict(Counter(s['label'] for s in test_samples)),
            'samples': samples,
        }
        with open(META_DIR / f'samples_{ticker}.json', 'w') as f:
            json.dump(meta, f)

        image_tasks.append((ticker, samples))

    print(f'Stocks to process: {len(image_tasks)}')
    print(f'Already done: {sum(1 for v in progress.values() if v.get("status") == "ok")}')

    if not image_tasks:
        print('All images already generated!')
    else:
        # Parallel image generation
        t0 = time.time()
        chunk_size = CONFIG['n_workers'] * 2
        completed = 0
        total = len(image_tasks)

        for i in range(0, total, chunk_size):
            chunk = image_tasks[i:i+chunk_size]

            with Pool(CONFIG['n_workers']) as pool:
                results = pool.map(generate_images_for_stock_worker, chunk)

            for j, r in enumerate(results):
                ticker = chunk[j][0]
                samples = chunk[j][1]
                train_s, test_s = temporal_split_samples(samples)
                progress[ticker] = {
                    'ticker': ticker,
                    'status': r['status'],
                    'n_samples': len(samples),
                    'n_train': len(train_s),
                    'n_test': len(test_s),
                    'generated': r['generated'],
                    'skipped': r['skipped'],
                    'errors': r['errors'],
                }

            with open(progress_path, 'w') as f:
                json.dump(progress, f, indent=2)

            completed += len(chunk)
            ok_count = sum(1 for r in results if r.get('status') == 'ok')
            elapsed = time.time() - t0
            rate = completed / elapsed * 60
            eta = (total - completed) / max(rate, 0.1)
            print(f'  [{completed}/{total}] {ok_count}/{len(chunk)} OK | '
                  f'{elapsed:.0f}s | {rate:.1f}/min | ETA {eta:.0f}min')

        print(f'\nPhase 2 done: {time.time()-t0:.0f}s')

    # ========================================
    # SUMMARY
    # ========================================
    ok_stocks = {t: v for t, v in progress.items() if v.get('status') == 'ok'}
    failed_stocks = {t: v for t, v in progress.items() if v.get('status') != 'ok'}

    total_samples = sum(v.get('n_samples', 0) for v in ok_stocks.values())
    total_images = sum(v.get('generated', 0) + v.get('skipped', 0) for v in ok_stocks.values())

    print(f'\n{"="*50}')
    print(f'PIPELINE COMPLETE')
    print(f'Stocks OK: {len(ok_stocks)} / Failed: {len(failed_stocks)}')
    print(f'Total samples: {total_samples:,}')
    print(f'Total images: {total_images:,}')

    if failed_stocks:
        print(f'\nFailed:')
        for t, v in list(failed_stocks.items())[:20]:
            print(f'  {t}: {v.get("status")}')

    print(f'\n{"Sector":>30} {"Stocks":>7} {"Samples":>10}')
    print('-' * 50)
    sector_stats = {}
    for t, v in ok_stocks.items():
        s = ticker_sector.get(t, 'Unknown')
        if s not in sector_stats:
            sector_stats[s] = {'n': 0, 'samples': 0}
        sector_stats[s]['n'] += 1
        sector_stats[s]['samples'] += v.get('n_samples', 0)
    for s in sorted(sector_stats, key=lambda x: -sector_stats[x]['n']):
        print(f'{s:>30} {sector_stats[s]["n"]:>7} {sector_stats[s]["samples"]:>10,}')

    summary = {
        'n_stocks_ok': len(ok_stocks),
        'n_stocks_failed': len(failed_stocks),
        'total_samples': total_samples,
        'total_images': total_images,
        'sector_stats': sector_stats,
        'ok_tickers': list(ok_stocks.keys()),
    }
    with open(META_DIR / 'pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nOutput saved to: {BASE_DIR}')
    print(f'Next: Upload cross_sectional_data/ to Google Drive for Part 8')
