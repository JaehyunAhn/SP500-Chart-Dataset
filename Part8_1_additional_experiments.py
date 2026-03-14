"""
Part 8-1: Additional Experiments
Append these cells to Part 8 notebook after Stage 2 completes.

Prerequisites: Part 8 Cells 1-5 already executed
(sector_tickers, _build_file_label_lists, make_tf_dataset,
 build_vgg16_feature_extractor, eval_predictions, RESULTS_DIR defined)
"""

# =====================================================================
# CELL A: 3-Class & Binary Classification Function Definitions
# =====================================================================

# Binary mapping: 6-class → down(0)/up(1)
# 3-class mapping: strong_down(0) / mild(1) / strong_up(2)
# Also tests label smoothing on original 6-class

CLASS_NAMES_BINARY = ['down', 'up']
CLASS_NAMES_3 = ['strong_down', 'mild', 'strong_up']

def map_6_to_binary(labels):
    """6-class → binary: 0,1,2 → 0(down), 3,4,5 → 1(up)"""
    return (np.array(labels) >= 3).astype(np.int32)

def map_6_to_3class(labels):
    """6-class → 3-class: 0→0(strong_down), 1,2,3,4→1(mild), 5→2(strong_up)"""
    labels = np.array(labels)
    out = np.ones(len(labels), dtype=np.int32)  # default: mild(1)
    out[labels == 0] = 0   # down_3plus → strong_down
    out[labels == 5] = 2   # up_3plus → strong_up
    return out

def build_vgg16_ft_block5_custom(n_classes):
    """VGG16 block5 fine-tune, custom number of classes"""
    base = VGG16(weights='imagenet', include_top=False,
                 input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base.layers:
        layer.trainable = 'block5' in layer.name
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_vgg16_label_smoothing(n_classes=6, smoothing=0.1):
    """VGG16 with label smoothing loss"""
    base = VGG16(weights='imagenet', include_top=False,
                 input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base.layers:
        layer.trainable = 'block5' in layer.name
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing),
        metrics=['accuracy'])
    return model

print('Part 8-1 functions defined: binary, 3-class, label smoothing')


# =====================================================================
# CELL B: Experiment 1 — Binary (Down/Up) Classification by Sector
# =====================================================================

print('='*70)
print('EXPERIMENT 1: Binary Classification (Down/Up) by Sector')
print('='*70)

binary_results_path = RESULTS_DIR / 'stage2_binary_results.json'

if binary_results_path.exists():
    with open(binary_results_path) as f:
        binary_results = json.load(f)
else:
    binary_results = {}

all_sectors = sorted(sector_tickers.keys())
todo = [s for s in all_sectors if s not in binary_results]

print(f'Done: {len(binary_results)}, Remaining: {len(todo)}')

t0_all = time.time()

for si, sector in enumerate(todo):
    t0 = time.time()
    n_stocks = len(sector_tickers[sector])
    print(f'\n[{si+1}/{len(todo)}] {sector}  ({n_stocks} stocks)')

    tickers = sector_tickers[sector]
    train_paths, train_labels_6, _ = _build_file_label_lists(tickers, 'train')
    test_paths, test_labels_6, _ = _build_file_label_lists(tickers, 'test')

    if len(train_paths) < 50 or len(test_paths) < 10:
        binary_results[sector] = {'status': 'insufficient_data'}
        continue

    # Map to binary
    y_train = map_6_to_binary(train_labels_6)
    y_test = map_6_to_binary(test_labels_6)

    print(f'  Train: {len(y_train):,} (up={y_train.mean()*100:.1f}%)  '
          f'Test: {len(y_test):,} (up={y_test.mean()*100:.1f}%)')

    result = {
        'sector': sector, 'n_stocks': n_stocks,
        'n_train': len(y_train), 'n_test': len(y_test),
        'test_up_ratio': round(y_test.mean() * 100, 2),
        'status': 'ok',
    }

    # Majority baseline (binary)
    up_ratio = y_test.mean()
    maj_bl = max(up_ratio, 1 - up_ratio)
    result['majority_baseline'] = round(maj_bl * 100, 2)

    # Class weights
    cw = compute_class_weights_from_labels(y_train)

    # ---- VGG16 binary ----
    print('  Training VGG16 (binary)...')
    try:
        tf.random.set_seed(42); np.random.seed(42)
        val_n = max(int(len(train_paths) * 0.15), 32)

        ds_tr = make_tf_dataset(train_paths, y_train[:-val_n].tolist(),
                                batch_size=64, shuffle=True)
        ds_vl = make_tf_dataset(train_paths[-val_n:], y_train[-val_n:].tolist(),
                                batch_size=64, shuffle=False)
        ds_te = make_tf_dataset(test_paths, y_test.tolist(),
                                batch_size=64, shuffle=False)

        model = build_vgg16_ft_block5_custom(n_classes=2)
        history = model.fit(
            ds_tr, epochs=50, validation_data=ds_vl,
            class_weight=cw,
            callbacks=[
                EarlyStopping('val_loss', patience=5,
                              restore_best_weights=True, verbose=1),
                ReduceLROnPlateau('val_loss', factor=0.5, patience=2, verbose=1),
            ],
            verbose=1,
        )
        y_pred = np.argmax(model.predict(ds_te, verbose=0), axis=1)

        acc = accuracy_score(y_test, y_pred) * 100
        f1m = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100
        n_correct = int((y_test == y_pred).sum())
        p_maj = stats.binomtest(n_correct, len(y_test), maj_bl, 'greater').pvalue

        result['vgg16'] = {
            'acc': round(acc, 2), 'f1_macro': round(f1m, 2),
            'majority_baseline': round(maj_bl * 100, 2),
            'p_vs_majority': float(p_maj),
            'epochs_run': len(history.history['loss']),
            'pred_dist': dict(Counter(y_pred.tolist())),
        }
        print(f'    Binary VGG16: acc={acc:.2f}% (bl={maj_bl*100:.1f}%) p={p_maj:.4f}')

        # Extract features for XGBoost
        feat_ext = build_vgg16_feature_extractor()
        ds_train_all = make_tf_dataset(train_paths, y_train.tolist(),
                                        batch_size=64, shuffle=False)
        feat_tr = feat_ext.predict(ds_train_all, verbose=0)
        feat_te = feat_ext.predict(ds_te, verbose=0)

        del model, feat_ext
        tf.keras.backend.clear_session(); gc.collect()
    except Exception as e:
        result['vgg16'] = {'status': f'error: {str(e)[:120]}'}
        print(f'    ERROR: {str(e)[:120]}')
        tf.keras.backend.clear_session(); gc.collect()
        feat_tr, feat_te = None, None

    # ---- XGBoost binary on GAP features ----
    print('  Training XGBoost (binary)...')
    try:
        if feat_tr is not None:
            clf = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.01,
                objective='binary:logistic',
                random_state=42, use_label_encoder=False,
                eval_metric='logloss', tree_method='hist', n_jobs=-1,
            )
            sw = np.array([cw[y] for y in y_train]) if cw else None
            clf.fit(feat_tr, y_train, sample_weight=sw, verbose=False)
            y_pred_xgb = clf.predict(feat_te)

            acc_x = accuracy_score(y_test, y_pred_xgb) * 100
            f1m_x = f1_score(y_test, y_pred_xgb, average='macro', zero_division=0) * 100
            n_correct_x = int((y_test == y_pred_xgb).sum())
            p_maj_x = stats.binomtest(n_correct_x, len(y_test), maj_bl, 'greater').pvalue

            result['xgboost'] = {
                'acc': round(acc_x, 2), 'f1_macro': round(f1m_x, 2),
                'p_vs_majority': float(p_maj_x),
                'pred_dist': dict(Counter(y_pred_xgb.tolist())),
            }
            print(f'    Binary XGB: acc={acc_x:.2f}% p={p_maj_x:.4f}')
            del clf, feat_tr, feat_te; gc.collect()
        else:
            result['xgboost'] = {'status': 'skipped'}
    except Exception as e:
        result['xgboost'] = {'status': f'error: {str(e)[:120]}'}

    elapsed = time.time() - t0
    result['elapsed_sec'] = round(elapsed, 1)
    print(f'  Done in {elapsed:.0f}s')

    binary_results[sector] = result
    with open(binary_results_path, 'w') as f:
        json.dump(binary_results, f, indent=2)

print(f'\nBinary experiment complete: {time.time()-t0_all:.0f}s')


# =====================================================================
# CELL C: Experiment 2 — 3-Class (Strong Down / Mild / Strong Up)
# =====================================================================

print('='*70)
print('EXPERIMENT 2: 3-Class (Strong Down / Mild / Strong Up) by Sector')
print('='*70)

three_class_path = RESULTS_DIR / 'stage2_3class_results.json'

if three_class_path.exists():
    with open(three_class_path) as f:
        three_class_results = json.load(f)
else:
    three_class_results = {}

todo = [s for s in all_sectors if s not in three_class_results]
print(f'Done: {len(three_class_results)}, Remaining: {len(todo)}')

t0_all = time.time()

for si, sector in enumerate(todo):
    t0 = time.time()
    n_stocks = len(sector_tickers[sector])
    print(f'\n[{si+1}/{len(todo)}] {sector}  ({n_stocks} stocks)')

    tickers = sector_tickers[sector]
    train_paths, train_labels_6, _ = _build_file_label_lists(tickers, 'train')
    test_paths, test_labels_6, _ = _build_file_label_lists(tickers, 'test')

    if len(train_paths) < 50 or len(test_paths) < 10:
        three_class_results[sector] = {'status': 'insufficient_data'}
        continue

    # Map to 3-class
    y_train = map_6_to_3class(train_labels_6)
    y_test = map_6_to_3class(test_labels_6)

    dist_train = dict(Counter(y_train.tolist()))
    dist_test = dict(Counter(y_test.tolist()))
    print(f'  Train dist: {dist_train}  Test dist: {dist_test}')

    result = {
        'sector': sector, 'n_stocks': n_stocks,
        'n_train': len(y_train), 'n_test': len(y_test),
        'train_class_dist': dist_train,
        'test_class_dist': dist_test,
        'status': 'ok',
    }

    maj_cls, maj_cnt = Counter(y_test.tolist()).most_common(1)[0]
    maj_bl = maj_cnt / len(y_test)
    result['majority_baseline'] = round(maj_bl * 100, 2)

    cw = compute_class_weights_from_labels(y_train)

    # ---- VGG16 3-class ----
    print('  Training VGG16 (3-class)...')
    try:
        tf.random.set_seed(42); np.random.seed(42)
        val_n = max(int(len(train_paths) * 0.15), 32)

        ds_tr = make_tf_dataset(train_paths, y_train[:-val_n].tolist(),
                                batch_size=64, shuffle=True)
        ds_vl = make_tf_dataset(train_paths[-val_n:], y_train[-val_n:].tolist(),
                                batch_size=64, shuffle=False)
        ds_te = make_tf_dataset(test_paths, y_test.tolist(),
                                batch_size=64, shuffle=False)

        model = build_vgg16_ft_block5_custom(n_classes=3)
        history = model.fit(
            ds_tr, epochs=50, validation_data=ds_vl,
            class_weight=cw,
            callbacks=[
                EarlyStopping('val_loss', patience=5,
                              restore_best_weights=True, verbose=1),
                ReduceLROnPlateau('val_loss', factor=0.5, patience=2, verbose=1),
            ],
            verbose=1,
        )
        y_pred = np.argmax(model.predict(ds_te, verbose=0), axis=1)

        metrics = eval_predictions(y_test, y_pred, maj_bl)
        metrics['epochs_run'] = len(history.history['loss'])
        result['vgg16'] = metrics
        print(f'    3-class VGG16: acc={metrics["acc"]}% f1m={metrics["f1_macro"]}%')

        # GAP features
        feat_ext = build_vgg16_feature_extractor()
        ds_train_all = make_tf_dataset(train_paths, y_train.tolist(),
                                        batch_size=64, shuffle=False)
        feat_tr = feat_ext.predict(ds_train_all, verbose=0)
        feat_te = feat_ext.predict(ds_te, verbose=0)

        del model, feat_ext
        tf.keras.backend.clear_session(); gc.collect()
    except Exception as e:
        result['vgg16'] = {'status': f'error: {str(e)[:120]}'}
        tf.keras.backend.clear_session(); gc.collect()
        feat_tr, feat_te = None, None

    # ---- XGBoost 3-class ----
    print('  Training XGBoost (3-class)...')
    try:
        if feat_tr is not None:
            clf = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.01,
                objective='multi:softmax', num_class=3,
                random_state=42, use_label_encoder=False,
                eval_metric='mlogloss', tree_method='hist', n_jobs=-1,
            )
            sw = np.array([cw[y] for y in y_train]) if cw else None
            clf.fit(feat_tr, y_train, sample_weight=sw, verbose=False)
            y_pred_xgb = clf.predict(feat_te)

            metrics_xgb = eval_predictions(y_test, y_pred_xgb, maj_bl)
            result['xgboost'] = metrics_xgb
            print(f'    3-class XGB: acc={metrics_xgb["acc"]}% f1m={metrics_xgb["f1_macro"]}%')
            del clf, feat_tr, feat_te; gc.collect()
        else:
            result['xgboost'] = {'status': 'skipped'}
    except Exception as e:
        result['xgboost'] = {'status': f'error: {str(e)[:120]}'}

    elapsed = time.time() - t0
    result['elapsed_sec'] = round(elapsed, 1)
    print(f'  Done in {elapsed:.0f}s')

    three_class_results[sector] = result
    with open(three_class_path, 'w') as f:
        json.dump(three_class_results, f, indent=2)

print(f'\n3-class experiment complete: {time.time()-t0_all:.0f}s')


# =====================================================================
# CELL D: Experiment 3 — Label Smoothing on 6-Class
# =====================================================================

print('='*70)
print('EXPERIMENT 3: 6-Class with Label Smoothing (0.1) by Sector')
print('='*70)

ls_results_path = RESULTS_DIR / 'stage2_label_smoothing_results.json'

if ls_results_path.exists():
    with open(ls_results_path) as f:
        ls_results = json.load(f)
else:
    ls_results = {}

todo = [s for s in all_sectors if s not in ls_results]
print(f'Done: {len(ls_results)}, Remaining: {len(todo)}')

t0_all = time.time()

for si, sector in enumerate(todo):
    t0 = time.time()
    n_stocks = len(sector_tickers[sector])
    print(f'\n[{si+1}/{len(todo)}] {sector}  ({n_stocks} stocks)')

    tickers = sector_tickers[sector]
    train_paths, train_labels, _ = _build_file_label_lists(tickers, 'train')
    test_paths, test_labels, _ = _build_file_label_lists(tickers, 'test')

    if len(train_paths) < 50 or len(test_paths) < 10:
        ls_results[sector] = {'status': 'insufficient_data'}
        continue

    y_train = np.array(train_labels, dtype=np.int32)
    y_test = np.array(test_labels, dtype=np.int32)

    result = {
        'sector': sector, 'n_stocks': n_stocks,
        'n_train': len(y_train), 'n_test': len(y_test),
        'status': 'ok',
    }

    maj_cls, maj_cnt = Counter(y_test.tolist()).most_common(1)[0]
    maj_bl = maj_cnt / len(y_test)
    result['majority_baseline'] = round(maj_bl * 100, 2)

    # Need one-hot for label smoothing loss
    cw = compute_class_weights_from_labels(y_train)

    # ---- VGG16 with label smoothing ----
    print('  Training VGG16 (label_smoothing=0.1)...')
    try:
        tf.random.set_seed(42); np.random.seed(42)
        val_n = max(int(len(train_paths) * 0.15), 32)

        # Label smoothing requires one-hot labels
        def _parse_image_onehot(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            img = tf.cast(img, tf.float32) / 255.0
            label_oh = tf.one_hot(label, NUM_CLASSES)
            return img, label_oh

        ds_tr = make_tf_dataset(train_paths[:-val_n], train_labels[:-val_n],
                                batch_size=64, shuffle=True,
                                parse_fn=_parse_image_onehot)
        ds_vl = make_tf_dataset(train_paths[-val_n:], train_labels[-val_n:],
                                batch_size=64, shuffle=False,
                                parse_fn=_parse_image_onehot)
        ds_te = make_tf_dataset(test_paths, test_labels,
                                batch_size=64, shuffle=False)  # normal for predict

        model = build_vgg16_label_smoothing(n_classes=NUM_CLASSES, smoothing=0.1)
        history = model.fit(
            ds_tr, epochs=50, validation_data=ds_vl,
            class_weight=cw,
            callbacks=[
                EarlyStopping('val_loss', patience=5,
                              restore_best_weights=True, verbose=1),
                ReduceLROnPlateau('val_loss', factor=0.5, patience=2, verbose=1),
            ],
            verbose=1,
        )
        y_pred = np.argmax(model.predict(ds_te, verbose=0), axis=1)

        metrics = eval_predictions(y_test, y_pred, maj_bl)
        metrics['epochs_run'] = len(history.history['loss'])
        metrics['label_smoothing'] = 0.1
        result['vgg16_ls'] = metrics
        print(f'    LS VGG16: acc={metrics["acc"]}% f1m={metrics["f1_macro"]}% '
              f'classes={metrics["n_pred_classes"]}/6')

        del model
        tf.keras.backend.clear_session(); gc.collect()
    except Exception as e:
        result['vgg16_ls'] = {'status': f'error: {str(e)[:120]}'}
        tf.keras.backend.clear_session(); gc.collect()

    elapsed = time.time() - t0
    result['elapsed_sec'] = round(elapsed, 1)
    print(f'  Done in {elapsed:.0f}s')

    ls_results[sector] = result
    with open(ls_results_path, 'w') as f:
        json.dump(ls_results, f, indent=2)

print(f'\nLabel smoothing experiment complete: {time.time()-t0_all:.0f}s')


# =====================================================================
# CELL E: Comparison Summary — 6-class vs 3-class vs Binary vs LS
# =====================================================================

print('='*70)
print('COMPARISON: Classification Granularity × Sector')
print('='*70)

# Load all results
with open(RESULTS_DIR / 'stage2_sector_results.json') as f:
    res_6class = json.load(f)

res_binary = {}
if (RESULTS_DIR / 'stage2_binary_results.json').exists():
    with open(RESULTS_DIR / 'stage2_binary_results.json') as f:
        res_binary = json.load(f)

res_3class = {}
if (RESULTS_DIR / 'stage2_3class_results.json').exists():
    with open(RESULTS_DIR / 'stage2_3class_results.json') as f:
        res_3class = json.load(f)

res_ls = {}
if (RESULTS_DIR / 'stage2_label_smoothing_results.json').exists():
    with open(RESULTS_DIR / 'stage2_label_smoothing_results.json') as f:
        res_ls = json.load(f)

# Build comparison table
rows = []
for sector in sorted(sector_tickers.keys()):
    row = {'sector': sector}

    # 6-class VGG16
    r6 = res_6class.get(sector, {})
    if r6.get('status') == 'ok' and 'acc' in r6.get('vgg16', {}):
        row['6c_vgg_acc'] = r6['vgg16']['acc']
        row['6c_vgg_f1'] = r6['vgg16']['f1_macro']
        row['6c_maj_bl'] = r6['majority_baseline']
    # 6-class XGBoost
    if r6.get('status') == 'ok' and 'acc' in r6.get('xgboost', {}):
        row['6c_xgb_acc'] = r6['xgboost']['acc']

    # Binary VGG16
    rb = res_binary.get(sector, {})
    if rb.get('status') == 'ok' and 'acc' in rb.get('vgg16', {}):
        row['bin_vgg_acc'] = rb['vgg16']['acc']
        row['bin_maj_bl'] = rb['majority_baseline']
        row['bin_p'] = rb['vgg16'].get('p_vs_majority', 1.0)

    # Binary XGBoost
    if rb.get('status') == 'ok' and 'acc' in rb.get('xgboost', {}):
        row['bin_xgb_acc'] = rb['xgboost']['acc']

    # 3-class VGG16
    r3 = res_3class.get(sector, {})
    if r3.get('status') == 'ok' and 'acc' in r3.get('vgg16', {}):
        row['3c_vgg_acc'] = r3['vgg16']['acc']
        row['3c_maj_bl'] = r3['majority_baseline']

    # Label smoothing VGG16
    rl = res_ls.get(sector, {})
    if rl.get('status') == 'ok' and 'acc' in rl.get('vgg16_ls', {}):
        row['ls_vgg_acc'] = rl['vgg16_ls']['acc']
        row['ls_vgg_f1'] = rl['vgg16_ls']['f1_macro']
        row['ls_npred'] = rl['vgg16_ls']['n_pred_classes']

    rows.append(row)

df_compare = pd.DataFrame(rows)
print(df_compare.to_string(index=False))

# Save
df_compare.to_csv(RESULTS_DIR / 'granularity_comparison.csv', index=False)
with open(RESULTS_DIR / 'granularity_comparison.json', 'w') as f:
    json.dump(rows, f, indent=2)

print(f'\nComparison saved to {RESULTS_DIR / "granularity_comparison.csv"}')

# ---- Key insight: does simpler classification help? ----
print('\n=== KEY FINDINGS ===')
if 'bin_vgg_acc' in df_compare.columns and '6c_vgg_acc' in df_compare.columns:
    valid = df_compare.dropna(subset=['bin_vgg_acc', '6c_vgg_acc'])
    if len(valid) > 0:
        print(f'Avg 6-class VGG16 acc: {valid["6c_vgg_acc"].mean():.1f}%')
        print(f'Avg Binary VGG16 acc:  {valid["bin_vgg_acc"].mean():.1f}%')
        sig_binary = valid[valid.get('bin_p', 1.0) < 0.05] if 'bin_p' in valid.columns else pd.DataFrame()
        print(f'Sectors where binary beats majority (p<0.05): {len(sig_binary)}/11')


# =====================================================================
# CELL F: Visualization — Granularity Comparison
# =====================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

sectors = df_compare['sector'].tolist()
x = np.arange(len(sectors))
width = 0.2

# Panel 1: VGG16 Accuracy across granularities
ax = axes[0]
if '6c_vgg_acc' in df_compare.columns:
    ax.barh(x - width, df_compare.get('6c_vgg_acc', 0), width,
            label='6-class', color='steelblue', alpha=0.8)
if '3c_vgg_acc' in df_compare.columns:
    ax.barh(x, df_compare.get('3c_vgg_acc', 0), width,
            label='3-class', color='coral', alpha=0.8)
if 'bin_vgg_acc' in df_compare.columns:
    ax.barh(x + width, df_compare.get('bin_vgg_acc', 0), width,
            label='Binary', color='forestgreen', alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(sectors, fontsize=8)
ax.set_xlabel('Accuracy (%)')
ax.set_title('VGG16 Accuracy: 6-class vs 3-class vs Binary')
ax.legend()

# Panel 2: Label smoothing effect
ax = axes[1]
if '6c_vgg_f1' in df_compare.columns:
    ax.barh(x - width/2, df_compare.get('6c_vgg_f1', 0), width,
            label='6-class (no LS)', color='steelblue', alpha=0.8)
if 'ls_vgg_f1' in df_compare.columns:
    ax.barh(x + width/2, df_compare.get('ls_vgg_f1', 0), width,
            label='6-class + Label Smoothing', color='orange', alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(sectors, fontsize=8)
ax.set_xlabel('F1-Macro (%)')
ax.set_title('Effect of Label Smoothing on F1-Macro')
ax.legend()

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / 'granularity_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Granularity comparison chart saved')
