# Hivedex Troubleshooting Guide

Solutions to common issues.

---

## Installation Issues

### "No module named 'X'"

**Solution:** Install missing dependency
```bash
pip install -r requirements.txt
```

Or install specific package:
```bash
pip install pandas altair vaderSentiment
```

---

### "yfinance not available" warning

**Cause:** Architecture mismatch on some Macs (arm64 vs x86_64)

**Impact:** Stock price fetching disabled, but core features work

**Solution:** This is optional. Ignore if not using stock data, or try:
```bash
pip uninstall yfinance
pip install --no-cache-dir yfinance
```

---

### Import errors in scripts

**Cause:** Running from wrong directory

**Solution:** Run from project root:
```bash
cd /path/to/Hivedex
python scripts/test_suite.py
```

Or add to path:
```python
import sys
sys.path.insert(0, '/path/to/Hivedex')
```

---

## API Issues

### Arctic Shift returns empty data

**Possible causes:**
1. Invalid subreddit name
2. No posts match keywords
3. Date range too narrow
4. API rate limiting

**Solutions:**
```python
# Check subreddit exists
subreddits = ["wallstreetbets"]  # Not "r/wallstreetbets"

# Broaden date range
start_date = "2024-01-01"
end_date = "2024-12-31"

# Use simpler keywords
keywords = ["NVDA"]  # Not "NVDA earnings Q3 2024"

# Add delay between requests
import time
time.sleep(1)  # 1 second delay
```

---

### GDELT returns no articles

**Possible causes:**
1. Keywords too specific
2. Date range outside 3-month limit
3. Network timeout

**Solutions:**
```python
# Simplify keywords
keywords = ["nvidia"]  # Not "nvidia data center AI chips"

# Check date range (GDELT has ~90 day limit)
from datetime import datetime, timedelta
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
```

---

### Rate limiting errors

**Solution:** Add delays between requests
```python
import time

for event in events:
    result = process_event(event)
    time.sleep(0.5)  # 500ms delay
```

---

## Data Issues

### "Event not found"

**Cause:** Event ID doesn't match catalog

**Solution:** Check available events:
```python
import pandas as pd
events = pd.read_csv('data/events_catalog.csv', comment='#')
print(events['event_id'].tolist())
```

---

### Accuracy is 0% or 100%

**Cause:** Likely data loading issue

**Solution:** Verify data:
```python
import pandas as pd
df = pd.read_csv('data/validation_results.csv')
print(df['prediction_correct'].value_counts())
```

---

### NaN values in results

**Cause:** Missing data or calculation errors

**Solution:** Check for missing columns:
```python
print(df.isnull().sum())
```

---

## Visualization Issues

### Charts not displaying

**In Jupyter:**
```python
import altair as alt
alt.renderers.enable('default')

# Or for JupyterLab
alt.renderers.enable('mimetype')
```

**Saving to file:**
```python
chart.save('chart.html')
```

---

### "altair not found"

**Solution:**
```bash
pip install altair
```

---

### Chart too large / crashes browser

**Solution:** Limit data points:
```python
alt.data_transformers.disable_max_rows()

# Or sample data
df_sample = df.sample(n=1000)
```

---

## Test Failures

### Run test suite

```bash
python scripts/test_suite.py
```

### Common failures

| Test | Likely Cause | Fix |
|------|-------------|-----|
| Events catalog loads | Missing file | Check data/ directory |
| Categories valid | New category added | Update valid set |
| Accuracy reasonable | Bad data | Regenerate validations |
| Imports | Missing package | pip install |

---

## Hex Deployment Issues

### Files not uploading

**Solution:** Check file sizes (Hex has limits)
- Max file size: Check Hex docs
- Use CSV instead of Parquet if issues

---

### Notebook errors in Hex

**Solution:**
1. Check relative paths (use '../data/' or 'data/')
2. Ensure all imports at top of notebook
3. Test locally first

---

### Threads AI not working

**Solution:**
1. Enable Threads in project settings
2. Import semantic model correctly
3. Check entity/metric definitions

---

## Performance Issues

### Slow processing

**Solutions:**
1. Enable caching:
```python
fetch_reddit_posts(..., use_cache=True)
```

2. Limit events:
```python
process_all_events(limit=10)
```

3. Process specific categories:
```python
process_all_events(categories=['movie', 'stock'])
```

---

### Memory errors

**Solution:** Process in batches:
```python
for i in range(0, len(events), 10):
    batch = events[i:i+10]
    process_batch(batch)
```

---

## Getting Help

### Debug mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check versions

```bash
python --version
pip list | grep -E "pandas|altair|vader"
```

### Verify installation

```bash
python scripts/test_suite.py
```

---

## Still stuck?

1. Check existing documentation
2. Search error message online
3. Review API documentation
4. Check example outputs in EXAMPLES.md

---

*Troubleshooting guide v1.0*
