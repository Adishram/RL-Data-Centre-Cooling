# Data directory

## Alibaba Cluster Trace 2017

This project uses the **Alibaba Cluster Trace 2017** `server_usage.csv`.

### Schema

| Column | Description |
|--------|-------------|
| `timestamp` | Unix timestamp of the measurement (seconds) |
| `machine_id` | Unique identifier for the physical machine |
| `cpu_util` | CPU utilization (0–100) |
| `mem_util` | Memory utilization (0–100) |
| `disk_util` | Disk I/O utilization (0–100) |
| `load1` | 1-minute load average |
| `load5` | 5-minute load average |
| `load15` | 15-minute load average |

Measurements are taken at ~60-second intervals, averaged over 300 seconds.

### How to obtain

1. Visit https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2017
2. Download only the `server_usage.csv` portion (or a slice of it)
3. Place the file in `data/raw/server_usage.csv`
4. Run `python -m data.prepare_dataset`

### Synthetic fallback

If the real dataset is unavailable, run:

```bash
python -m data.prepare_dataset --synthetic
```

This generates `data/mini_alibaba.csv` with realistic but **synthetic** utilisation
patterns. The synthetic data is clearly labelled and never passed off as real.
