"""
Prometheus metrics for the compute node.

All prometheus_client imports are guarded so the module loads cleanly
even when the optional ``prometheus_client`` package is not installed.
"""

METRICS_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram

    jobs_claimed_total = Counter(
        "node_jobs_claimed_total",
        "Total number of jobs claimed by this node",
    )
    jobs_completed_total = Counter(
        "node_jobs_completed_total",
        "Total number of jobs completed successfully",
    )
    jobs_failed_total = Counter(
        "node_jobs_failed_total",
        "Total number of jobs that failed after all retries",
    )
    active_jobs = Gauge(
        "node_active_jobs",
        "Number of jobs currently being processed",
    )
    queue_depth = Gauge(
        "node_queue_depth",
        "Number of jobs waiting in the priority queue",
    )
    inference_latency_seconds = Histogram(
        "node_inference_latency_seconds",
        "End-to-end inference latency from claim to on-chain submission",
        buckets=[1, 5, 10, 30, 60, 120],
    )
    ipfs_upload_duration_seconds = Histogram(
        "node_ipfs_upload_duration_seconds",
        "Time taken to upload inference result to IPFS",
    )
    heartbeat_timestamp = Gauge(
        "node_heartbeat_timestamp",
        "Unix epoch timestamp of the last successful heartbeat",
    )
    node_reputation = Gauge(
        "node_reputation",
        "On-chain reputation score of this node",
    )

    METRICS_AVAILABLE = True

except Exception:  # pragma: no cover — only fires when prometheus_client missing
    pass
