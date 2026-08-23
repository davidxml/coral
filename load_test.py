import requests
import concurrent.futures
import time
import statistics

API_URL = "http://127.0.0.1:8000/predict"
PAYLOAD = {"text": "Congratulations! Click here to claim your free university laptop."}

TOTAL_REQUESTS = 5000
CONCURRENT_WORKERS = 30  # much lower than 5000 - avoids client-side CPU contention


def send_request(request_id):
    """Sends a single request to the ML API and times it individually."""
    start = time.time()
    try:
        response = requests.post(API_URL, json=PAYLOAD, timeout=10)
        elapsed = time.time() - start
        return {
            "id": request_id,
            "status": response.status_code,
            "latency": elapsed,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "id": request_id,
            "status": None,
            "latency": elapsed,
            "error": str(e),
        }


def percentile(data, p):
    """Simple percentile calculation without extra dependencies."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def run_load_test(total_requests=TOTAL_REQUESTS, concurrent_workers=CONCURRENT_WORKERS):
    print(f"Firing {total_requests} requests with {concurrent_workers} concurrent workers...")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        results = list(executor.map(send_request, range(total_requests)))

    end_time = time.time()
    total_time = end_time - start_time

    latencies = [r["latency"] for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]
    success_count = len(latencies)

    print("-" * 40)
    print(f"Total requests:      {total_requests}")
    print(f"Successful:          {success_count}")
    print(f"Failed:              {len(errors)}")
    print(f"Total time:          {total_time:.4f} seconds")
    print(f"Throughput:          {success_count / total_time:.2f} requests/sec")

    if latencies:
        print("-" * 40)
        print("Per-request latency (seconds):")
        print(f"  min:    {min(latencies):.4f}")
        print(f"  mean:   {statistics.mean(latencies):.4f}")
        print(f"  median: {statistics.median(latencies):.4f}")
        print(f"  p95:    {percentile(latencies, 95):.4f}")
        print(f"  p99:    {percentile(latencies, 99):.4f}")
        print(f"  max:    {max(latencies):.4f}")

    if errors:
        print("-" * 40)
        print(f"First 5 errors:")
        for e in errors[:5]:
            print(f"  Request {e['id']}: {e['error']}")

    print("-" * 40)


if __name__ == "__main__":
    run_load_test()