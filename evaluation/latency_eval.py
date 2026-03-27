import time

class LatencyEvaluator:

    def evaluate(self, rag_service, test_data):

        times = []

        for item in test_data:
            start = time.time()
            rag_service.process_query(item["question"])
            end = time.time()

            times.append(end - start)

        return {
            "avg_latency": sum(times) / len(times),
            "max_latency": max(times),
            "min_latency": min(times)
        }
