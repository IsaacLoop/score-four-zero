import queue
import threading
import time

import numpy as np
import torch


class GpuBatchEvaluator:
    """
    Owns the single training model during self-play and batches inference requests
    from several CPU workers into one GPU forward pass.
    """

    def __init__(
        self,
        model,
        request_queue,
        response_queues,
        max_batch_size: int = 1024,
        max_wait_s: float = 0.001,
    ):
        self.model = model
        self.request_queue = request_queue
        self.response_queues = tuple(response_queues)
        self.max_batch_size = int(max_batch_size)
        self.max_wait_s = float(max_wait_s)
        self.device = next(model.parameters()).device
        self._thread = None
        self._stats_lock = threading.Lock()
        self.total_queue_wait_s = 0.0
        self.total_forward_time_s = 0.0
        self.total_states = 0
        self.total_batches = 0

    def start(self):
        if self._thread is not None:
            return self

        self.model.eval()
        self._thread = threading.Thread(
            target=self._run,
            name="gpu-batch-evaluator",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self):
        if self._thread is None:
            return

        self.request_queue.put(None)
        self._thread.join()
        self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_stats(self):
        with self._stats_lock:
            return {
                "queue_wait_s": float(self.total_queue_wait_s),
                "forward_s": float(self.total_forward_time_s),
                "states": int(self.total_states),
                "batches": int(self.total_batches),
            }

    def _run(self):
        deferred_request = None
        stop_requested = False

        while True:
            pending_requests = []
            pending_state_count = 0

            if deferred_request is not None:
                pending_requests.append(deferred_request)
                pending_state_count = len(deferred_request[2])
                deferred_request = None

            if not pending_requests:
                request = self.request_queue.get()
                if request is None:
                    break
                pending_requests.append(request)
                pending_state_count = len(request[2])

            deadline = time.perf_counter() + self.max_wait_s

            while pending_state_count < self.max_batch_size:
                remaining_wait_s = deadline - time.perf_counter()
                if remaining_wait_s <= 0:
                    break

                try:
                    request = self.request_queue.get(timeout=remaining_wait_s)
                except queue.Empty:
                    break

                if request is None:
                    stop_requested = True
                    break

                request_state_count = len(request[2])
                if (
                    pending_requests
                    and pending_state_count + request_state_count > self.max_batch_size
                ):
                    deferred_request = request
                    break

                pending_requests.append(request)
                pending_state_count += request_state_count

            batch_started_at = time.perf_counter()
            x_batch = np.concatenate(
                [request_x_batch for _, _, request_x_batch in pending_requests],
                axis=0,
            )
            x_batch = torch.as_tensor(
                x_batch,
                dtype=torch.float32,
                device=self.device,
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            forward_started_at = time.perf_counter()
            with torch.inference_mode():
                policy_logits_batch, value_batch = self.model(x_batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            forward_elapsed_s = time.perf_counter() - forward_started_at

            policy_logits_batch = policy_logits_batch.detach().cpu().numpy()
            value_batch = value_batch.squeeze(-1).detach().cpu().numpy()

            with self._stats_lock:
                self.total_forward_time_s += forward_elapsed_s
                self.total_states += int(len(x_batch))
                self.total_batches += 1
                for _, request_enqueued_at, request_x_batch in pending_requests:
                    self.total_queue_wait_s += (
                        batch_started_at - request_enqueued_at
                    ) * len(request_x_batch)

            batch_offset = 0
            for worker_id, _, request_x_batch in pending_requests:
                request_state_count = len(request_x_batch)
                self.response_queues[worker_id].put(
                    (
                        policy_logits_batch[
                            batch_offset : batch_offset + request_state_count
                        ],
                        value_batch[
                            batch_offset : batch_offset + request_state_count
                        ],
                    )
                )
                batch_offset += request_state_count

            if stop_requested:
                break


class GpuBatchEvaluatorClient:
    """
    Small worker-side wrapper around the evaluator queues.
    Workers only see evaluate(x_batch) and timing counters.
    """

    def __init__(self, worker_id: int, request_queue, response_queue):
        self.worker_id = int(worker_id)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.total_wait_s = 0.0
        self.total_requests = 0
        self.total_states = 0

    def evaluate(self, x_batch):
        x_batch = np.ascontiguousarray(x_batch, dtype=np.float32)
        request_started_at = time.perf_counter()
        self.request_queue.put((self.worker_id, request_started_at, x_batch))
        policy_logits_batch, value_batch = self.response_queue.get()
        self.total_wait_s += time.perf_counter() - request_started_at
        self.total_requests += 1
        self.total_states += len(x_batch)
        return policy_logits_batch, value_batch

    def get_stats(self):
        return {
            "wait_s": float(self.total_wait_s),
            "requests": int(self.total_requests),
            "states": int(self.total_states),
        }
