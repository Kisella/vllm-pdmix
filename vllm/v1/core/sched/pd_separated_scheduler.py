# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import time
from collections import deque
from collections.abc import Iterable
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import BatchType, SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class SchedulingPhase(enum.Enum):
    PREFILL_FIRST = "prefill_first"
    PREFILL_LAST = "prefill_last"
    DECODE = "decode"


class PDSeparatedScheduler(Scheduler):
    """Scheduler that separates prefill and decode into distinct steps.

    In edge-cloud PD-separated mode the four cardinal phases are:
      - PREFILL_FIRST  (edge head segment)
      - PREFILL_LAST   (edge tail segment, sourced from cloud-returned outputs)
      - DECODE_FIRST   (Phase 4)
      - DECODE_LAST    (Phase 4)

    This class owns the request bookkeeping for *first* segments
    (``chunk_prefill_first`` + parent's ``waiting`` / ``running``) and the
    ready queues for *last* segments (``prefills_last_ready`` /
    ``decodes_last_ready``), which are filled by the EngineCore from the
    POST_OUT channel before each ``schedule()`` call.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Requests that have started their P-first segment but have not yet
        # been fully consumed (still chunking, or still in flight on cloud).
        self.chunk_prefill_first: list[Request] = []

        # SchedulerOutputs returned from cloud (POST_OUT channel) carrying
        # the metadata needed to execute the edge tail segment.
        # Populated by EngineCore.step() before calling self.schedule().
        self.prefills_last_ready: deque[SchedulerOutput] = deque()
        self.decodes_last_ready: deque[SchedulerOutput] = deque()

        self._step_counter: int = 0

        # In-flight prefill limit (head-segment batches).
        self.prefill_inflight_limit: int = getattr(
            self.scheduler_config, "pd_prefill_inflight_limit", 1
        )
        self.prefill_inflight_count: int = 0

    def schedule(self) -> SchedulerOutput:
        return self._schedule_pd_separated()

    def _schedule_pd_separated(self) -> SchedulerOutput:
        phase = self._select_scheduling_phase()
        # Enforce prefill inflight limit: if a new PREFILL_FIRST would exceed
        # the limit, fall back to decode or emit an empty batch.
        if (
            phase == SchedulingPhase.PREFILL_FIRST
            and self.prefill_inflight_count >= self.prefill_inflight_limit
        ):
            if self.running:
                phase = SchedulingPhase.DECODE
            else:
                return SchedulerOutput.make_empty()
        self._step_counter += 1
        print(
            f"\r\n[PD] Step{self._step_counter}, phase is {phase.value},    "
            f"waiting[]: {len(self.waiting)}, "
            f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}, "
            f"running[]: {len(self.running)}, "
            f"prefills_last_ready[]: {len(self.prefills_last_ready)}, "
            f"decodes_last_ready[]: {len(self.decodes_last_ready)}, "
            f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}"
        )
        for req in self.chunk_prefill_first:
            print(
                f"[PD] chunk_prefill_first[{req.request_id}],    "
                f"num_prompt_tokens: {req.num_prompt_tokens}, "
                f"num_tokens: {req.num_tokens}, "
                f"num_computed_tokens: {req.num_computed_tokens}, "
                f"chunk_num: {req.chunk_num}"
            )
        for req in self.running:
            print(
                f"[PD] running[{req.request_id}],    "
                f"num_prompt_tokens: {req.num_prompt_tokens}, "
                f"num_tokens: {req.num_tokens}, "
                f"num_computed_tokens: {req.num_computed_tokens}, "
                f"chunk_num: {req.chunk_num}"
            )
        if phase == SchedulingPhase.PREFILL_LAST:
            return self._pick_prefill_last_batch()
        if phase == SchedulingPhase.PREFILL_FIRST:
            if not self.chunk_prefill_first and not self.waiting:
                print(
                    "[PD] prefill_first phase but no prefill work, "
                    "auto-switch to decode"
                )
                return self._pick_decode_batch()
            return self._pick_prefill_first_batch()
        # SchedulingPhase.DECODE
        if not self.running:
            print(
                "[PD] decode phase but no decode work, "
                "auto-switch to prefill_first"
            )
            return self._pick_prefill_first_batch()
        return self._pick_decode_batch()

    def _select_scheduling_phase(self) -> SchedulingPhase:
        # Phase 3: PREFILL_LAST always wins when there is cloud-returned work.
        # Releasing edge KV early by sampling first keeps backpressure low.
        if self.prefills_last_ready:
            return SchedulingPhase.PREFILL_LAST

        policy = self.scheduler_config.pd_scheduling_policy
        if policy == "prefill_first":
            if self.chunk_prefill_first or self.waiting:
                return SchedulingPhase.PREFILL_FIRST
            if self.running:
                return SchedulingPhase.DECODE
            return SchedulingPhase.PREFILL_FIRST
        elif policy == "decode_first":
            if self.running:
                return SchedulingPhase.DECODE
            if self.chunk_prefill_first or self.waiting:
                return SchedulingPhase.PREFILL_FIRST
            return SchedulingPhase.DECODE
        elif policy == "strict_alternation":
            return (
                SchedulingPhase.PREFILL_FIRST
                if self._step_counter % 2 == 0
                else SchedulingPhase.DECODE
            )
        else:
            raise ValueError(f"Unknown PD scheduling policy: {policy}")

    def _pick_prefill_first_batch(self) -> SchedulerOutput:
        saved_running = self.running
        saved_chunk_prefill_first = self.chunk_prefill_first
        saved_max_num_running_reqs = self.max_num_running_reqs

        self.running = list(saved_chunk_prefill_first)
        self.chunk_prefill_first = []
        self.max_num_running_reqs -= len(saved_running)

        scheduler_output = None
        try:
            scheduler_output = super().schedule()
        finally:
            self.max_num_running_reqs = saved_max_num_running_reqs
            if scheduler_output is not None:
                if scheduler_output.total_num_scheduled_tokens == 0:
                    scheduler_output.batch_type = BatchType.EMPTY
                else:
                    scheduler_output.batch_type = BatchType.PREFILL_FIRST
                    self.prefill_inflight_count += 1
                new_chunk_prefill_first = [
                    req for req in self.running if req.is_prefill_chunk
                ]
                new_running = [
                    req for req in self.running if not req.is_prefill_chunk
                ]
                for req in self.chunk_prefill_first:
                    if req not in new_chunk_prefill_first:
                        new_chunk_prefill_first.append(req)
                self.chunk_prefill_first = new_chunk_prefill_first
                self.running = saved_running + new_running
                print(
                    f"[PD] _pick_prefill_first_batch done: "
                    f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}, "
                    f"running[]: {len(self.running)}, "
                    f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}"
                )
                for (
                    req_id,
                    num_scheduled_token,
                ) in scheduler_output.num_scheduled_tokens.items():
                    req = self.requests[req_id]
                    print(
                        f"[PD] Scheduled[{req_id}],    "
                        f"num_tokens: {req.num_tokens}, "
                        f"num_scheduled_token: {num_scheduled_token}, "
                        f"num_computed_tokens: {req.num_computed_tokens}, "
                        f"is_prefill_chunk: {req.is_prefill_chunk}, "
                        f"chunk_num: {req.chunk_num}"
                    )
            else:
                self.chunk_prefill_first = saved_chunk_prefill_first
                self.running = saved_running

        return scheduler_output  # type: ignore[return-value]

    def _pick_prefill_last_batch(self) -> SchedulerOutput:
        """Pop one cloud-returned SchedulerOutput from prefills_last_ready.

        The cloud has already rewritten ``batch_type=PREFILL_LAST`` and kept
        all original KV / sampling metadata intact, so the edge worker can
        directly run segment_e + sampler on it. We also remove the involved
        requests from ``chunk_prefill_first`` so the parent class's
        ``update_from_output`` does not double-account them.
        """
        if not self.prefills_last_ready:
            return SchedulerOutput.make_empty()
        so = self.prefills_last_ready.popleft()
        assert so.batch_type == BatchType.PREFILL_LAST, (
            f"prefills_last_ready expects PREFILL_LAST, got {so.batch_type}"
        )
        # Drop these reqs from chunk_prefill_first; the edge has now received
        # the cloud round-trip and is about to sample.
        last_req_ids = set(so.num_scheduled_tokens.keys())
        if last_req_ids:
            self.chunk_prefill_first = [
                req for req in self.chunk_prefill_first
                if req.request_id not in last_req_ids
            ]
        print(
            f"[PD] _pick_prefill_last_batch popped {len(last_req_ids)} reqs; "
            f"remaining prefills_last_ready[]: {len(self.prefills_last_ready)}"
        )
        return so

    def _pick_decode_batch(self) -> SchedulerOutput:
        saved_chunk_prefill_first = self.chunk_prefill_first
        saved_waiting = self.waiting
        saved_skipped = self.skipped_waiting

        self.chunk_prefill_first = []
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)

        scheduler_output = None
        try:
            scheduler_output = super().schedule()
        finally:
            if scheduler_output is not None:
                if scheduler_output.total_num_scheduled_tokens == 0:
                    scheduler_output.batch_type = BatchType.EMPTY
                else:
                    scheduler_output.batch_type = BatchType.PURE_DECODE
                for req in list(self.waiting):
                    saved_waiting.prepend_request(req)
                self.chunk_prefill_first = saved_chunk_prefill_first
                self.waiting = saved_waiting
                self.skipped_waiting = saved_skipped
                print(
                    f"[PD] _pick_decode_batch done: "
                    f"running: {len(self.running)}, "
                    f"chunk_prefill_first: {len(self.chunk_prefill_first)}"
                )
                for (
                    req_id,
                    num_scheduled_token,
                ) in scheduler_output.num_scheduled_tokens.items():
                    req = self.requests[req_id]
                    print(
                        f"[PD] Scheduled[{req_id}],    "
                        f"num_tokens: {req.num_tokens}, "
                        f"num_scheduled_token: {num_scheduled_token}, "
                        f"num_computed_tokens: {req.num_computed_tokens}, "
                        f"is_prefill_chunk: {req.is_prefill_chunk}, "
                        f"chunk_num: {req.chunk_num}"
                    )
            else:
                self.chunk_prefill_first = saved_chunk_prefill_first
                self.waiting = saved_waiting
                self.skipped_waiting = saved_skipped

        return scheduler_output  # type: ignore[return-value]

    def _migrate_prefill_to_running(self) -> None:
        completed = [
            req for req in self.chunk_prefill_first if not req.is_prefill_chunk
        ]
        if completed:
            print(
                f"[PD] _migrate_prefill_to_running: moving {len(completed)} "
                f"requests from chunk_prefill_first to running"
            )
        for req in completed:
            self.chunk_prefill_first.remove(req)
            self.running.append(req)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_preemptions += 1
        if request.spec_token_ids:
            request.spec_token_ids = []
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        if request.is_prefill_chunk:
            print(
                f"[PD] _preempt_request: request {request.request_id} "
                f"stays in chunk_prefill_first "
                f"(computed={request.num_computed_tokens})"
            )
            self.chunk_prefill_first.append(request)
        else:
            print(
                f"[PD] _preempt_request: request {request.request_id} "
                f"goes back to waiting (decode or finished prefill)"
            )
            request.num_computed_tokens = 0
            self.waiting.prepend_request(request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        was_prefill_map = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            was_prefill_map[req_id] = self.requests[req_id].is_prefill_chunk

        super()._update_after_schedule(scheduler_output)

        for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
            if was_prefill_map[req_id] and num_scheduled_token > 0:
                self.requests[req_id].chunk_num += 1
                print(
                    f"[PD] _update_after_schedule: request {req_id} "
                    f"chunk_num={self.requests[req_id].chunk_num} "
                    f"tokens={self.requests[req_id].num_tokens} "
                    f"scheduled={num_scheduled_token} "
                    f"computed={self.requests[req_id].num_computed_tokens} "
                    f"is_prefill_chunk={self.requests[req_id].is_prefill_chunk}"
                )

        self._migrate_prefill_to_running()
        self.finished_req_ids = set()

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, Any]:
        if scheduler_output.batch_type == BatchType.PREFILL_LAST:
            if self.prefill_inflight_count > 0:
                self.prefill_inflight_count -= 1
            print(
                f"[PD] update_from_output PREFILL_LAST done, "
                f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}"
            )
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        self.chunk_prefill_first = [
            req for req in self.chunk_prefill_first if not req.is_finished()
        ]
        return outputs

    def get_request_counts(self) -> tuple[int, int]:
        num_running, num_waiting = super().get_request_counts()
        return num_running + len(self.chunk_prefill_first), num_waiting

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        return super().get_num_unfinished_requests() + len(
            self.chunk_prefill_first
        )

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        result = super().finish_requests(request_ids, finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        to_remove = set()
        for req_id in request_ids:
            req = self.requests.get(req_id)
            if req and req.is_finished():
                to_remove.add(req)

        if to_remove:
            self.chunk_prefill_first = remove_all(
                self.chunk_prefill_first, to_remove
            )

        return result

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        if reset_running_requests:
            timestamp = time.monotonic()
            while self.chunk_prefill_first:
                request = self.chunk_prefill_first.pop()
                self.kv_cache_manager.free(request)
                self.encoder_cache_manager.free(request)
                request.status = RequestStatus.PREEMPTED
                request.num_computed_tokens = 0
                if request.spec_token_ids:
                    request.spec_token_ids = []
                request.num_preemptions += 1
                if self.log_stats:
                    request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True
                self.waiting.prepend_request(request)

        return super().reset_prefix_cache(reset_running_requests, reset_connector)

    def make_stats(self, *args, **kwargs):
        stats = super().make_stats(*args, **kwargs)
        if stats is not None:
            stats.num_running_reqs += len(self.chunk_prefill_first)
        return stats

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        saved_running = self.running
        self.running = list(self.running) + [
            r for r in self.chunk_prefill_first if r not in self.running
        ]
        try:
            result = super()._handle_invalid_blocks(invalid_block_ids)
        finally:
            self.running = saved_running
        return result


class AsyncPDSeparatedScheduler(AsyncScheduler, PDSeparatedScheduler):
    """Async scheduler with PD separation."""
    pass
