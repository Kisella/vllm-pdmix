# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object

class HiddenChannelType:
    """Data-plane hidden tensor channel for edge-cloud PD separation.

    A generous PREFILL/DECODE pool is generated at import time. ``init`` is
    idempotent and can be called again during engine bootstrap.
    """

    _initialized: bool = False
    _MIN_PREFILL_POOL: int = 16
    _MIN_DECODE_POOL: int = 8

    def __init__(self, value: str) -> None:
        self._value = value

    @property
    def value(self) -> str:
        return self._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HiddenChannelType):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"<HiddenChannelType.{self._value}>"

    def __reduce__(self):
        return (_hidden_channel_from_value, (self._value,))

    @classmethod
    def init(
        cls,
        dp_size: int,
        prefill_per_dp: int = 2,
        decode_per_dp: int = 1,
    ) -> None:
        if cls._initialized:
            return
        cls._initialized = True

        need_prefill = dp_size * prefill_per_dp
        need_decode = dp_size * decode_per_dp
        prefill_pool = cls._MIN_PREFILL_POOL
        decode_pool = cls._MIN_DECODE_POOL
        while prefill_pool < need_prefill:
            prefill_pool *= 2
        while decode_pool < need_decode:
            decode_pool *= 2

        for i in range(1, prefill_pool + 1):
            setattr(cls, f"PREFILL_{i}", cls(f"prefill_{i}"))
        for i in range(1, decode_pool + 1):
            setattr(cls, f"DECODE_{i}", cls(f"decode_{i}"))
        setattr(cls, "DECODE", getattr(cls, "DECODE_1"))

    @staticmethod
    def prefill(i: int) -> "HiddenChannelType":
        return getattr(HiddenChannelType, f"PREFILL_{i}")

    @staticmethod
    def decode(i: int) -> "HiddenChannelType":
        return getattr(HiddenChannelType, f"DECODE_{i}")


HiddenChannelType.init(dp_size=8)


def _hidden_channel_from_value(value: str) -> HiddenChannelType:
    """Return a hidden-channel instance during pickle deserialization."""
    return HiddenChannelType(value)


class BatchType(enum.Enum):
    """Composition of a single SchedulerOutput batch.

    Tagged by the producing scheduler so that the non-leader PP rank's
    PassiveScheduler can route the batch without re-inspecting per-request
    state.

    - PD_MIX:        prefill and decode requests in the same batch (default
                     for the legacy mixed scheduler)
    - PURE_PREFILL:  every scheduled request is in its prefill phase
                     (legacy PP / PDmix mode)
    - PURE_DECODE:   every scheduled request is in its decode phase
                     (legacy PP / PDmix mode)
    - EMPTY:         no tokens scheduled this step (sync-only batch, e.g.
                     propagating finished_req_ids)
    - PREFILL_FIRST: edge-cloud PD separation — prefill batch executing the
                     edge head segment (segment_a)
    - PREFILL_LAST:  edge-cloud PD separation — prefill batch executing the
                     edge tail segment (segment_e + sampler)
    - DECODE_FIRST:  edge-cloud PD separation — decode batch executing the
                     edge head segment (Phase 4)
    - DECODE_LAST:   edge-cloud PD separation — decode batch executing the
                     edge tail segment (Phase 4)
    - PREFILL_DRAFT_FIRST: prefill-phase draft chain head (edge->cloud,
                     reuses the parent chunk's prefill channel)
    - PREFILL_DRAFT_LAST:  prefill-phase draft chain tail (cloud->edge,
                     same channel as its PREFILL_DRAFT_FIRST)
    - DECODE_DRAFT_FIRST:  decode-phase draft chain head (edge->cloud,
                     DECODE channel)
    - DECODE_DRAFT_LAST:   decode-phase draft chain tail (cloud->edge,
                     DECODE channel)
    """
    PD_MIX = "pd_mix"
    PURE_PREFILL = "pure_prefill"
    PURE_DECODE = "pure_decode"
    EMPTY = "empty"

    PREFILL_FIRST = "prefill_first"
    PREFILL_LAST = "prefill_last"
    DECODE_FIRST = "decode_first"
    DECODE_LAST = "decode_last"
    PREFILL_DRAFT_FIRST = "prefill_draft_first"
    PREFILL_DRAFT_LAST = "prefill_draft_last"
    DECODE_DRAFT_FIRST = "decode_draft_first"
    DECODE_DRAFT_LAST = "decode_draft_last"


@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None
    prompt_is_token_ids: list[bool] | None = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            prefill_token_ids=prefill_token_ids,
        )

    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@dataclass
class CachedRequestData:
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    # For requests not scheduled in the last step, propagate the token ids to the
    # connector. Won't contain requests that were scheduled in the prior step.
    # Values are np.ndarray(int32): pickle protocol 5 zerocopies them via
    # PickleBuffer, avoiding the per-int PyLong alloc that dominated the worker
    # dequeue (pickle.loads) cost under bench load.
    all_token_ids: dict[str, "npt.NDArray[np.int32]"]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    # Version of dataclass repr with token IDs obfuscated.
    def anon_repr(self) -> str:
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """Cache mapping of req_id to num_output_tokens for O(1) lookup.

        This cached property is safe because CachedRequestData instances
        are created fresh each scheduling iteration and not mutated during
        computation of iteration details.
        """
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    # Request IDs that are preempted in this step.
    # Only used for v2 model runner.
    preempted_req_ids: set[str] | None = None

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # Used for adjusting acceptance rate calculation.
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV Cache Connector metadata.
    kv_connector_metadata: KVConnectorMetadata | None = None

    # EC Cache Connector metadata
    ec_connector_metadata: ECConnectorMetadata | None = None

    # Block IDs freshly allocated from the pool during this scheduling step.
    # The worker zeros the corresponding GPU memory before the blocks are used,
    # preventing stale NaN/data from corrupting attention or SSM computation.
    new_block_ids_to_zero: list[int] | None = None
    
    # Composition of the scheduled batch. Producers (schedulers) tag this
    # field so downstream consumers — notably the non-leader PP rank's
    # PassiveScheduler — can route the batch without re-inspecting per-request
    # state. The base PD-mix Scheduler leaves it at the default (PD_MIX);
    # PDSeparatedScheduler overrides it to PURE_PREFILL / PURE_DECODE / EMPTY.
    batch_type: BatchType = BatchType.PD_MIX

    # Head-token for edge-cloud PD-separation pairing.
    # EngineCore assigns a UUID on PF/DF, cloud PassiveEngineCore echoes it
    # back on PL/DL via POST_OUT, and the edge worker embeds it into the
    # intermediate tensors payload so control-plane / data-plane alignment
    # can be verified before running the tail segment.
    head_token: str | None = None

    # Data-plane hidden tensor channel for edge-cloud PD separation. Prefill
    # head/tail batches use one of two prefill channels; decode uses the
    # dedicated decode channel. Scheduled speculative drafts also use the
    # decode channel. The cloud echoes this field back unchanged.
    hidden_channel: HiddenChannelType | None = None

    # Scheduled draft control-plane identity. Each draft step uses its own
    # head_token for edge/cloud pairing; the fields below identify the parent
    # request, the whole draft chain, and the current draft step. This is
    # shared by MTP and Eagle-style draft models.
    parent_req_id: str | None = None
    draft_task_id: str | None = None
    draft_step_idx: int | None = None

    # Rejection-corrected sampling state produced by the edge target step.
    # It is carried only by draft-FIRST step 0 so the cloud can update its
    # target/draft state before running the independently scheduled draft.
    num_accepted_tokens: list[int] | None = None
    valid_sampled_token_count: list[int] | None = None

    # Edge-side hint to the cloud PassiveScheduler. True means decode work is
    # active (or imminent), so interleaving prefill slices is worthwhile.
    # False denotes cold-start/pure-prefill and avoids slicing overhead.
    cloud_suggest_slicing: bool | None = None

    # Edge-cloud draft metadata invalidation. Task ids whose deferred draft
    # was dropped on the edge before the draft chain fully consumed the
    # cloud-side cached attention metadata (e.g. every request of the parent
    # verify batch finished). The cloud model runner purges the corresponding
    # cache entries. Produced only by the edge scheduler and consumed only by
    # the cloud model runner. Only tasks whose draft was never dispatched (or
    # was already fully consumed) are listed, so purging cannot race an
    # in-flight DRAFT batch.
    cloud_draft_invalidate_task_ids: list[str] | None = None

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            batch_type=BatchType.EMPTY,
        )


@dataclass
class GrammarOutput:
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"
