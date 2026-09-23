"""Finite full-model stage identities, bounded control slices, and reload order.

This module specifies executable host operations without creating a runtime.
Actual device source/compiled binding, durable checkpoint and full-vocabulary
implementations must satisfy these contracts before any physical admission.
"""
from dataclasses import dataclass

MODEL = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
STAGE_RANGES = ((0, 20), (20, 44), (44, 64))
CONTEXT_CAPACITY = 8
HIDDEN_WORDS = 5120
VOCABULARY = 248320
MAX_COPY_HOST_BYTES = 16 << 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Stage:
    stage_id: int
    layers: tuple[int, ...]
    input_kind: str
    output_kind: str

    @property
    def first_layer(self):
        return self.layers[0]

    @property
    def last_layer(self):
        return self.layers[-1]


def stages():
    return tuple(Stage(i, tuple(range(start, stop)),
                       'original_token_embedding' if i == 0 else 'previous_stage_actual_hidden',
                       'actual_hidden_boundary' if i < 2 else 'original_final_norm_full_vocabulary')
                 for i, (start, stop) in enumerate(STAGE_RANGES))


def control_slices(width, height=1160):
    """Partition the all-PE eight-word control bank below the existing copy cap."""
    require(type(width) is int and 0 < width <= 750 and height == 1160, 'Qualified application dimensions')
    rows = MAX_COPY_HOST_BYTES // (width * 8 * 4)
    require(rows > 0, 'At least one complete application row fits')
    return [dict(symbol='checkpoint_control', dtype='u32', bits=32, x=0, y=y,
                 width=width, height=min(rows, height-y), count=8)
            for y in range(0, height, rows)]


@dataclass(frozen=True)
class Context:
    ordinal: int
    stage: Stage
    positions: tuple[int, ...]
    request_id: int
    generation: int
    restore_next_position: int | None
    input_kind: str
    checkpoint_next_position: int
    terminal_output: bool
    reset_before_compute: bool = False

    @property
    def restore_generation(self):
        return self.generation - 1 if self.reset_before_compute else self.generation


def context_record(context):
    """Canonical small identity used by capture, audit and retention receipts."""
    return dict(ordinal=context.ordinal,stage_id=context.stage.stage_id,
                layers=list(context.stage.layers),positions=list(context.positions),
                request_id=context.request_id,generation=context.generation,
                restore_next_position=context.restore_next_position,
                restore_generation=context.restore_generation,input_kind=context.input_kind,
                checkpoint_next_position=context.checkpoint_next_position,
                terminal_output=context.terminal_output,reset_before_compute=context.reset_before_compute)


def contexts(prompt_tokens, output_tokens=4, request_id=1, generation=1, reset_replay=True):
    """Prefill five rows per stage; then reload all three stages per decode row.

    Tokens after prefill are inputs from the previous actual full-head argmax.
    Reference tokens are deliberately absent from the execution contract.
    """
    require(type(request_id) is int and 0 < request_id < 2**32, 'Bounded request identity')
    require(type(generation) is int and 0 < generation < 2**32, 'Bounded logical generation')
    require(isinstance(prompt_tokens, (tuple, list)) and len(prompt_tokens) == 5 and
            all(type(t) is int and 0 <= t < VOCABULARY for t in prompt_tokens), 'Five original prompt token IDs')
    require(type(output_tokens) is int and 1 <= output_tokens <= 4, 'Finite one-to-four actual output tokens')
    require(len(prompt_tokens) + output_tokens - 1 <= CONTEXT_CAPACITY, 'Context8 includes every computed position')
    result = []
    all_stages = stages()
    for stage in all_stages:
        result.append(Context(len(result), stage, tuple(range(5)), request_id, generation,
                              None, stage.input_kind, 5, stage.stage_id == 2))
    for position in range(5, 5 + output_tokens - 1):
        for stage in all_stages:
            result.append(Context(len(result), stage, (position,), request_id, generation,
                                  position, 'actual_token_feedback_embedding' if stage.stage_id == 0 else stage.input_kind,
                                  position+1, stage.stage_id == 2))
    require(type(reset_replay) is bool, 'Explicit reset replay scope')
    if reset_replay:
        require(generation < 2**32-1, 'Reset generation does not wrap')
        for stage in all_stages:
            result.append(Context(len(result), stage, tuple(range(5)), request_id, generation+1,
                                  5+output_tokens-1, stage.input_kind, 5, stage.stage_id == 2,
                                  reset_before_compute=True))
    return tuple(result)


class TransitionLedger:
    """Require released owners and causal payload/checkpoint identities at reload.

    Receipts are produced by independently bounded device owners. This host
    ledger never substitutes its own hidden vectors, state arrays or token IDs.
    """
    def __init__(self, expected, prompt_tokens):
        self.expected = tuple(expected)
        require(len(prompt_tokens)==5 and all(type(t) is int and 0<=t<VOCABULARY for t in prompt_tokens),
                'Ledger binds the exact original five prompt tokens')
        self.prompt_tokens = tuple(prompt_tokens)
        self.index = 0
        self.active = None
        self.checkpoints = {}
        self.boundaries = {}
        self.output_tokens = []
        self.reset_token = None
        self.initial_logits = None

    def begin(self, context, checkpoint=None, boundary=None, feedback_token=None):
        require(self.active is None and self.index < len(self.expected) and context == self.expected[self.index],
                'One physical context in exact original stage/position order')
        stage = context.stage.stage_id
        require(context.request_id == self.expected[0].request_id and context.generation == self.expected[0].generation + int(context.reset_before_compute),
                'Explicit unchanged generation or device reset increment')
        if context.restore_next_position is None:
            require(checkpoint is None and stage not in self.checkpoints, 'Fresh stage with empty valid state')
        else:
            require(checkpoint is not None and checkpoint == self.checkpoints.get(stage) and
                    checkpoint['next_position'] == context.restore_next_position and
                    checkpoint['generation'] == context.restore_generation, 'Exact own-stage durable checkpoint')
        if stage > 0:
            require(boundary is not None and boundary == self.boundaries.get((context.generation, stage-1, context.positions)) and
                    boundary['source_stage'] == stage-1 and boundary['destination_stage'] == stage and
                    boundary['positions'] == list(context.positions), 'Exact preceding-stage actual hidden batch')
        else:
            require(boundary is None, 'Stage0 consumes an original embedding')
            if context.restore_next_position is not None and not context.reset_before_compute:
                require(self.output_tokens and type(feedback_token) is int and feedback_token == self.output_tokens[-1],
                        'Decode embedding token is the preceding actual argmax')
            else:
                require(feedback_token is None, 'Initial prefill prompt is fixed separately')
        self.active = context

    def input_expectation(self, context):
        """Issue causal identities only for the currently admitted transition."""
        require(context == self.active, 'Input expectation belongs to the active exact context')
        stage = context.stage.stage_id
        common = dict(request_id=context.request_id, generation=context.generation,
                      positions=list(context.positions), destination_stage=stage)
        if stage == 0:
            tokens = (self.output_tokens[-1],) if context.restore_next_position is not None and not context.reset_before_compute else self.prompt_tokens
            require(len(tokens)==len(context.positions), 'Every input row has one causal token')
            return dict(common,kind='original_embedding_rows',token_ids=list(tokens))
        boundary = self.boundaries[context.generation,stage-1,context.positions]
        return dict(common,kind='actual_device_stage_boundary',boundary_receipt=dict(boundary))

    def finish(self, release, checkpoint, boundary=None, argmax=None):
        context = self.active
        require(context is not None, 'An exact context is active')
        stage = context.stage.stage_id
        require(release['child_reaped'] and release['groups_empty'] and release['owned_jobs_released'] and
                release['actual_owned_assignments'] == [], 'Actual previous device owner released before replacement')
        if context.reset_before_compute:
            require(checkpoint['device_reset_performed'] and checkpoint['reset_from_generation'] == context.generation-1 and
                    checkpoint['reset_to_generation'] == context.generation and checkpoint['reset_valid_position'] == 0,
                    'Actual reset occurs after restoring old state; a fresh constructor is insufficient')
        require(checkpoint['stage_id'] == stage and checkpoint['layers'] == list(context.stage.layers) and
                checkpoint['request_id'] == context.request_id and checkpoint['generation'] == context.generation and
                checkpoint['next_position'] == context.checkpoint_next_position and checkpoint['all_semantic_banks_complete'] and
                checkpoint['durable'] and len(checkpoint['sha256']) == 64, 'Complete own-stage state at the global fence')
        if stage < 2:
            require(argmax is None and boundary is not None and boundary['source_stage'] == stage and
                    boundary['destination_stage'] == stage+1 and boundary['positions'] == list(context.positions) and
                    boundary['request_id'] == context.request_id and boundary['generation'] == context.generation and
                    boundary['dtype'] == '<u2' and boundary['shape'] == [len(context.positions), HIDDEN_WORDS] and
                    boundary['actual_device_hidden'] and boundary['durable'] and len(boundary['sha256']) == 64,
                    'Complete durable actual BF16 stage output batch')
            if context.reset_before_compute:
                original = self.boundaries.get((context.generation-1, stage, context.positions))
                require(original is not None and all(boundary[k] == original[k] for k in ['sha256', 'dtype', 'shape']),
                        'Reset reproduces every BF16 value of the original stage prefill boundary')
            self.boundaries[context.generation, stage, context.positions] = dict(boundary)
        else:
            require(boundary is None and argmax is not None and argmax['position'] == context.positions[-1] and
                    argmax['request_id'] == context.request_id and argmax['generation'] == context.generation and
                    argmax['vocabulary_size'] == VOCABULARY and argmax['all_logits_captured'] and
                    argmax['original_final_norm_and_head'] and argmax['logits_dtype'] == '<u2' and
                    argmax['logits_shape'] == [VOCABULARY] and type(argmax['token_id']) is int and
                    0 <= argmax['token_id'] < VOCABULARY and len(argmax['logits_sha256']) == 64,
                    'Token selected from the complete actual vocabulary output')
            if context.reset_before_compute:
                require(argmax['token_id'] == self.output_tokens[0] and self.initial_logits is not None and
                        all(argmax[k] == self.initial_logits[k] for k in ['logits_sha256', 'logits_dtype', 'logits_shape']),
                        'Reset prefill reproduces every original full-vocabulary logit')
                self.reset_token = argmax['token_id']
            else:
                if not self.output_tokens:
                    self.initial_logits = {k: argmax[k] for k in ['logits_sha256', 'logits_dtype', 'logits_shape']}
                self.output_tokens.append(argmax['token_id'])
        self.checkpoints[stage] = dict(checkpoint)
        self.active = None
        self.index += 1

    @property
    def complete(self):
        return self.active is None and self.index == len(self.expected)
