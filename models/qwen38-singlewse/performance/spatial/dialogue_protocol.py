"""Single-conversation admission oracle for future device-driven serving.

This models acknowledgement ordering, not neural computation or device speed.
The caller must supply real complete-stage retirement events when integrated.
Selected output and committed state are deliberately separate: a stopped final
token stays pending and is consumed once before a subsequent prompt suffix.
"""
from dataclasses import dataclass
from spatial.pipeline_protocol import Tag


@dataclass(frozen=True)
class DialogueInput:
    tag: Tag
    token: int


class DialogueAdmission:
    def __init__(self, context, vocabulary=248320):
        if type(context) is not int or context < 2 or vocabulary != 248320:
            raise ValueError('Explicit context and original vocabulary required')
        self.context = context
        self.vocabulary = vocabulary
        self.generation = 0
        self.initialized = False
        self.tokens = []
        self.committed = 0
        self.live = None
        self.active = False
        self.turn = 0
        self.prompt_end = 0
        self.output_limit = 0
        self.output = []

    def _token(self, token):
        if type(token) is not int or not 0 <= token < self.vocabulary:
            raise ValueError('Original token ID required')

    def _all_stages(self, acknowledgements):
        values = tuple(acknowledgements)
        if (len(values) != 66 or any(type(v) is not int for v in values)
                or set(values) != set(range(66))):
            raise ValueError('All66 unique actual stage acknowledgements required')

    def reset(self, generation, *, drained_state_clear_acks):
        if self.live is not None or self.active or type(generation) is not int or generation != self.generation+1:
            raise ValueError('Reset requires idle state and the next generation')
        self._all_stages(drained_state_clear_acks)
        self.generation = generation
        self.initialized = True
        self.tokens.clear()
        self.committed = 0
        self.turn = 0
        self.prompt_end = 0
        self.output_limit = 0
        self.output = []

    def begin_turn(self, full_tokenized_prefix, *, max_output_tokens):
        """Accept the exact original-template prefix including actual prior output."""
        prefix = tuple(full_tokenized_prefix)
        if not self.initialized or self.active or self.live is not None:
            raise ValueError('Turn admission requires initialized, drained prior work')
        for token in prefix:
            self._token(token)
        if (len(prefix) <= len(self.tokens) or prefix[:len(self.tokens)] != tuple(self.tokens)
                or type(max_output_tokens) is not int or max_output_tokens <= 0):
            raise ValueError('New input must extend the exact committed/selected prefix')
        if len(prefix)+max_output_tokens > self.context:
            raise ValueError('Declared conversation capacity exceeded; no truncation or implicit reset')
        self.tokens = list(prefix)
        self.prompt_end = len(prefix)
        self.output_limit = max_output_tokens
        self.output = []
        self.turn += 1
        self.active = True

    def admit(self):
        """Offer the oldest uncommitted token, including a prior pending final ID."""
        if not self.active:
            raise ValueError('No active turn')
        if self.live is not None:
            return None  # Explicit dependent-token backpressure.
        if len(self.output) >= self.output_limit:
            raise ValueError('Output limit reached; end the turn before accepting more input')
        if not 0 <= self.committed < len(self.tokens) <= self.context:
            raise ValueError('Missing dependent input or invalid committed position')
        self.live = DialogueInput(Tag(0, self.generation, self.committed), self.tokens[self.committed])
        return self.live

    def complete(self, offered, *, retired_stage_acks, selected_token=None, content=None):
        """Commit only after all stages update state and retire this input's work."""
        if self.live is None or offered != self.live:
            raise ValueError('Stale, duplicate, or wrong input completion')
        self._all_stages(retired_stage_acks)
        produces_output = self.committed+1 >= self.prompt_end
        if produces_output:
            self._token(selected_token)
            if type(content) is not bool or len(self.tokens) >= self.context or len(self.output) >= self.output_limit:
                raise ValueError('Classified output within the reserved capacity required')
        elif selected_token is not None or content is not None:
            raise ValueError('Intermediate prefix positions cannot emit assistant output')
        self.committed += 1
        self.live = None
        if produces_output:
            self.tokens.append(selected_token)
            self.output.append((selected_token, content))
        return selected_token

    def end_turn(self):
        """Keep conversation state and final selected token; release no state here."""
        if not self.active or self.live is not None or self.committed < self.prompt_end or not self.output:
            raise ValueError('Turn end requires a completed prompt and fully retired output work')
        self.active = False
        return dict(turn=self.turn, committed_tokens=self.committed,
                    selected_prefix_tokens=len(self.tokens), pending_token=self.tokens[self.committed],
                    output_ids=[token for token, _ in self.output],
                    content_tokens=sum(is_content for _, is_content in self.output))
