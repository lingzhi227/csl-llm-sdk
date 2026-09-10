"""Executable protocol specification only; not a CSL or fabric qualification."""
from .wp16_diag_layout import EDGES

MAGIC_DATA = 0x4d4c5044
MAGIC_ACK = 0x4d4c5041
VERSION = 1
GUARDS = (0xa55aa55a, 0x5aa55aa5)
FIELDS = ('generation', 'input_index', 'contraction_id', 'edge_id', 'tensor_role',
          'channel_start', 'channel_count', 'output_start', 'output_count', 'payload_words')
CONTROL_FIELDS = ('generation', 'input_index', 'contraction_id', 'total_columns',
                  'tile_index', 'column_offset', 'valid_columns', 'output_start')
STATE_FIELDS = ('phase', 'generation', 'input_index', 'contraction_id', 'total_columns',
                'tile_count', 'consumed_columns', 'commit_id', 'error', 'last_valid_columns',
                'accumulate_count', 'begin_count', 'finalize_count', 'release_count',
                'pe_role', 'output_start')
HANDOFF_FIELDS = ('phase', 'generation', 'input_index', 'contraction_id', 'edge_id',
                 'committed_payload_mask', 'command_seen', 'receipt_complete', 'ack_sent',
                 'ack_received', 'data_send_complete', 'command_unblocked', 'error',
                 'data_words', 'ack_words', 'commit_count', 'event_counter', 'arm_event',
                 'command_event', 'receipt_event', 'commit_event', 'ack_sent_event',
                 'ack_received_event', 'unblock_event', 'tensor_role', 'channel_start',
                 'channel_count', 'output_start', 'output_count', 'release_count',
                 'receipt_before_command', 'retained')
MLP_STATE_FIELDS = ('phase', 'generation', 'input_index', 'contraction_id', 'incoming_mask',
                    'silu_count', 'product_count', 'down_input_count', 'down_tile_count',
                    'down_consumed', 'output_count', 'complete_epoch', 'error',
                    'release_count', 'pe_role', 'reserved_zero')


def descriptor(edge):
    if type(edge) is not int or not 0 <= edge < 3:
        raise ValueError('Named diagnostic edge required')
    return (1, 0, 1, edge, EDGES[edge]['role'], 0, 128, 0, 128, 128)


def words32(words):
    if any(type(value) is not int or not 0 <= value <= 0xffffffff for value in words):
        raise ValueError('Exact unsigned32 integer words required')


def packet(edge, body):
    if len(body) != 128:
        raise ValueError('Exactly128 payload words required')
    words32(body)
    if any(value >> 16 or ((value >> 7) & 255) == 255 for value in body):
        raise ValueError('Finite BF16 payload in low16 bits required')
    return (GUARDS[0], MAGIC_DATA, VERSION, *descriptor(edge), *body, GUARDS[1])


def acknowledgement(latched, error):
    return (GUARDS[0], MAGIC_ACK, VERSION, *latched, error, GUARDS[1])


def verify_ack(edge, words):
    words32(words)
    if tuple(words) != acknowledgement(descriptor(edge), 0):
        raise ValueError('Exact identity-matching successful ACK required before sender release')


class Receiver:
    """Two joins and atomic numerical commit, using tiny integer-word arrays."""
    def __init__(self, pe):
        if pe not in (2, 3):
            raise ValueError('Only nonlinear and down receivers exist')
        self.pe = pe
        self.allowed = (0, 1) if pe == 2 else (2,)
        self.committed = {}
        self.next_index = 0
        self.latched = None
        self.command_seen = self.ack_sent = False
        self.receipt_seen = False
        self.error = 0
        self.events = []

    @property
    def command_complete(self):
        return self.command_seen and self.receipt_seen and self.ack_sent

    def arm(self, control):
        if self.latched is not None and not self.command_complete:
            raise ValueError('Previous command/receipt/ACK join is incomplete')
        if self.error or self.next_index >= len(self.allowed):
            raise ValueError('Receiver is failed or already complete')
        words32(control)
        expected = descriptor(self.allowed[self.next_index])
        if tuple(control) != expected:
            raise ValueError('Stale, duplicate, reordered or wrong-scope descriptor')
        self.latched = tuple(control)
        self.command_seen = self.receipt_seen = self.ack_sent = False
        self.events.append('armed')

    def command(self):
        if self.latched is None or self.command_seen:
            raise ValueError('Exactly one command for an armed receipt required')
        self.command_seen = True
        self.events.append('command')

    def receive(self, words):
        if self.latched is None or self.receipt_seen:
            raise ValueError('Exactly one complete frame for an armed receipt required')
        # Examine the complete frame before any numerical payload is published.
        error = 0
        try:
            words32(words)
            if len(words) != 142 or words[0] != GUARDS[0] or words[-1] != GUARDS[1]:
                error = 1
            elif (words[1], words[2]) != (MAGIC_DATA, VERSION):
                error = 2
            elif tuple(words[3:13]) != self.latched:
                error = 3
            elif any(value >> 16 or ((value >> 7) & 255) == 255 for value in words[13:141]):
                error = 4
        except ValueError:
            error = 5  # Host specification only: fabric words are already typed u32.
        self.receipt_seen = True
        self.events.append('receipt')
        if error:
            self.error = error
        else:
            edge = self.latched[3]
            self.committed[edge] = tuple(words[13:141])
            self.next_index += 1
            self.events.append('commit')
        return acknowledgement(self.latched, error)

    def acknowledge_sent(self):
        if not self.receipt_seen or self.ack_sent:
            raise ValueError('ACK send completion must follow exactly one receipt')
        self.ack_sent = True
        self.events.append('ack_sent')

    @property
    def ready_for_compute(self):
        return not self.error and self.command_complete and self.next_index == len(self.allowed)
