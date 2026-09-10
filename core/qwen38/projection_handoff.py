"""WP14 frozen scalar metadata and host protocol model; no device qualification."""
from dataclasses import dataclass, field

SLABS = (0, 1, 4, 5)
DATA_MAGIC = 0x514B4441
ACK_MAGIC = 0x514B414B
VERSION = 1
DATA_WORDS = 138
ACK_WORDS = 8
WIRE_GUARDS = (0xA55AA55A, 0x5AA55AA5)


def descriptor(rank, generation=1, token=1):
    if type(rank) is not int or not 0 <= rank < 4:
        raise ValueError('Physical rank must be0..3')
    if any(type(v) is not int or not 1 <= v < 2**32 for v in (generation, token)):
        raise ValueError('Positive u32 generation/token required')
    return [generation, token, rank, SLABS[rank], rank//2, (rank % 2)*128, 128, rank*128]


def validate_descriptor(values):
    if len(values) != 8 or values != descriptor(values[2], values[0], values[1]):
        raise ValueError('Projection role/slab/destination mismatch')


def data_frame(control, words):
    validate_descriptor(control)
    if len(words) != 128 or any(type(v) is not int or not 0 <= v <= 65535 for v in words):
        raise ValueError('Exactly128 zero-extended BF16 payload words required')
    return [DATA_MAGIC, VERSION, *control, *words]


def parse_frame(frame, expected):
    validate_descriptor(expected)
    if len(frame) != DATA_WORDS or frame[:10] != [DATA_MAGIC, VERSION, *expected]:
        raise ValueError('Complete fixed data header required')
    payload = frame[10:]
    if any(type(v) is not int or not 0 <= v <= 65535 for v in payload):
        raise ValueError('Payload high16 must be zero')
    return payload


def acknowledgement(control, status=0):
    validate_descriptor(control)
    if type(status) is not int or not 0 <= status < 2**32:
        raise ValueError('ACK status is a u32')
    gen, token, rank, slab, _, _, _, destination = control
    return [ACK_MAGIC, VERSION, gen, token, rank, slab, destination, status]


@dataclass
class ConsumerModel:
    """Checks atomic commit and receive/command/ACK joins without assuming order."""
    generation: int = 1
    token: int = 1
    mask: int = 0
    words: list = field(default_factory=lambda: [None]*512)
    armed: list | None = None
    command_arrived: bool = False
    committed: bool = False
    ack_complete: bool = False
    unblocks: int = 0

    def arm(self, control):
        validate_descriptor(control)
        if self.armed is not None or control[:2] != [self.generation, self.token] or self.mask & (1 << control[2]):
            raise ValueError('Busy/stale/duplicate handoff')
        self.armed = list(control)
        self.command_arrived = self.committed = self.ack_complete = False

    def command(self):
        if self.armed is None or self.command_arrived:
            raise ValueError('Unarmed/duplicate command')
        self.command_arrived = True
        self._join()

    def receive(self, frame):
        if self.armed is None or self.committed:
            raise ValueError('Unarmed/duplicate receive')
        payload = parse_frame(frame, self.armed)
        rank, offset = self.armed[2], self.armed[7]
        self.words[offset:offset+128] = payload
        self.mask |= 1 << rank
        self.committed = True
        return acknowledgement(self.armed)

    def sent_ack(self):
        if not self.committed or self.ack_complete:
            raise ValueError('ACK before commit or duplicate completion')
        self.ack_complete = True
        self._join()

    def _join(self):
        if self.command_arrived and self.ack_complete:
            self.unblocks += 1
            self.armed = None

    def preprocessing_input(self):
        if self.mask != 15 or self.armed is not None or any(v is None for v in self.words):
            raise ValueError('Incomplete Q/K operands')
        return self.words.copy()
