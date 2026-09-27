"""State contract for cross-operator queue/route reuse, mirrored in CSL.

This is a protocol reference and an adversarial test oracle, not a fabric timing
simulator. Row delivery is supplied only after both actual contraction roots have
delivered; the backend must implement that distributed proof, not infer it from
one local send callback.
"""


class RegionEpoch:
    def __init__(self):
        self.epoch=-1
        self.phase='idle'
        self.native_idle=False
        self.row_delivered=False
        self.first_color=None
        self.outstanding=0
        self.consumer_done=False

    def begin(self, epoch):
        if self.phase!='idle' or epoch!=self.epoch+1 or self.outstanding or not 0<=epoch<65535:
            raise ValueError('Unreleased or non-consecutive epoch')
        self.epoch=epoch;self.phase='native';self.native_idle=False
        self.row_delivered=False;self.first_color=None;self.consumer_done=False

    def sent_native(self):
        if self.phase!='native' or self.native_idle:
            raise ValueError('Unexpected native callback')
        self.native_idle=True

    def delivered_row(self):
        if self.phase!='native' or self.row_delivered:
            raise ValueError('Unexpected row fence')
        self.row_delivered=True

    def can_install(self):
        return self.phase=='native' and self.native_idle and self.row_delivered

    def routes_installed(self):
        if not self.can_install():
            raise ValueError('Old native routes are still live')
        self.phase='routes'

    def bind_first(self, color):
        if self.phase!='routes' or color!=4:
            raise ValueError('First gather color must be rebound before readiness')
        self.first_color=color;self.phase='bound'

    def acknowledge(self):
        if self.phase!='bound' or self.first_color!=4:
            raise ValueError('Readiness without queue initialization')
        self.phase='down'

    def send_chunk(self, epoch):
        if self.phase!='down' or epoch!=self.epoch or self.outstanding or self.consumer_done:
            raise ValueError('Wrong epoch, unreturned credit or released buffer')
        self.outstanding=1

    def sent_chunk(self):
        if self.phase!='down' or self.outstanding!=1:
            raise ValueError('Unexpected chunk callback')
        self.outstanding=0

    def consumed(self):
        if self.phase!='down' or self.consumer_done:
            raise ValueError('Unexpected successor completion')
        self.consumer_done=True

    def release(self):
        if self.phase!='down' or not self.consumer_done or self.outstanding:
            raise ValueError('Consumer or send still owns buffers')
        self.phase='idle'


class GroupReadiness:
    """A quantized down packet cannot outrun any of its64 row transitions."""
    def __init__(self):
        self.values=set();self.rows=set()

    def arrive(self, row, *, value=False, ready=False):
        if not 0<=row<64 or not (value or ready):
            raise ValueError('Real group128 row')
        for enabled,seen in [(value,self.values),(ready,self.rows)]:
            if enabled:
                if row in seen:raise ValueError('Duplicate arrival')
                seen.add(row)

    def may_quantize(self):
        return len(self.values)==len(self.rows)==64
