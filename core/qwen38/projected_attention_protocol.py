"""WP15 serialized 10PE original projection -> QK -> persistent attention wire."""
from dataclasses import dataclass

PE_COUNT=10
QK_PE=8
ATTENTION_PE=9
DATA_MAGIC=0x50414441
ACK_MAGIC=0x5041414b
VERSION=1
LEFT_GUARD=0xa55aa55a
RIGHT_GUARD=0x5aa55aa5
DESCRIPTOR_WORDS=10
DATA_WORDS=140
ACK_WORDS=13
QK_MASK=0x33
ATTENTION_MASK=0xfcc
ROLES=('raw_Q','raw_gate','raw_K','raw_V','rotated_Q','rotated_K')


@dataclass(frozen=True)
class Fragment:
    identity:int
    source:int
    destination:int
    role:int
    row_offset:int
    destination_offset:int

    @property
    def link(self):return self.source

    @property
    def receiver_queue(self):
        if self.source==8:return 6
        sources=(0,1,4,5) if self.destination==8 else (2,3,6,7)
        return 2+sources.index(self.source)

    def descriptor(self,contraction,generation,position):
        if not 1<=contraction<2**32 or not 1<=generation<2**32 or not 0<=position<8:
            raise ValueError('Invalid contraction/request/cache identity')
        return [contraction,generation,position,self.source,self.destination,self.identity,
                self.role,self.row_offset,128,self.destination_offset]


FRAGMENTS=tuple(
    Fragment(i,i,8 if i in (0,1,4,5) else 9,i//2,(i%2)*128,
             (i//4)*256+(i%2)*128 if i in (0,1,4,5) else
             (768 if i<4 else 512)+(i%2)*128)
    for i in range(8)) + tuple(Fragment(8+i,8,9,4+i//2,(i%2)*128,i*128) for i in range(4))


def frame(fragment,identity,payload):
    if len(payload)!=128 or any(type(x) is not int or not 0<=x<65536 for x in payload):
        raise ValueError('Exactly128 canonical native BF16 words required')
    return [LEFT_GUARD,DATA_MAGIC,VERSION,*fragment.descriptor(*identity),*payload,RIGHT_GUARD]


def parse_frame(words,fragment,identity):
    expected=[DATA_MAGIC,VERSION,*fragment.descriptor(*identity)]
    if len(words)!=DATA_WORDS+2 or words[0]!=LEFT_GUARD or words[-1]!=RIGHT_GUARD or words[1:13]!=expected:
        raise ValueError('Guard/header/request/token/fragment mismatch')
    if any(type(v) is not int or not 0<=v<65536 for v in words[13:-1]):
        raise ValueError('Noncanonical BF16 payload')
    return words[13:-1]


def ack(fragment,identity,status=0):
    return [LEFT_GUARD,ACK_MAGIC,VERSION,*fragment.descriptor(*identity),status,RIGHT_GUARD]


def verify_ack(words,fragment,identity):
    if words!=ack(fragment,identity):raise ValueError('ACK identity/status mismatch')


def routes():
    return [dict(link=i,source=i,destination=8 if i in (0,1,4,5) else 9,
                 data_color=2+i,ack_color=11+i,
                 source_queue=6 if i==8 else 2,
                 receiver_queue=next(f.receiver_queue for f in FRAGMENTS if f.source==i),
                 hops=(8 if i in (0,1,4,5) else 9)-i)
            for i in range(9)]


def fabric_budget():
    per_fragment=(DATA_WORDS+ACK_WORDS)*4
    return dict(tokens=2,data_frames=24,ack_frames=24,data_words=DATA_WORDS,ack_words=ACK_WORDS,
                injected_bytes=24*per_fragment,route_byte_hops=2*sum(f.destination-f.source for f in FRAGMENTS)*per_fragment,
                note='Logical injected payload and line route byte-hops; not measured hardware utilization')


class CommitModel:
    """Host protocol model for stale/duplicate/partial-commit adversarial checks."""
    def __init__(self,pe):
        if pe not in (8,9):raise ValueError('Consumer PE required')
        self.pe=pe;self.identity=None;self.mask=0
        self.payload=[0xffff]*(512 if pe==8 else 1024)
        self.released=True;self.generation=0;self.next_position=0;self.last_contraction=0

    def begin(self,contraction,generation,position):
        if not self.released or contraction!=self.last_contraction+1:
            raise ValueError('Previous token retained or contraction not monotonic')
        if generation!=self.generation:
            if generation<=self.generation or position!=0:raise ValueError('New request identity')
        elif position!=self.next_position:raise ValueError('Same-request cache position')
        FRAGMENTS[0].descriptor(contraction,generation,position)
        self.identity=(contraction,generation,position);self.mask=0;self.released=False

    def commit(self,fragment,words):
        if self.released or fragment.destination!=self.pe or self.mask&(1<<fragment.identity):
            raise ValueError('Wrong destination or duplicate fragment')
        payload=parse_frame(words,fragment,self.identity)
        # All validation precedes every destination write.
        self.payload[fragment.destination_offset:fragment.destination_offset+128]=payload
        self.mask|=1<<fragment.identity

    def release(self):
        if self.released or self.mask!=(QK_MASK if self.pe==8 else ATTENTION_MASK):
            raise ValueError('Incomplete token operands')
        self.last_contraction,self.generation,position=self.identity
        self.next_position=position+1;self.released=True
