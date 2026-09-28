"""Adversarial protocol oracle, not a device execution/timing simulator."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Tag:
    request: int
    generation: int
    position: int


class StageMailbox:
    def __init__(self, requests, chunks=40, context=96):
        if requests<1 or chunks<1:raise ValueError('Positive capacities required')
        self.requests=requests;self.chunks=chunks;self.context=context;self.slots={};self.serial=0
        self.generation=[0]*requests;self.next_position=[0]*requests

    def grant(self, tag):
        if not 0<=tag.request<self.requests or tag.generation!=self.generation[tag.request] or tag.position!=self.next_position[tag.request] or not 0<=tag.position<self.context:
            raise ValueError('Stale generation or nonconsecutive request position')
        if len(self.slots)==2:return None  # Explicit backpressure, no overwrite.
        if any(s['tag'].request==tag.request for s in self.slots.values()):
            raise ValueError('Request already owns a live slot')
        self.serial+=1
        self.slots[self.serial]=dict(tag=tag,chunks=set(),committed=False,sent=False,credit=False)
        return self.serial

    def _slot(self,lease,tag):
        s=self.slots.get(lease)
        if s is None or s['tag']!=tag:raise ValueError('Stale/wrong request lease')
        return s

    def arrive(self,lease,tag,chunk):
        s=self._slot(lease,tag)
        if s['committed'] or not 0<=chunk<self.chunks or chunk in s['chunks']:
            raise ValueError('Duplicate/late/out-of-range chunk')
        s['chunks'].add(chunk)

    def commit_state(self,lease,tag):
        s=self._slot(lease,tag)
        if len(s['chunks'])!=self.chunks or s['committed']:raise ValueError('Missing operands or repeated state update')
        s['committed']=True;self.next_position[tag.request]+=1

    def complete(self,lease,tag,*,send=False,credit=False):
        s=self._slot(lease,tag)
        if not s['committed'] or not(send or credit):raise ValueError('Output not computed')
        for flag,enabled in [('sent',send),('credit',credit)]:
            if enabled:
                if s[flag]:raise ValueError('Duplicate completion')
                s[flag]=True
        if s['sent'] and s['credit']:del self.slots[lease]

    def reset(self,request,generation,*,state_cleared):
        if not 0<=request<self.requests or any(s['tag'].request==request for s in self.slots.values()) or not state_cleared:
            raise ValueError('Reset requires drained slots and actual state-clear acknowledgement')
        if generation!=self.generation[request]+1:raise ValueError('Nonconsecutive generation')
        self.generation[request]=generation;self.next_position[request]=0


class FeedbackAdmission:
    """Independent requests may overlap; one request cannot predict its next token."""
    def __init__(self,prompts,context=96):
        if not prompts or any(not p or len(p)>=context or any(not 0<=t<248320 for t in p) for p in prompts):
            raise ValueError('Nonempty fixed tokenized prompts fitting declared context required')
        requests=len(prompts);self.prompts=[tuple(p) for p in prompts]
        self.requests=requests;self.context=context;self.live={};self.positions=[0]*requests;self.generations=[0]*requests
        self.feedback=[None]*requests

    def admit(self,request,generation,token):
        if not 0<=request<self.requests or not 0<=token<248320:raise ValueError('Request/token range')
        if generation!=self.generations[request] or request in self.live or self.positions[request]>=self.context:
            raise ValueError('Stale, outstanding or over-context request')
        position=self.positions[request];prompt=self.prompts[request]
        expected=prompt[position] if position<len(prompt) else self.feedback[request]
        if token!=expected:raise ValueError('Input is neither the next prompt token nor actual autoregressive feedback')
        tag=Tag(request,generation,self.positions[request]);self.live[request]=tag;return tag

    def selected(self,tag,token,*,all_layers_and_head_completed):
        if self.live.get(tag.request)!=tag or not all_layers_and_head_completed or not 0<=token<248320:
            raise ValueError('No complete original-model feedback for this request')
        del self.live[tag.request];self.positions[tag.request]+=1;self.feedback[tag.request]=token
        return token

    def reset(self,request,generation,stage_clear_acks):
        if not 0<=request<self.requests or request in self.live or set(stage_clear_acks)!=set(range(66)):
            raise ValueError('All66 actual stage drain/state-clear acknowledgements required')
        if generation!=self.generations[request]+1:raise ValueError('Nonconsecutive generation')
        self.generations[request]=generation;self.positions[request]=0;self.feedback[request]=None
