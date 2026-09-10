"""Small checked byte-submission schedule; all candidate accumulation stays CSL."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    generation:int
    index:int
    offset:int
    valid:int


class Schedule:
    def __init__(self,total,generation):
        if type(total) is not int or not 1<=total<=5120 or type(generation) is not int or generation<=0:
            raise ValueError('Invalid total extent or generation')
        self.total=total;self.generation=generation;self.consumed=0;self.completed=0;self.finalized=False

    def submit(self,tile):
        if self.finalized or self.consumed==self.total:raise ValueError('No unconsumed columns')
        if any(type(x) is not int for x in (tile.generation,tile.index,tile.offset,tile.valid)):
            raise ValueError('Tile fields must be integer')
        expected=Tile(self.generation,self.completed,self.consumed,min(112,self.total-self.consumed))
        if tile!=expected:raise ValueError('Missing, duplicate, out-of-order or wrong-extent tile')
        self.consumed+=tile.valid;self.completed+=1

    def finalize(self):
        if self.finalized or self.consumed!=self.total:raise ValueError('Incomplete or already finalized schedule')
        self.finalized=True
        return self.generation


def tiles(total,generation):
    if type(total) is not int or not 1<=total<=5120:raise ValueError('Invalid total')
    return [Tile(generation,i,offset,min(112,total-offset)) for i,offset in enumerate(range(0,total,112))]
