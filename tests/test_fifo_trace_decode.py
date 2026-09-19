"""Host-only replay and malformed-record checks for the published finite trace."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
CAPTURE=ROOT/'examples/layer3_fifo_trace/capture'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

old=sys.modules.get('observer_decode')
sys.modules['observer_decode']=load('observer_decode',CAPTURE/'observer_decode.py')
try:
    decoder=load('published_fifo_decoder',CAPTURE/'fifo_observer_decode.py')
finally:
    if old is None:sys.modules.pop('observer_decode',None)
    else:sys.modules['observer_decode']=old


def tokens(source,late=False):
    events=[]
    if source['kind']=='root':
        events.append((2,0,0,4))
        for p in range(1,source['packets']+1):
            n=(31,31,26)[(p-1)%3]
            events.extend((e,p,n,5) for e in (3,4,5))
            if p%3==0:events.append((6,p,p//3,4))
        index=len(events) if late else 0
    else:
        events.append((7,0,0,4))
        for p,n in enumerate([31,31,26]*8+[8,8],1):
            events.extend((e,p,n,2) for e in (8,9,10))
            events.append((11,p,0,4))
        index=len(events) if late else 1
    events.insert(index,(1,events[index-1][1] if index else 0,0,4))
    return [(e<<28)|(i<<20)|(p<<14)|(n<<8)|f for i,(e,p,n,f) in enumerate(events,1)]


def record(source,values):
    raw=[0]*160;raw[:8]=[2*len(values),0x46494631,1,source['identity'],source['x'],source['y'],source['maximum'],len(values)]
    raw[31]=2*len(values);raw[32:32+len(values)]=values
    return raw


class FIFOTraceDecodeTests(unittest.TestCase):
    def test_all_finite_prefixes_and_compute_interleaving(self):
        count=0
        for source in decoder.SOURCES:
            for late in (False,True):
                values=tokens(source,late)
                self.assertEqual(len(values),source['maximum'])
                for n in range(len(values)+1):
                    result=decoder.decode_trace(record(source,values[:n]),source)
                    self.assertTrue(result['valid'],result)
                    self.assertFalse(result['source_counters_observed'])
                    self.assertFalse(result['later_bound_violation_excluded'])
                    count+=1
        self.assertEqual(count,820)

    def test_corruption_is_retained_and_rejected(self):
        for source in decoder.SOURCES:
            for index,mask in [(0,1),(1,1),(3,1),(8,1),(31,1),(32,1<<20),(33,8),(34,1<<14),(159,1)]:
                raw=record(source,tokens(source)[:3]);raw[index]^=mask
                result=decoder.decode_trace(raw,source)
                self.assertFalse(result['valid'],(source,index))
                self.assertEqual(result['raw_u32'],raw)

    def test_saved_physical_words_and_padding(self):
        words=json.loads((ROOT/'evidence/layer3-fifo-idle-words.json').read_text())['raw_u32']
        self.assertEqual(len(words),8000)
        rows=[words[i:i+160] for i in range(0,8000,160)]
        result=decoder.decode_observation(rows)
        self.assertTrue(result['all_rows_coherent'])
        self.assertTrue(result['all_trace_records_valid'])
        self.assertTrue(result['all_padding_valid'])
        self.assertEqual(result['global_observer']['ready_counts'],[1,20,0,0,0])
        self.assertEqual(sum(h['received_packet_count'] for h in result['heads']),549)
        self.assertEqual([t['count'] for t in result['traces']],[62,12,12,98,98,98])
        self.assertFalse(result['full_math_acceptance'])
        self.assertFalse(result['full_layer_complete'])
        bad=copy.deepcopy(rows);bad[0][32]=1
        self.assertEqual(decoder.decode_observation(bad)['original_padding_bad_rows'],[0])
        bad=copy.deepcopy(rows);bad[1][0]=1
        self.assertEqual(decoder.decode_observation(bad)['inactive_trace_bad_rows'],[0])

    def test_invalid_rectangle_extent(self):
        with self.assertRaises(ValueError):decoder.decode_observation([[0]*160 for _ in range(49)])
        with self.assertRaises(ValueError):decoder.decode_observation([[True]*160 for _ in range(50)])

if __name__=='__main__':unittest.main()
