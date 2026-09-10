import json
from pathlib import Path
import struct
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from prepare_wp13_reference import hidden_bytes
from plan_wp13_device import symbolic_sequence


class ReproductionTests(unittest.TestCase):
    def test_inputs_match_published_experiment_words(self):
        published=json.loads((ROOT/'evidence/wp13-inputs.json').read_text())
        words=[x for row in published['u16_words'] for x in row]
        self.assertEqual(hidden_bytes(),struct.pack('<'+str(len(words))+'H',*words))

    def test_symbolic_order_matches_real_pilot_journal(self):
        actual=[];launches=0
        for line in (ROOT/'evidence/wp13-slab00-journal.jsonl').read_text().splitlines():
            event=json.loads(line)
            if event['kind']=='copy_enter':
                actual.append({k:event[k] for k in ('sequence','direction','name','pe','count','bits','host_bytes','native_bytes')}|{'kind':'copy'})
            elif event['kind']=='launch_enter':
                launches+=1;actual.append({'kind':'launch','ordinal':launches,'name':event['name']})
        self.assertEqual(symbolic_sequence(1,4),actual)


if __name__=='__main__':unittest.main()
