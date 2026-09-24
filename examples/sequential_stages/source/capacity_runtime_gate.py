"""Bind the five-position run to the actual compiled context8 observer contract."""
from pathlib import Path
import json
from observer_capacity import contract,PER_SERIAL,FIRST_STICKY_ERROR,publications,wire_limit
from runtime_pins import document
from stage_contract import require

COMPILE_MANIFEST_SHA='ae042bd9963b2191b026b193548b20711a920bdab63eb7b9431ffdf48569c0d6'
DEVICE={
 'attention/observer.csl':dict(bytes=2639,sha256='c4bb45860727d144480133d6e9191ea731c853fdfe0cae3944937af5c8f510c0'),
 'linear/observer.csl':dict(bytes=2712,sha256='753ea50f1aa939ad00425e7d81005fa392b4b215a02bd308db8ed79724b022eb'),
 'linear/layer0_pe.csl':dict(bytes=34051,sha256='6ede9981aab608bb4b665f6e12ccb5652f1628105b746df3b42419f2b4e303c3')}


def verify(root,evidence,inputs,context):
 require(context.ordinal==0 and context.stage.stage_id==0 and context.stage.layers==tuple(range(20)) and
         context.positions==tuple(range(5)) and context.restore_next_position is None and not context.reset_before_compute,
         'Exactly five continuous original positions; no counter reset or scope expansion')
 require(inputs['compile_source_manifest_pin']['sha256']==COMPILE_MANIFEST_SHA,'New actual Stage0 artifact must belong to the capacity repair')
 compiled=document(Path(evidence)/'manifest.json',inputs['compile_source_manifest_pin'],128<<10)
 require(all(compiled['files'][n]==p for n,p in DEVICE.items()),'Both actual device observers and first sticky error source match')
 expected=contract()
 actual=document(Path(inputs['compile_root'])/'OBSERVER-CAPACITY.json',compiled['files']['OBSERVER-CAPACITY.json'],128<<10)
 require(actual==expected==json.loads((Path(root)/'OBSERVER-CAPACITY.json').read_bytes()),'Device and host use one unchanged reviewed capacity contract')
 from runtime_families import load
 families=load(context,root)
 for family in ('attention','linear'):
  needed=len(context.positions)*PER_SERIAL[family]+FIRST_STICKY_ERROR
  require(needed=={'attention':5286,'linear':18656}[family] and needed<=publications(family) and
          families[family]['module'].ORIGIN_WIRE_LIMIT==wire_limit(family),
          'Fifth position plus first error fits continuous device/host observer bounds')
 return dict(positions=list(context.positions),normal_publications={f:5*PER_SERIAL[f] for f in PER_SERIAL},
             maximum_including_first_error={f:5*PER_SERIAL[f]+1 for f in PER_SERIAL},
             device_publications=expected['publications'],host_wire_limits=expected['wire_limits'],counters_reset=False)
