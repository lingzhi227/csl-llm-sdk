"""Count scheduled controller-network words; never convert them to time/TPS.

Filters select RAMP delivery, not the cardinal multicast branches. These counts
describe the generated fixed routes under successful completion. Native tree
reductions, root/fusion ingress and SDK traffic are deliberately outside scope.
"""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'performance'));sys.path.insert(0,str(Path(__file__).parent))
from stage_layer_mlp_compile import build
from spatial.layer_mlp_network import build_mlp_network
from spatial.layer_fused_routes import translate_routes


def audit(stage,network):
    schedule=network['grant_schedule'];phases={};link_words=Counter()
    delta={'WEST':(-1,0),'EAST':(1,0),'NORTH':(0,-1),'SOUTH':(0,1)}
    def edge(pe,tx):
        dx,dy=delta[tx];return (*pe,pe[0]+dx,pe[1]+dy)
    for kind,columns in [(0,32),(1,64)]:
        groups=sum(g['kind']==kind for g in schedule);copies=1 if network.get('shared_inputs') else 2
        frames=groups*(128//columns)*copies
        words=frames*(columns+5)
        phases['initial_input' if kind==0 else 'fused_activation']=dict(groups=groups,native_columns=columns,
            wire_frames=frames,wire_words=words,operand_halfwords=groups*128*copies,header_words=frames*5)
    input_words=sum(p['wire_words'] for p in phases.values())
    grant_words=3*(len(schedule)+len(network['prepare_schedule']))
    for route in network['routes']:
        if route['color'] not in (2,20):continue
        words=input_words if route['color']==2 else grant_words
        for tx in route['tx']:
            if tx!='RAMP':link_words[edge(route['pe'],tx)]+=words
    bus={tuple(r['pe']):r for r in network['routes'] if r['color']==18}
    sources={s['id']:tuple(s['pe']) for s in network['senders']};controller=tuple(network['controller'])
    for grant in schedule:
        pe=sources[grant['target']];seen=set()
        while pe!=controller:
            if pe in seen:raise ValueError('Return cycle')
            seen.add(pe);tx=bus[pe]['tx'];assert len(tx)==1 and tx[0]!='RAMP'
            link=edge(pe,tx[0]);link_words[link]+=grant['words'];pe=link[2:]
    native=sum(len(b['slices'])*(32 if b['tag']<1024 else 64) for b in network['input_bindings'])
    return dict(scope='Scheduled controller input/grant/return network only; excludes native reductions, projection-to-fusion routes and SDK traffic.',
        phases=phases,controller_native_broadcast_words=input_words,controller_grant_words=grant_words,
        controller_response_words=sum(g['words'] for g in schedule),native_operand_halfwords_delivered=native,
        directed_link_count=len(link_words),sum_directed_word_hops=sum(link_words.values()),
        busiest_directed_links=[dict(from_pe=list(e[:2]),to_pe=list(e[2:]),words=n) for e,n in link_words.most_common(8)],
        estimated_cycles=None,estimated_tps=None,physical_measurement=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--output',type=Path,required=True);p.add_argument('--shared-inputs',action='store_true');args=p.parse_args()
    attempt=ROOT/'performance/evidence'/args.attempt
    files,_,_,_=build('layer_00',shared_inputs=args.shared_inputs);manifest=json.loads((attempt/'source-manifest.json').read_text())['files']
    for name,raw in files.items():
        if hashlib.sha256(raw).hexdigest()!=manifest[name]:raise ValueError('Current lowering differs from frozen attempt: '+name)
    plan=json.loads((ROOT/'performance/evidence/layer-native-schedule-002/layer-schedule.json').read_text())
    stage=next(s for s in plan['stages'] if s['id']=='layer_00');gate=next(r for r in stage['regions'] if r['role']=='gate_up')
    template=json.loads((ROOT/'performance/evidence/layer-fused-routes-003/routes-78x67.json').read_text())
    original=next(r for s in plan['stages'] for r in s['regions'] if r['id']==template['region'])
    network=build_mlp_network(stage,translate_routes(template,original,gate),args.shared_inputs)
    binding=json.loads(files['network-binding.json']);actual=hashlib.sha256(json.dumps(network,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert actual==binding['network_sha256']
    result=dict(attempt=args.attempt,network_sha256=actual,source_manifest_sha256=hashlib.sha256((attempt/'source-manifest.json').read_bytes()).hexdigest(),**audit(stage,network))
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
