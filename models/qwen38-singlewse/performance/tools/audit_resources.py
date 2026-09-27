"""Read-only accounting snapshot; does not cancel jobs or reserve hardware."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
script='''import datetime,getpass,json,subprocess

def query(*args):
 return json.loads(subprocess.run(['csctl','get',*args,'-o','json'],capture_output=True,text=True,check=True,timeout=25).stdout)
jobs=query('jobs','--my-jobs','--all-states')['items'];systems=query('systems')['items']
owned={j['meta']['name'].split('/')[-1]:j for j in jobs if j['spec']['user']['username']==getpass.getuser()}
terminal={'SUCCEEDED','FAILED','CANCELLED','CANCELED','DELETED'}
active=[{'id':n,'phase':j['status']['phase']} for n,j in owned.items() if j['status']['phase'] not in terminal]
assigned=[];other=0
for s in systems:
 names={str(v).split('/')[-1] for v in [s.get('jobId',''),*s.get('jobIds',[])]}-{''}
 for n in names & owned.keys():assigned.append({'system':s['meta']['name'],'id':n})
 other+=len(names-owned.keys())
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),own_jobs=len(owned),own_active=active,own_assignments=assigned,other_assignment_count=other,
 no_owned_hardware_allocation=not active and not assigned,provider_billed_node_hours=None,
 note='Job/system accounting snapshot; final provider billing is not available from these fields. SSH disconnect is not resource release.')))
'''
r=subprocess.run(['sh','/path/to/alcf-session.sh','host','python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=60)
result=json.loads(r.stdout)
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
