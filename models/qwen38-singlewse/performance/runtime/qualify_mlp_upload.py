"""Real protobuf/gRPC loopback framing of the actual artifact; zero cluster jobs.

Runs under4GiB address space,1GiB sampled RSS and45s timeout via the launcher.
The local server has one worker and drops bytes after checking their digest.
"""
import argparse,hashlib,json,logging,resource,sys,time,types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
resource.setrlimit(resource.RLIMIT_FSIZE,(128<<20,128<<20))
import numpy as np
import grpc
from bounded_client import runtime_class
from cerebras.sdk.client import sdk_appliance_client as sdk


def status():
    keep={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.split(':')[0] in ['VmPeak','VmSize','VmHWM','VmRSS','Threads']:keep[line.split(':')[0]]=line.split(':',1)[1].strip()
    return keep


def main():
    p=argparse.ArgumentParser();p.add_argument('--map-banks',action='store_true');a=p.parse_args()
    root=Path('/srv/qwen38-singlewse-hardware/layer-mlp-hw-002')
    artifact=Path(json.loads((root/'artifact.json').read_text())['artifact'])
    meta=json.loads((root/'fixture.json').read_text());data=dict(np.load(root/'fixture.npz',allow_pickle=False))
    banks=np.load(root/'banks.npy',mmap_mode='r',allow_pickle=False) if a.map_banks else None
    workers=json.loads((root/'workers.json').read_text());profiles=json.loads((root/'profiles.json').read_text())
    logging.basicConfig(level=logging.INFO,force=True);logging.disable(logging.NOTSET)
    for name in ['grpc._common','grpc._channel']:logging.getLogger(name).disabled=False
    checkpoints=[dict(phase='before_channel',**status())];records=[]
    def receive(messages,context):
        for m in messages:
            raw=m.data_chunk;assert len(raw)==m.num_bytes==m.total_bytes==artifact.stat().st_size
            records.append(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        return b'OK'
    options=[('grpc.max_send_message_length',70<<20),('grpc.max_receive_message_length',70<<20)]
    pool=ThreadPoolExecutor(max_workers=1);server=grpc.server(pool,options=options)
    handler=grpc.stream_unary_rpc_method_handler(receive,request_deserializer=sdk.sdk_appliance_pb2.SdkArtifactsArgs.FromString,response_serializer=lambda x:x)
    server.add_generic_rpc_handlers([grpc.method_handlers_generic_handler('qualification',{'upload':handler})])
    port=server.add_insecure_port('127.0.0.1:0');server.start()
    channel=grpc.insecure_channel('127.0.0.1:'+str(port),options=options)
    def serialize(message):
        checkpoints.append(dict(phase='before_serialize',**status()))
        try:result=message.SerializeToString()
        except BaseException as e:
            print(json.dumps(dict(phase='serialize_failure',type=type(e).__name__,message=str(e),memory=status())),flush=True);raise
        checkpoints.append(dict(phase='after_serialize',**status()));return result
    call=channel.stream_unary('/qualification/upload',request_serializer=serialize,response_deserializer=lambda x:x)
    def upload(messages):
        assert call(messages,timeout=25)==b'OK';return types.SimpleNamespace(code=0,message='Loopback only')
    Runtime=runtime_class(single_message_limit=64<<20);instance=object.__new__(Runtime)
    instance.stub=lambda:types.SimpleNamespace(sdk_upload_files=upload)
    try:
        instance._upload_files(str(root),'no-cluster-job',[str(artifact)],None)
        assert records==[dict(bytes=artifact.stat().st_size,sha256=json.loads((root/'artifact.json').read_text())['sha256'])]
        result=dict(passed=True,physical_jobs=0,cluster_contact=False,loopback_only=True,bank_mapped=a.map_banks,
            artifact_bytes=artifact.stat().st_size,records=records,memory=checkpoints,after=status())
        print(json.dumps(result),flush=True)
    finally:
        channel.close();server.stop(0).wait(timeout=5);pool.shutdown(wait=True)


if __name__=='__main__':main()
