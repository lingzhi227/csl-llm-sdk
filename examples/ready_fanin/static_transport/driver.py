"""One simulator-only entry point, with platform lifetime through normal stop."""
from pathlib import Path
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
from capture import capture
ROOT=Path(__file__).resolve().parent

def main():
    platform=None
    def create():
        nonlocal platform
        platform=get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3)
        return SdkRuntime(str(ROOT/'out'),platform)
    return capture(ROOT/'cases/lifecycle',create,MemcpyDataType,MemcpyOrder)

if __name__=='__main__':main()
