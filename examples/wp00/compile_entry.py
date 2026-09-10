import json, os, sys
from pathlib import Path
info = {'python': sys.executable, 'version': sys.version, 'cwd': os.getcwd(), 'tmpdir': os.environ.get('TMPDIR'), 'pid': os.getpid(), 'cgroup': Path('/proc/self/cgroup').read_text()}
Path('container-compile.json').write_text(json.dumps(info, indent=2)+'\n')
os.execvp('sdk_debug_shell', ['sdk_debug_shell', 'compile', *sys.argv[1:]])
