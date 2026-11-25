"""Launch model experiments with a process-local macOS OpenMP path when available."""
import os
import sys
import subprocess
from pathlib import Path
import importlib.util

if __name__ == '__main__':
    environment = os.environ.copy()
    if sys.platform == 'darwin':
        spec = importlib.util.find_spec('sklearn')
        if spec and spec.origin:
            libraries = Path(spec.origin).parent / '.dylibs'
            if (libraries / 'libomp.dylib').is_file():
                previous = environment.get('DYLD_LIBRARY_PATH', '')
                environment['DYLD_LIBRARY_PATH'] = str(libraries) + (os.pathsep + previous if previous else '')
    command = [sys.executable, str(Path(__file__).with_name('phase6_models.py')), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, env=environment))
