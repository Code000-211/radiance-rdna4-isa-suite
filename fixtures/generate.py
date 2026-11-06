"""Optional fixture regeneration. Requires llvmlite with AMDGPU and ld.lld.
Never launches a GPU kernel. Packaged fixtures avoid this dependency for tests.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import llvmlite.binding as llvm

root=Path(__file__).resolve().parent
llvm.initialize_all_targets();llvm.initialize_all_asmprinters()
target=llvm.Target.from_triple('amdgcn-amd-amdhsa').create_target_machine(cpu='gfx1201',opt=3)
mod=llvm.parse_assembly((root/'smoke.ll').read_text());mod.verify()
linker=shutil.which('ld.lld')
if not linker:raise SystemExit('ld.lld is required')
with tempfile.TemporaryDirectory() as d:
    obj=Path(d)/'smoke.o';out=Path(d)/'smoke.hsaco'
    obj.write_bytes(target.emit_object(mod))
    subprocess.run([linker,'-shared',str(obj),'-o',str(out)],check=True)
    # Deliberately do not overwrite the reviewed fixtures.
    (root/'smoke-regenerated.o').write_bytes(obj.read_bytes())
    (root/'smoke-regenerated.hsaco').write_bytes(out.read_bytes())
print('Generated new CPU-only fixtures; review hashes before replacing packaged files.')
