"""Kompilasi prototype dan self-test offline; tidak menjalankan Face."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess
base=Path(__file__).resolve().parent
output=base/'poc11c_silent_sink_v1'
if output.exists():raise SystemExit('Menolak menimpa binary yang sudah ada')
env=dict(os.environ,PKG_CONFIG_PATH='/usr/local/lib64/pkgconfig')
flags=subprocess.check_output(['pkg-config','--cflags','--libs','libndn-cxx'],env=env,text=True)
args=['g++','-std=c++17','-O1',str(base/'poc11c_silent_sink_v1.cpp'),'-o',str(output),*shlex.split(flags),'-lboost_json']
compile_result=subprocess.run(args,capture_output=True,text=True)
if compile_result.returncode:raise SystemExit(compile_result.stderr)
test=subprocess.run([str(output),'--self-test'],capture_output=True,text=True,check=True)
print(json.dumps({'compiler_command':args,'self_test_stdout':test.stdout,'network_face_constructed':False,'binary_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
