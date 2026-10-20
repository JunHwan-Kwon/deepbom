import os,sys,json,tempfile
from pathlib import Path
from deepbom import analyzers
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);d=p/'qualified_fixture-1.0.0.dist-info';d.mkdir()
 (d/'METADATA').write_text('Name: qualified-fixture\nVersion: 1.0.0\n')
 (d/'entry_points.txt').write_text('[deepbom.analyzers]\nprobe = qualified_fixture:analyze\n')
 (p/'qualified_fixture.py').write_text('def analyze(context):\n import time\n time.sleep(context.get("sleep",0))\n return context["request"]\n')
 row=dict(name='probe',version='1.0.0',operation='protocol',result_schema='deepbom.evaluation_protocol.v1',required_inputs=['request'],executes_model=False,cost='bounded metadata',dependencies=[],execution=dict(python='>=3.9',platforms=['linux'],network='inherit',gpu='not_requested',limits=dict(wall_seconds=2,cpu_seconds=2,address_space_bytes=512*1024*1024,file_bytes=1024*1024)))
 (d/'deepbom-analyzers.json').write_text(json.dumps({'schema':analyzers.CONTRACT_V2,'analyzers':[row]}))
 sys.path.insert(0,tmp);os.environ['PYTHONPATH']=tmp+os.pathsep+os.environ.get('PYTHONPATH','')
 descriptor=analyzers.discover()[0];assert 'qualified_fixture' not in sys.modules
 request={'operation':'protocol','value':'fixture payload'}
 result,receipt=analyzers._run_process_plugin('probe',descriptor,{'request':request},{})
 assert result==request and receipt['process_stopped'];assert 'qualified_fixture' not in sys.modules
 try:analyzers._run_process_plugin('probe',descriptor,{'request':request},{'network':'none'})
 except ValueError:pass
 else:raise AssertionError('unenforced network limit accepted')
 try:analyzers._run_process_plugin('probe',descriptor,{'request':request,'sleep':30},{})
 except TimeoutError:pass
 else:raise AssertionError('hang survived')
 print(json.dumps({'passed':['import-free discovery','selected child execution','output returned','mandatory unsupported restriction denied','hang killed'],'receipt':receipt}))
