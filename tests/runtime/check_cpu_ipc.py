"""Exercise delayed messages, dropped notifications, timeout and shutdown."""
import os,sys,time,threading,json,queue
from pathlib import Path
os.environ['Q6_CUDA_BLOCKING_WAIT']='0'
os.environ.pop('Q6_CPU_WAIT_CONTROL',None)
from flashnext_gguf.optimizations.q6_cpu_ipc import install
install()
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
writer=MessageQueue(1,1,max_chunk_bytes=4096,max_chunks=2);handle=writer.export_handle()
ready=threading.Event();done=threading.Event();errors=[];box={};result={}
def reader():
 try:
  mq=MessageQueue.create_from_handle(handle,0);box['reader']=mq;mq.wait_until_ready();ready.set()
  received=[]
  for i in range(200):
   row=mq.dequeue(timeout=2);assert row['index']==i;assert row['payload']=='x'*(8192 if i%7==0 else 8);received.append(row['index'])
  result['ordered_messages']=len(received)
  start=time.monotonic()
  try:mq.dequeue(timeout=.025)
  except TimeoutError:result['timeout_seconds']=time.monotonic()-start
  else:raise AssertionError('Missing timeout')
  done.set()
  try:mq.dequeue(timeout=5,indefinite=True)
  except RuntimeError as e:assert str(e)=='cancelled';result['shutdown_cancelled']=True
  else:raise AssertionError('Missing shutdown')
 except BaseException as e:errors.append(repr(e));ready.set();done.set()
t=threading.Thread(target=reader);t.start();writer.wait_until_ready();assert ready.wait(5)
# Suppress write notifications to verify reads rely on authoritative SHM flags.
writer._spin_condition.notify=lambda:None
for i in range(200):
 if i%11==0:time.sleep(.002)
 writer.enqueue({'index':i,'payload':'x'*(8192 if i%7==0 else 8)},timeout=2)
assert done.wait(5);box['reader'].shutdown();t.join(5);assert not t.is_alive();assert not errors,errors
assert result['ordered_messages']==200 and result['shutdown_cancelled']
box['reader'].buffer.shared_memory.close()
writer.buffer.shared_memory.close();writer.buffer.shared_memory.unlink();writer.buffer.is_creator=False
print(json.dumps(result))
