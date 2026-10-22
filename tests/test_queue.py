import asyncio,json,unittest
from flashnext_gguf.serving.q6_queue import Q6Queue

class Q6QueueTests(unittest.IsolatedAsyncioTestCase):
 async def asyncSetUp(self):
  self.open=asyncio.Event();self.entered=[]
  async def app(scope,receive,send):
   self.entered.append(scope.get('id'))
   await receive()
   if scope.get('error'):raise RuntimeError('upstream failed')
   await self.open.wait()
   await send({'type':'http.response.start','status':200})
   await send({'type':'http.response.body','body':b'ok'})
  self.queue=Q6Queue(app,api_key='test-key');self.tasks=[]
 async def asyncTearDown(self):
  for task in self.tasks:task.cancel()
  await asyncio.gather(*self.tasks,return_exceptions=True)
 def launch(self,i,**extra):
  incoming=asyncio.Queue();incoming.put_nowait({'type':'http.request','body':b'{}'})
  output=[]
  async def send(message):output.append(message)
  scope={'type':'http','method':'POST','path':'/v1/chat/completions','headers':[(b'authorization',b'Bearer test-key')],'id':i,**extra}
  task=asyncio.create_task(self.queue(scope,incoming.get,send));self.tasks.append(task)
  return task,incoming,output
 async def spin(self,predicate):
  for _ in range(100):
   if predicate():return
   await asyncio.sleep(.001)
  self.fail('Timed out waiting for queue condition')
 async def test_two_running_four_waiting_and_fifo(self):
  req=[self.launch(i) for i in range(7)]
  await self.spin(lambda:req[6][0].done())
  self.assertEqual(self.queue.snapshot()['running'],2);self.assertEqual(self.queue.snapshot()['waiting'],4)
  self.assertEqual(req[6][2][0]['status'],429)
  self.open.set();await asyncio.gather(*(r[0] for r in req))
  self.assertEqual(self.entered,list(range(6)))
  self.assertEqual(self.queue.snapshot()['running'],0);self.assertEqual(self.queue.snapshot()['waiting'],0)
 async def test_waiting_disconnect_releases_queue_position(self):
  req=[self.launch(i) for i in range(3)]
  await self.spin(lambda:self.queue.snapshot()['waiting']==1)
  req[2][1].put_nowait({'type':'http.disconnect'});await req[2][0]
  self.assertEqual(self.queue.snapshot()['waiting'],0)
  self.assertEqual(self.queue.snapshot()['running'],2)
 async def test_cancellation_and_error_release_slots(self):
  a=self.launch(1);await self.spin(lambda:len(self.entered)==1);a[0].cancel()
  await asyncio.gather(a[0],return_exceptions=True)
  self.assertEqual(self.queue.snapshot()['running'],0)
  b=self.launch(2,error=True)
  with self.assertRaises(RuntimeError):await b[0]
  self.assertEqual(self.queue.snapshot()['running'],0)
 async def test_auth_and_status_do_not_occupy_slots(self):
  for headers,status in [([],401),([(b'authorization',b'Bearer test-key')],200)]:
   output=[]
   async def send(m):output.append(m)
   await self.queue({'type':'http','method':'GET','path':'/queue/status','headers':headers},None,send)
   self.assertEqual(output[0]['status'],status)
  self.assertEqual(self.queue.snapshot()['running'],0)
 async def test_stream_holds_slot_until_final_body(self):
  finish=asyncio.Event();started=asyncio.Event()
  async def streaming(scope,receive,send):
   await receive();await send({'type':'http.response.start','status':200})
   await send({'type':'http.response.body','body':b'partial','more_body':True});started.set()
   await finish.wait();await send({'type':'http.response.body','body':b'final','more_body':False})
  self.queue=Q6Queue(streaming,max_running=1,api_key='test-key')
  a=self.launch(0);await started.wait();b=self.launch(1)
  await self.spin(lambda:self.queue.snapshot()['waiting']==1)
  self.assertEqual(self.queue.snapshot()['running'],1)
  finish.set();await asyncio.gather(a[0],b[0])
  self.assertEqual(self.queue.snapshot()['running'],0)
  self.assertEqual(self.queue.snapshot()['waiting'],0)

if __name__=='__main__':unittest.main()
