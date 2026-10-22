# SPDX-License-Identifier: Apache-2.0
"""Bounded FIFO requests for Q6; leases cover the complete response stream."""
import asyncio
import hmac
import json
from .admission import Admission, QueueFull

PATHS={'/v1/chat/completions','/v1/completions','/v1/responses','/v1/messages'}


class Q6Queue:
    def __init__(self,app,*,max_running=2,max_waiting=4,api_key=''):
        if max_running not in (1,2):raise ValueError('Q6 supports one or two admitted requests')
        self.app=app;self.max_running=max_running;self.api_key=api_key
        self.admission=Admission(max_waiting)

    def authenticated(self,scope):
        if not self.api_key:return True
        headers=dict(scope.get('headers',[]))
        auth=headers.get(b'authorization',b'').decode('utf-8',errors='replace')
        token=auth[7:] if auth.lower().startswith('bearer ') else ''
        if not token and scope.get('path','').startswith('/v1/messages'):
            token=headers.get(b'x-api-key',b'').decode('utf-8',errors='replace')
        return hmac.compare_digest(token.encode(),self.api_key.encode())

    def snapshot(self):
        return {**self.admission.snapshot(),'max_running':self.max_running,'policy':'q6_request_fifo'}

    @staticmethod
    async def response(send,status,data):
        body=json.dumps(data,ensure_ascii=False).encode()
        headers=[(b'content-type',b'application/json'),(b'content-length',str(len(body)).encode())]
        if status==429:headers.append((b'retry-after',b'1'))
        await send({'type':'http.response.start','status':status,'headers':headers})
        await send({'type':'http.response.body','body':body})

    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        path=scope.get('path','').rstrip('/')
        if path=='/queue/status' and scope.get('method')=='GET':
            if not self.authenticated(scope):
                return await self.response(send,401,{'error':{'message':'Unauthorized','type':'authentication_error'}})
            return await self.response(send,200,self.snapshot())
        if scope.get('method')!='POST' or path not in PATHS or not self.authenticated(scope):
            return await self.app(scope,receive,send)

        # Drain a bounded request body first, then listen for disconnect while queued.
        # This avoids retaining dead clients and does not read the generation stream.
        messages=[];size=0
        while True:
            item=await receive()
            if item['type']=='http.disconnect':return
            messages.append(item);size+=len(item.get('body',b''))
            if size>64*1024**2:
                return await self.response(send,413,{'error':{'message':'Request exceeds 64 MiB'}})
            if not item.get('more_body',False):break
        acquire=asyncio.create_task(self.admission.acquire(tuple(range(self.max_running))))
        disconnect=asyncio.create_task(receive())
        slot=None
        try:
            done,_=await asyncio.wait({acquire,disconnect},return_when=asyncio.FIRST_COMPLETED)
            if disconnect in done and disconnect.result()['type']=='http.disconnect':return
            try:slot=await acquire
            except QueueFull:
                return await self.response(send,429,{'error':{'message':f'Q6 queue full: {self.max_running} running + {self.admission.max_waiting} waiting','type':'rate_limit_error'}})
            disconnect.cancel();await asyncio.gather(disconnect,return_exceptions=True)
            index=0
            async def replay():
                nonlocal index
                if index<len(messages):
                    result=messages[index];index+=1;return result
                return await receive()
            await self.app(scope,replay,send)
        finally:
            disconnect.cancel();await asyncio.gather(disconnect,return_exceptions=True)
            if slot is not None:self.admission.release(slot)
            elif acquire.done() and not acquire.cancelled() and acquire.exception() is None:
                self.admission.release(acquire.result())
            else:
                acquire.cancel();await asyncio.gather(acquire,return_exceptions=True)
