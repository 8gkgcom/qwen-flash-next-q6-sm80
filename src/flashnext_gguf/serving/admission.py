# SPDX-License-Identifier: Apache-2.0
"""One FIFO admission queue; a lease lasts through the response stream."""
import asyncio
from collections import deque


class QueueFull(Exception):pass


class Admission:
    def __init__(self,max_waiting=4):
        self.max_waiting=max_waiting
        self.busy=[False,False]
        self.waiters=deque()
        self.draining=set()

    def snapshot(self):
        return {'running':sum(self.busy),'waiting':len(self.waiters),'max_running':2,
                'max_waiting':self.max_waiting,'slots':list(self.busy),
                'draining':sorted(self.draining),'policy':'fifo'}

    def set_draining(self,slots):
        self.draining=set(slots)
        self._dispatch()

    def _dispatch(self):
        # Dispatch the oldest eligible waiter to an available execution slot.
        for future,allowed in list(self.waiters):
            if future.cancelled():
                self.waiters.remove((future,allowed));continue
            slot=next((i for i in allowed if not self.busy[i] and i not in self.draining),None)
            if slot is not None:
                self.waiters.remove((future,allowed))
                self.busy[slot]=True
                future.set_result(slot)

    async def acquire(self,allowed=(0,1)):
        self._dispatch()
        slot=next((i for i in allowed if not self.busy[i] and i not in self.draining),None)
        if slot is not None:
            self.busy[slot]=True
            return slot
        if len(self.waiters)>=self.max_waiting:raise QueueFull()
        future=asyncio.get_running_loop().create_future()
        item=(future,allowed);self.waiters.append(item)
        try:return await asyncio.shield(future)
        except BaseException:
            if item in self.waiters:self.waiters.remove(item)
            elif future.done() and not future.cancelled():self.release(future.result())
            future.cancel()
            raise

    def release(self,slot):
        self.busy[slot]=False
        self._dispatch()
