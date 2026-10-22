import unittest
from types import SimpleNamespace as NS
from flashnext_gguf.optimizations.q6_prefill_guard import schedule_guard

def req(prefill=False,ready=0):
 return NS(is_prefill_chunk=prefill,next_decode_eligible_step=ready,spec_token_ids=[-1,-1])
def sched(running,waiting=()):
 return NS(running=running,waiting=waiting,skipped_waiting=[],current_step=5,
   scheduler_config=NS(async_scheduling=True),num_spec_tokens=2,prefill_capacity_bound=True,max_num_running_reqs=2)

class GuardTests(unittest.TestCase):
 def step(self,s):
  def original(s,throttle_prefills):
   s.current_step+=1
   rows=[r for r in s.running if r.next_decode_eligible_step<=s.current_step and not(throttle_prefills and r.is_prefill_chunk)]
   return rows,throttle_prefills,s.max_num_running_reqs
  return schedule_guard(original,s)
 def test_pure_dual_decode_unchanged(self):
  s=sched([req(),req()]);rows,throttle,limit=self.step(s)
  self.assertEqual(len(rows),2);self.assertFalse(throttle);self.assertEqual(limit,2)
 def test_alternating_prefill_decode(self):
  a,b=req(True),req();s=sched([a,b])
  self.assertEqual(self.step(s)[0],[a]);self.assertEqual(b.next_decode_eligible_step,0)
  self.assertEqual(self.step(s)[0],[b]);self.assertTrue(s.prefill_capacity_bound)
  self.assertEqual(self.step(s)[0],[a]);self.assertEqual(b.spec_token_ids,[-1,-1])
 def test_new_prefill_while_decoding(self):
  a=req();s=sched([a],[req(True)])
  rows,throttle,limit=self.step(s);self.assertEqual(rows,[]);self.assertFalse(throttle);self.assertEqual(limit,2)
  rows,throttle,limit=self.step(s);self.assertEqual(rows,[a]);self.assertTrue(throttle);self.assertEqual(limit,1)
  self.assertEqual(s.max_num_running_reqs,2)
 def test_inflight_decoder_keeps_original_eligibility(self):
  a=req(ready=9);s=sched([req(True),a]);s._q6_next_phase='decode'
  self.assertEqual(self.step(s)[0],[s.running[0]]);self.assertEqual(a.next_decode_eligible_step,9)
 def test_error_restores_eligibility(self):
  a=req();s=sched([a,req(True)])
  def fail(*a,**k):raise RuntimeError('test')
  with self.assertRaises(RuntimeError):schedule_guard(fail,s)
  self.assertEqual(a.next_decode_eligible_step,0)
 def test_error_restores_capacity(self):
  s=sched([req(),req(True)]);s._q6_next_phase='decode'
  def fail(*a,**k):raise RuntimeError('test')
  with self.assertRaises(RuntimeError):schedule_guard(fail,s)
  self.assertTrue(s.prefill_capacity_bound);self.assertEqual(s.max_num_running_reqs,2)

if __name__=='__main__':unittest.main()
