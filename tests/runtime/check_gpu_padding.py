import pathlib
script_dir=pathlib.Path(__file__).resolve().parent
source=(script_dir/'test_gpu_ops.py').read_text()
# Reuse the already-tested initialization and real-weight fixtures only.
exec(compile(source[:source.index('results=[]')],str(script_dir/'test_gpu_ops.py'),'exec'))
a=source.index('e1=next(');b=source.index('for m in [1,2,3,4,64,128]:',a)
exec(compile(source[a:b],str(script_dir/'test_gpu_ops.py'),'exec'))
results=[]
for m in [1,3,64,128]:
    x=torch.randn(m,2560,device='cuda',dtype=torch.bfloat16)
    ids=torch.randint(0,16,(m,10),device='cuda',dtype=torch.int32)
    ids[:,5:]=-1;ids[0,:]=-1
    weights=torch.softmax(torch.randn(m,10,device='cuda'),-1)
    y=_fused_moe_gguf(x,w1,w2,weights,ids,e1['qtype'],e2['qtype'],'silu')
    torch.cuda.synchronize()
    ref=torch.zeros_like(y,dtype=torch.float32)
    for k in range(10):
        ii=ids[:,k].clamp_min(0).long()
        g=torch.bmm(dg[ii],x.float().unsqueeze(2)).squeeze(2)
        u=torch.bmm(du[ii],x.float().unsqueeze(2)).squeeze(2)
        v=torch.bmm(dd[ii],(torch.nn.functional.silu(g)*u).unsqueeze(2)).squeeze(2)
        ref+=v.masked_fill((ids[:,k]<0)[:,None],0)*weights[:,k,None]
    err=float((y.float()-ref).norm()/ref.norm().clamp_min(1e-20))
    assert err<.08,err
    assert bool((y[0]==0).all())
    row={'m':m,'relative_l2':err,'padded_row_exact_zero':True}
    if m<=3:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _fused_moe_gguf(x,w1,w2,weights,ids,e1['qtype'],e2['qtype'],'silu')
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_y=_fused_moe_gguf(x,w1,w2,weights,ids,e1['qtype'],e2['qtype'],'silu')
        graph.replay();torch.cuda.synchronize()
        assert torch.equal(y,graph_y)
        ids.copy_(torch.randint(0,16,(m,10),device='cuda',dtype=torch.int32))
        ids[:,-1]=-1
        expected=_fused_moe_gguf(x,w1,w2,weights,ids,e1['qtype'],e2['qtype'],'silu')
        graph.replay();torch.cuda.synchronize()
        assert torch.equal(expected,graph_y)
        row['cuda_graph_dynamic_ids_exact']=True
    results.append(row);print(row,flush=True)
(output/'gpu-padding-tests.json').write_text(json.dumps(results,indent=2))
