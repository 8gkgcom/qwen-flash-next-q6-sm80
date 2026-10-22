#!/usr/bin/env python3
"""Print or start an isolated Q6 container; never stop an existing workload."""
from pathlib import Path
import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate(c):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*',c['name']):
        raise ValueError('Invalid container name')
    if len(c['gpu_ids'])!=2 or len(set(c['gpu_ids']))!=2 or any(not re.fullmatch(r'(\d+|GPU-[a-fA-F0-9-]+)',g) for g in c['gpu_ids']):
        raise ValueError('Select two distinct GPU indices or UUIDs')
    limits = {'max_model_len':(4096,262144),'max_num_seqs':(1,2),
              'max_num_batched_tokens':(128,1024),'max_waiting':(0,4),'port':(1024,65535),'mtp_tokens':(1,2)}
    for key,(lo,hi) in limits.items():
        if type(c[key]) is not int or not lo<=c[key]<=hi:
            raise ValueError('Outside validated range: '+key)
    if c['pp_partition'] not in ([25,23],[24,24]):
        raise ValueError('Validated PP partitions are 25/23 and 24/24')
    if not 0.5<=c['gpu_memory_utilization']<=0.96:
        raise ValueError('Validated memory budget is at most 0.96')
    if c['draft_q8_marlin'] and c['pp_partition']!=[25,23]:
        raise ValueError('Q8 draft screening requires the validated 25/23 layout')
    if c['dynamic_mtp'] and (c['max_num_seqs'],c['mtp_tokens'])!=(2,2):
        raise ValueError('Dynamic MTP requires two slots and MTP2')
    if c['reasoning_effort'] not in ('low','medium','xhigh'):
        raise ValueError('Unsupported reasoning effort')
    if not 0<=c['temperature']<=2 or not 0<c['top_p']<=1 or type(c['top_k']) is not int or c['top_k']<1:
        raise ValueError('Invalid sampling settings')
    if not 0<=c['presence_penalty']<=2:
        raise ValueError('Invalid presence penalty')
    if not Path(c['data_dir']).is_absolute():
        raise ValueError('data_dir must be absolute')
    if c.get('model_dir') and not Path(c['model_dir']).is_absolute():
        raise ValueError('model_dir must be absolute when specified')
    for name in ['data_dir','model_dir','api_key_file']:
        if any(char in c.get(name,'') for char in (',','\r','\n')):
            raise ValueError('Docker mount path cannot contain comma or newline: '+name)
    return c


def engine_plan(c):
    validate(c)
    spec = dict(method='mtp',num_speculative_tokens=c['mtp_tokens'],quantization='gguf')
    if c['dynamic_mtp']:
        spec['num_speculative_tokens_per_batch_size']=[[1,1,2],[2,2,1]]
    args = [sys.executable,'-m','vllm.entrypoints.openai.api_server',
        '--model','/data/runtime/model-config','--served-model-name',c['served_model_name'],
        '--host','0.0.0.0','--port','8000','--load-format','flashnext_gguf','--quantization','gguf',
        '--dtype','bfloat16','--tensor-parallel-size','1','--pipeline-parallel-size','2',
        '--distributed-executor-backend','mp','--max-model-len',str(c['max_model_len']),
        '--max-num-seqs',str(c['max_num_seqs']),'--max-num-batched-tokens',str(c['max_num_batched_tokens']),
        '--gpu-memory-utilization',str(c['gpu_memory_utilization']),'--mamba-cache-mode','align',
        '--block-size','1632','--long-prefill-token-threshold',str(max(128,c['max_num_batched_tokens']-256) if c['max_num_seqs']==2 else c['max_num_batched_tokens']),
        '--generation-config','vllm','--reasoning-parser','qwen3','--enable-auto-tool-choice',
        '--tool-call-parser','qwen3_coder','--enable-prefix-caching','--enable-chunked-prefill',
        '--default-chat-template-kwargs',json.dumps(dict(enable_thinking=True,reasoning_effort=c['reasoning_effort'],preserve_thinking=False)),
        '--override-generation-config',json.dumps(dict(temperature=c['temperature'],top_p=c['top_p'],top_k=c['top_k'],min_p=0.,presence_penalty=c['presence_penalty'],frequency_penalty=0.,repetition_penalty=1.)),
        '--compilation-config',json.dumps(dict(mode=0,cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=list(range(1,(c['mtp_tokens']+1)*c['max_num_seqs']+1)))),
        '--speculative-config',json.dumps(spec),'--middleware','flashnext_gguf.serving.middleware.Q6Middleware']
    env = dict(CUDA_VISIBLE_DEVICES=','.join(c['gpu_ids']),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VLLM_PP_LAYER_PARTITION=','.join(map(str,c['pp_partition'])),
        VLLM_PLE_CPU_OFFLOAD='1',VLLM_PLUGINS='flashnext_gguf',VLLM_USE_FLASHINFER_SAMPLER='0',
        VLLM_USE_V2_MODEL_RUNNER='1',VLLM_USE_SPINLOOP_EXT='0',PYTHONSAFEPATH='1',
        FLASHNEXT_GGUF_ROOT='/data/runtime',FLASHNEXT_PLE_RESIDENT='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        VLLM_CACHE_ROOT='/data/cache/vllm',TRITON_CACHE_DIR='/data/cache/triton',
        TORCHINDUCTOR_CACHE_DIR='/data/cache/torchinductor',XDG_CACHE_HOME='/data/cache',
        VLLM_ALLOW_LONG_MAX_MODEL_LEN='0',PYTHONUNBUFFERED='1',NCCL_P2P_DISABLE='0',
        Q6_DYNAMIC_GRAPH_FIX=str(int(c['dynamic_mtp'])),Q6_MIXED_PREFILL_GUARD=str(int(c['max_num_seqs']==2)),
        Q6_DRAFT_Q8_MARLIN=str(int(c['draft_q8_marlin'])),Q6_SHORTCONV_ASYNC=str(int(c['shortconv_async'])),
        Q6_CPU_IPC=str(int(c['cpu_ipc'])),Q6_MAX_RUNNING=str(c['max_num_seqs']),Q6_MAX_WAITING=str(c['max_waiting']),
        Q38_RAM_COMPUTE_OPT='0',Q38_HYBRID_DRAFT='0',VLLM_PLE_GDS_RUNNER='0',FLASHNEXT_DEBUG_MOE='0')
    return args,env


def container_plan(c, config_path):
    validate(c)
    model_dir=Path(c.get('model_dir') or Path(c['data_dir'])/'models').resolve()
    args = ['docker','run','--name',c['name'],'--gpus','all','--shm-size','8g',
            '--ulimit','memlock=-1:-1','--cap-add','IPC_LOCK',
            '-p',f'127.0.0.1:{c["port"]}:8000',
            '--mount',f'type=bind,source={Path(c["data_dir"]).resolve()},target=/data',
            '--mount',f'type=bind,source={model_dir},target=/data/models,readonly',
            '--mount',f'type=bind,source={config_path.resolve()},target=/config/local.json,readonly']
    if c['api_key_file']:
        args += ['--mount',f'type=bind,source={Path(c["api_key_file"]).resolve()},target=/run/secrets/api_key,readonly']
    return args+[c['image']]


def check_idle_gpus(selected):
    inventory = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,compute_cap','--format=csv,noheader,nounits'],text=True)
    rows = [line.split(',') for line in inventory.splitlines() if line.strip()]
    lookup = {key.strip():(uuid.strip(),int(total),cap.strip()) for index,uuid,total,cap in rows for key in (index,uuid)}
    if any(g not in lookup for g in selected):
        raise RuntimeError('Selected GPU not found')
    uuids = {lookup[g][0] for g in selected}
    if len(uuids)!=2:
        raise RuntimeError('Two identifiers refer to the same GPU')
    if any(lookup[g][1]<65000 for g in selected):
        raise RuntimeError('This preset requires two GPUs with approximately 64 GiB each')
    if any(lookup[g][2]!='8.0' for g in selected):
        raise RuntimeError('This preset is restricted to SM80 compute capability 8.0')
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    for line in processes.splitlines():
        if line.split(',')[0].strip() in uuids:
            raise RuntimeError('Selected GPU is occupied; no existing process was stopped')


def preflight(c):
    data = Path(c['data_dir'])
    if not Path(c.get('model_dir') or data/'models').is_dir():
        raise RuntimeError('Model directory does not exist')
    for name in ['runtime/prepared.json','runtime/gguf-manifest.json','runtime/metadata.json','runtime/model-config/config.json']:
        if not (data/name).is_file():
            raise RuntimeError('Prepare the model loading view first: '+name)
    if c['api_key_file'] and not Path(c['api_key_file']).is_file():
        raise RuntimeError('API key file does not exist')
    check_idle_gpus(c['gpu_ids'])
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        available = int(next(x.split()[1] for x in meminfo.read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        if available<60*1024**3:
            raise RuntimeError('Need at least 60 GiB available host RAM before loading')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',c['port']))


def main():
    if os.name!='posix':
        raise SystemExit('Use Linux or WSL for this deployment launcher; the configuration contains Linux paths')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--start',action='store_true',help='Actually run the foreground container after preflight')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--inside-container',action='store_true')
    args = p.parse_args()
    c = validate(json.loads(args.config.read_text(encoding='utf-8')))
    if args.inside_container:
        if not Path('/.dockerenv').exists():
            raise SystemExit('--inside-container is reserved for the isolated image')
        command,extra = engine_plan(c)
        if args.dry_run:
            print(shlex.join(command));return
        check_idle_gpus(c['gpu_ids'])
        for folder in ['/data/cache','/tmp/q6-sm80-evidence']:
            Path(folder).mkdir(parents=True,exist_ok=True)
        env = os.environ.copy()
        # Prevent inherited GDS/NVFP4 tuning flags from changing this preset.
        for key in list(env):
            if key.startswith(('Q38_','VLLM_PLE_','Q6_')):
                env.pop(key)
        env.update(extra)
        secret = Path('/run/secrets/api_key')
        if c['api_key_file']:
            key = secret.read_text().strip()
            if not key:
                raise SystemExit('Configured API key file is empty')
            env['VLLM_API_KEY']=key
        else:
            env.pop('VLLM_API_KEY',None)
        os.execvpe(command[0],command,env)
    else:
        command = container_plan(c,args.config)
        print(shlex.join(command))
        if args.start and not args.dry_run:
            preflight(c)
            subprocess.run(command,check=True)
        else:
            print('Dry run only. Use --start on an idle machine after reviewing the configuration.')


if __name__=='__main__':
    main()
