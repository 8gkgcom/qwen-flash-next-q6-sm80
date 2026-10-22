#!/usr/bin/env python3
"""Create a text-only loading view from read-only GGUF files; no CUDA use."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HASH_KEYS = ('qwen4exp.ple.layer_multipliers', 'qwen4exp.ple.head_offsets', 'qwen4exp.ple.head_vocab_sizes')


def text_config(config):
    cfg = dict(config.get('text_config',config))
    expected = {'hidden_size':2560,'num_hidden_layers':48,'num_experts':512,
                'num_experts_per_tok':10,'vocab_size':248320,'heads_per_ngram':8,
                'ngram_size':3,'hc_count':4,'ple_layer_ids':[2]}
    for key,value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'Unsupported model configuration: {key}')
    cfg.update(architectures=['Qwen3_8FlashNextForCausalLM'],
               model_type='qwen3_8_flash_next_text',tie_word_embeddings=False,
               ple_embedding_dtype='float8_e4m3fn')
    # The legacy PLE dtype field is retained for config parity; Q8PLEEmbedding
    # reads Q8_0 rows and returns BF16. It never converts this table to FP8.
    cfg.pop('quantization_config',None)
    cfg.pop('compression_config',None)
    return cfg


def fetch_metadata(folder, lock):
    folder.mkdir(parents=True,exist_ok=True)
    for name,expected in lock['files'].items():
        target = folder/name
        if target.exists():
            data = target.read_bytes()
        else:
            url = f'https://huggingface.co/{lock["repository"]}/resolve/{lock["revision"]}/{name}'
            data = urllib.request.urlopen(url,timeout=120).read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Metadata checksum mismatch: '+name)
        if not target.exists():
            target.write_bytes(data)


def inspect_gguf(folder, model_lock):
    import gguf
    main = sorted(folder.glob(model_lock['main_pattern']))
    draft = folder/model_lock['draft_file']
    expected_names = [f'Qwen3.8-Flash-Next-Uncensored-Q6_K-{i:05d}-of-00005.gguf' for i in range(1,6)]
    if [p.name for p in main] != expected_names or not draft.is_file():
        raise ValueError('Expected all five Q6_K main shards and the separate MTP draft')
    entries,metadata,seen = [],{},set()
    for path in [*main,draft]:
        reader = gguf.GGUFReader(path,mode='r')
        if reader.get_field('general.architecture').contents() != 'qwen4exp':
            raise ValueError('Unexpected GGUF architecture')
        role = 'draft' if path==draft else 'target'
        if role=='target':
            for key,field in reader.fields.items():
                if key.startswith('qwen4exp.') or key=='general.architecture':
                    value = field.contents()
                    if key in metadata and metadata[key]!=value:
                        raise ValueError('Conflicting GGUF metadata: '+key)
                    metadata[key] = value
        for tensor in reader.tensors:
            identity = (role,tensor.name)
            if identity in seen:
                raise ValueError('Duplicate tensor: '+tensor.name)
            seen.add(identity)
            end = int(tensor.data_offset)+int(tensor.n_bytes)
            if end > path.stat().st_size:
                raise ValueError('Truncated GGUF: '+path.name)
            entries.append(dict(file=str(path.resolve()),name=tensor.name,
                shape=tensor.shape.tolist(),type=tensor.tensor_type.name,
                qtype=int(tensor.tensor_type),offset=int(tensor.data_offset),
                bytes=int(tensor.n_bytes),role=role))
        del reader
    for key in HASH_KEYS:
        if key not in metadata or not isinstance(metadata[key],list) or not all(type(n) is int for n in metadata[key]):
            raise ValueError('Missing exact integer PLE hash constants: '+key)
    for key,value in {'qwen4exp.block_count':48,'qwen4exp.embedding_length':2560,
                      'qwen4exp.expert_count':512}.items():
        if metadata.get(key)!=value:
            raise ValueError('Unsupported GGUF configuration: '+key)
    ple = [e for e in entries if e['name']=='per_layer_token_embd.weight']
    if len(ple)!=1 or ple[0]['role']!='target' or ple[0]['type']!='Q8_0' or ple[0]['shape'][0]!=160:
        raise ValueError('Expected one target Q8_0 PLE table of row width 160')
    target_layers = {int(e['name'].split('.')[1]) for e in entries if e['role']=='target' and e['name'].startswith('blk.')}
    if target_layers!=set(range(48)):
        raise ValueError('Incomplete target layer coverage')
    if not any(e['role']=='draft' and e['name'].startswith('blk.48.') for e in entries):
        raise ValueError('Missing MTP block 48')
    return entries,metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=Path('/data/models'))
    p.add_argument('--metadata-dir',type=Path,default=Path('/data/metadata'))
    p.add_argument('--output',type=Path,default=Path('/data/runtime'))
    p.add_argument('--download-metadata',action='store_true')
    args = p.parse_args()
    model_dir,output = args.model_dir.resolve(),args.output.resolve()
    if output.exists():
        raise SystemExit('Output already exists; choose a new directory to preserve the previous loading view')
    if output.is_relative_to(model_dir) or model_dir.is_relative_to(output):
        raise SystemExit('Model weights and prepared output must be separate directories')
    lock = json.loads((ROOT/'dependencies.lock.json').read_text())
    if args.download_metadata:
        fetch_metadata(args.metadata_dir,lock['metadata'])
    for name,expected in lock['metadata']['files'].items():
        if hashlib.sha256((args.metadata_dir/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Metadata checksum mismatch: '+name)
    cfg = text_config(json.loads((args.metadata_dir/'config.json').read_text()))
    entries,metadata = inspect_gguf(model_dir,lock['model'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='q6-prepare-',dir=output.parent) as tmp:
        stage = Path(tmp)/'runtime'
        (stage/'model-config').mkdir(parents=True)
        (stage/'model-config/config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        for name in lock['metadata']['files']:
            if name!='config.json':
                shutil.copyfile(args.metadata_dir/name,stage/'model-config'/name)
        (stage/'gguf-manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
        (stage/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (stage/'prepared.json').write_text(json.dumps(dict(model=lock['model'],metadata=lock['metadata'],
            tensors=len(entries),payload_hashes_verified=False,
            validation='pinned metadata hashes; tensor headers, file ranges, layer coverage; weights are not rewritten'),indent=2)+'\n')
        os.rename(stage,output)
    print(json.dumps({'prepared':str(output),'tensors':len(entries),'cuda_used':False}))


if __name__=='__main__':
    main()
