import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = load('serve')
installer = load('install_overlay')
fetch = load('fetch_sources')
prepare = load('prepare_model')
compare = load('compare_benchmarks')
audit = load('audit_release')


class ReleaseTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT/'config/example.json').read_text())

    def test_production_preset_and_dynamic_mtp(self):
        c=self.config();args,env=serve.engine_plan(c)
        self.assertEqual(args[args.index('--max-model-len')+1],'262144')
        self.assertEqual(args[args.index('--max-num-batched-tokens')+1],'1024')
        spec=json.loads(args[args.index('--speculative-config')+1])
        self.assertEqual(spec['num_speculative_tokens_per_batch_size'],[[1,1,2],[2,2,1]])
        self.assertEqual(env['VLLM_PLE_CPU_OFFLOAD'],'1')
        self.assertEqual(env['Q6_MAX_WAITING'],'4')
        self.assertEqual(env['Q38_RAM_COMPUTE_OPT'],'0')

    def test_single_slot_does_not_enable_mixed_guard(self):
        c=self.config();c.update(max_num_seqs=1,dynamic_mtp=False)
        args,env=serve.engine_plan(c)
        self.assertEqual(env['Q6_MIXED_PREFILL_GUARD'],'0')
        self.assertNotIn('num_speculative_tokens_per_batch_size',json.loads(args[args.index('--speculative-config')+1]))

    def test_unvalidated_presets_rejected(self):
        for updates in [dict(max_num_batched_tokens=2048),dict(gpu_memory_utilization=.99),
                        dict(pp_partition=[27,21]),dict(gpu_ids=['0','0']),
                        dict(max_num_seqs=1),dict(max_num_seqs=True)]:
            c=self.config();c.update(updates)
            with self.subTest(updates=updates),self.assertRaises(ValueError):serve.validate(c)

    def test_secret_is_mounted_without_reading_or_printing_contents(self):
        c=self.config();c['api_key_file']='/run/user/key.txt'
        with patch.object(Path,'read_text',side_effect=AssertionError('Must not read secret')):
            command=serve.container_plan(c,Path('/tmp/config.json'))
        self.assertTrue(any('target=/run/secrets/api_key,readonly' in part for part in command))
        self.assertIn('127.0.0.1:18421:8000',command)
        self.assertNotIn('--privileged',command)

    def test_busy_gpu_rejected_without_stopping_workload(self):
        outputs=['0, GPU-aaa, 65536, 8.0\n1, GPU-bbb, 65536, 8.0\n','GPU-bbb, 123\n']
        with patch.object(serve.subprocess,'check_output',side_effect=outputs) as commands:
            with self.assertRaisesRegex(RuntimeError,'occupied'):serve.check_idle_gpus(['0','1'])
            self.assertEqual(len(commands.call_args_list),2)
            self.assertTrue(all(call.args[0][0]=='nvidia-smi' for call in commands.call_args_list))

    def test_overlay_checks_every_file_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);a=r/'src';b=r/'dest';a.mkdir();b.mkdir()
            (a/'first.py').write_text('new');(b/'first.py').write_text('old')
            (a/'second.py').write_text('new2');(b/'second.py').write_text('foreign')
            h=lambda s:hashlib.sha256(s.encode()).hexdigest()
            manifest={'first.py':dict(before_sha256=h('old'),after_sha256=h('new')),
                      'second.py':dict(before_sha256=h('old2'),after_sha256=h('new2'))}
            with self.assertRaisesRegex(ValueError,'incompatible'):installer.plan_overlay(b,a,manifest)
            self.assertEqual((b/'first.py').read_text(),'old')

    def test_archive_traversal_and_symlink_rejected(self):
        for name,kind in [('root/../../outside',tarfile.REGTYPE),('root/link',tarfile.SYMTYPE)]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);archive=root/'bad.tar'
                with tarfile.open(archive,'w') as tar:
                    member=tarfile.TarInfo(name);member.type=kind
                    if kind==tarfile.SYMTYPE:member.linkname='/etc/passwd'
                    tar.addfile(member,io.BytesIO())
                with self.assertRaises(ValueError):fetch.extract_checked(archive,root/'out')

    def test_source_fingerprints_match_packaged_files(self):
        selected={p.relative_to(ROOT).as_posix() for p in audit.files(ROOT)}
        for manifest,source in [('overlay-files.json','runtime/overlay'),('plugin-files.json','runtime/plugin-overlay')]:
            entries=json.loads((ROOT/'provenance'/manifest).read_text())
            for rel,item in entries.items():
                self.assertIn(f'{source}/{rel}',selected,'Required overlay missing from release selection')
                self.assertEqual(installer.digest(ROOT/source/rel),item['after_sha256'],rel)

    def test_release_selection_keeps_model_code_and_excludes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            keep='runtime/overlay/vllm/models/example/model.py'
            for rel in [keep,'models/weights.gguf','build/source.py','dist/release.zip',
                        'src/__pycache__/module.pyc','.git/config']:
                path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
            self.assertEqual({p.relative_to(root).as_posix() for p in audit.files(root)},{keep})

    def test_model_view_rejects_unrelated_architecture(self):
        with self.assertRaisesRegex(ValueError,'hidden_size'):prepare.text_config({'hidden_size':4096})

    def test_comparison_rejects_changed_sampling(self):
        a=dict(model='test',sampling=dict(temperature=1),max_tokens=512,prompt_sha256='same',metric='same',runs=[])
        b={**a,'sampling':dict(temperature=.7)}
        with self.assertRaisesRegex(ValueError,'sampling'):compare.compare(a,b)

    def test_comparison_reports_output_difference(self):
        base=dict(model='test',sampling=dict(temperature=1),max_tokens=512,prompt_sha256='same',metric='same')
        a={**base,'runs':[dict(prompt_id='a',decode_tps=100,text_sha256='first')]}
        b={**base,'runs':[dict(prompt_id='a',decode_tps=110,text_sha256='second')]}
        result=compare.compare(a,b)
        self.assertAlmostEqual(result['mean_change_percent'],10)
        self.assertFalse(result['prompts'][0]['all_outputs_identical'])

    def test_audit_rejects_sensitive_fixture_without_echoing_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            token='hf_'+'x'*30
            (root/'private.txt').write_text(token+'\n'+'192'+'.168.2.3')
            result=audit.audit(root)
            self.assertEqual({i['kind'] for i in result['issues']},{'huggingface-token','private-network-address'})
            self.assertNotIn(token,json.dumps(result))


if __name__=='__main__':unittest.main()
