"""脚手架生成、滚动保留与恢复训练的行为测试。"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'assets/project-template'
spec = importlib.util.spec_from_file_location('artifacts', TEMPLATE / 'utils/artifacts.py')
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


class RetentionTests(unittest.TestCase):
    def test_keeps_best_last_and_latest_periodic_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for epoch in [1, 2, 9999, 10000]:
                artifacts.save_evaluation({'epoch': epoch}, path, epoch == 2, True, 2)
            self.assertEqual({p.name for p in path.iterdir()},
                             {'best.json', 'last.json', 'epoch_9999.json', 'epoch_10000.json'})
            self.assertEqual(json.loads((path / 'best.json').read_text())['epoch'], 2)
            self.assertEqual(json.loads((path / 'last.json').read_text())['epoch'], 10000)

    def test_failed_write_preserves_previous_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.json'
            artifacts.write_json_atomic(path, {'epoch': 1})
            with self.assertRaises(ValueError):
                artifacts.write_json_atomic(path, {'metric': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'epoch': 1})
            self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_pruning_scope_and_unlimited_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name in ['epoch_0001.pt', 'epoch_0002.pt', 'best.pt', 'last.pt', 'notes.pt']:
                (path / name).touch()
            self.assertEqual(artifacts.prune_snapshots(path, 0, '*.pt'), [])
            with self.assertRaises(ValueError):
                artifacts.prune_snapshots(path, -1, '*.pt')
            artifacts.prune_snapshots(path, 1, '*.pt')
            self.assertEqual({p.name for p in path.iterdir()},
                             {'epoch_0002.pt', 'best.pt', 'last.pt', 'notes.pt'})


class ConfigurationTests(unittest.TestCase):
    def test_rejects_invalid_configuration(self):
        import copy
        import yaml
        spec = importlib.util.spec_from_file_location('config', TEMPLATE / 'utils/config.py')
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        base = yaml.safe_load((TEMPLATE / 'configs/default.yaml').read_text())
        config.validate_config(base)
        for key, value in [('epohcs', 10), ('epochs', 0), ('batch_size', 1.5),
                           ('log_interval', 0), ('ckpt_keep', -1), ('val_split', float('nan'))]:
            cfg = copy.deepcopy(base)
            cfg['train'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                config.validate_config(cfg)

    def test_ablation_paths_are_distinct_and_existing_files_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, str(TEMPLATE / 'scripts/make_ablation.py'),
                   '--base', str(TEMPLATE / 'configs/default.yaml'), '--out-dir', tmp,
                   '--mode', 'oat', '--grid', 'dataset.seed=1', 'experiment.seed=1']
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            before = {p.name: p.read_bytes() for p in Path(tmp).glob('*.yaml')}
            self.assertEqual(len(before), 2)
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(tmp).glob('*.yaml')})


class ScaffoldTests(unittest.TestCase):
    def test_rejects_parent_path_without_deleting_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / 'sentinel'
            sentinel.write_text('keep')
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/new_project.py'),
                                     '..', '--dest', tmp, '--force'], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), 'keep')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
    def test_training_resume_evaluation_and_retention(self):
        import torch
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / 'example'
            env = dict(os.environ, MPLCONFIGDIR=str(Path(tmp) / 'mpl'))

            def run(*args, ok=True):
                result = subprocess.run([sys.executable, *map(str, args)], cwd=project if project.exists() else ROOT,
                                        env=env, capture_output=True, text=True, timeout=120)
                if ok:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result

            run(ROOT / 'scripts/new_project.py', 'example', '--dest', tmp)
            cfg = yaml.safe_load((project / 'configs/default.yaml').read_text())
            cfg['dataset']['n_samples'] = 32
            cfg['model']['hidden_dim'] = 8
            cfg['train'].update(epochs=4, batch_size=8, device='cpu', ckpt_interval=1,
                                ckpt_keep=2, eval_interval=1, eval_keep=2)
            (project / 'configs/test.yaml').write_text(yaml.safe_dump(cfg))
            mismatch = run('train.py', '--config', 'configs/test.yaml', '--set',
                           'model.out_dim=2', 'experiment.output_root=invalid-runs', ok=False)
            self.assertIn('形状不一致', mismatch.stderr)
            self.assertFalse(list((project / 'invalid-runs').rglob('*.pt')))
            run('train.py', '--config', 'configs/test.yaml')
            continuous = next(p for p in (project / 'experiments').iterdir() if p.is_dir())
            run('train.py', '--config', 'configs/test.yaml', '--set', 'train.epochs=2')
            resumed = next(p for p in (project / 'experiments').iterdir() if p.is_dir() and p != continuous)
            run('train.py', '--resume', resumed, '--set', 'train.epochs=4')
            a = torch.load(continuous / 'checkpoints/last.pt', weights_only=False)
            b = torch.load(resumed / 'checkpoints/last.pt', weights_only=False)
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
            self.assertEqual(len(list((resumed / 'checkpoints').glob('epoch_*.pt'))), 2)
            self.assertEqual(len(list((resumed / 'evaluations').glob('epoch_*.json'))), 2)
            self.assertEqual(json.loads((resumed / 'evaluations/best.json').read_text())['epoch'], b['best_epoch'])
            self.assertTrue((resumed / 'config.initial.yaml').is_file())
            before = (resumed / 'config.yaml').read_text()
            run('train.py', '--resume', resumed, '--set', 'train.optimizer.lr=0.1', ok=False)
            self.assertEqual((resumed / 'config.yaml').read_text(), before)
            altered = yaml.safe_load(before)
            altered['train']['optimizer']['lr'] = 0.1
            (resumed / 'config.yaml').write_text(yaml.safe_dump(altered))
            run('train.py', '--resume', resumed, ok=False)
            self.assertEqual(yaml.safe_load((resumed / 'config.yaml').read_text()), altered)
            (resumed / 'config.yaml').write_text(before)
            run('-c', 'from utils.checkpoint import BestTracker; '
                'b=BestTracker(); b.update(0.5); '
                '\nfor v in [float("nan"), float("inf")]:\n'
                ' try: b.update(v)\n'
                ' except ValueError: pass\n'
                ' else: raise AssertionError("non-finite metric accepted")\n'
                ' assert b.best == 0.5')
            run('eval.py', '--exp-dir', resumed)
            val = (resumed / 'eval.json').read_text()
            run('eval.py', '--exp-dir', resumed, '--split', 'train')
            self.assertEqual((resumed / 'eval.json').read_text(), val)
            self.assertTrue((resumed / 'eval_train_best.json').is_file())
            run('scripts/analyze.py', 'summary', '--exp-root', 'experiments')
            run('scripts/analyze.py', 'curves', '--exp-root', 'experiments')
            run('scripts/analyze.py', 'sweep', '--exp-root', 'experiments',
                '--x', 'model.depth', '--x2', 'model.hidden_dim', '--y', 'best_val_mse')
            run('-c', 'from train import build_dataloaders; from utils import load_config; '
                'c=load_config("configs/test.yaml"); '
                'a=build_dataloaders(c,42); b=build_dataloaders(c,43); '
                'assert a[0].dataset.indices == b[0].dataset.indices')


if __name__ == '__main__':
    unittest.main()
