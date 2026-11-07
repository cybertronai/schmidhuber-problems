"""Real filesystem publication/crash controls with an explicitly fake serializer.

No torch/GPU is imported; native tensor parity remains a separate required gate.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_cuda as publication


class Tensor:
    dtype = 'float32'
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0])) if values and isinstance(values[0], list) else (len(values),)
    def detach(self): return self
    def cpu(self): return self
    def copy_(self, other): self.values = copy.deepcopy(other.values)


class Truth:
    def __init__(self, value): self.value = value
    def all(self): return self
    def item(self): return self.value


class Serializer:
    float32 = 'float32'
    @staticmethod
    def save(state, file):
        file.write(json.dumps({k: v.values for k, v in state.items()}, sort_keys=True).encode())
    @staticmethod
    def load(file, **kwargs):
        assert kwargs == {'map_location': 'cpu', 'weights_only': True}
        return {k: Tensor(v) for k, v in json.load(file).items()}
    @staticmethod
    def equal(a, b): return a.values == b.values
    @staticmethod
    def isfinite(tensor):
        values = tensor.values
        if values and isinstance(values[0], list): values = sum(values, [])
        return Truth(all(math.isfinite(x) for x in values))


class Model:
    torch = Serializer
    sizes = (2, 1)
    def __init__(self):
        self.weights, self.biases = [Tensor([[.1, .2]])], [Tensor([.3])]


def meta(updates=32):
    return {'epoch': 0, 'cursor': updates, 'updates': updates, 'sourceSHA256': 'a'*64}


class CrashPublication(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='checkpoint-fixture-', dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = Model()

    def test_stale_legacy_partials_do_not_block(self):
        (self.root / 'checkpoint-000000032.partial').mkdir()
        (self.root / 'LATEST.partial').write_bytes(b'prior interrupted pointer')
        final = publication.save_checkpoint(self.model, self.root, meta())
        self.assertTrue(final.is_dir())
        self.assertEqual(b'prior interrupted pointer', (self.root / 'LATEST.partial').read_bytes())

    def test_interrupted_pointer_then_identical_commit_reused(self):
        with mock.patch.object(publication.os, 'replace', side_effect=OSError('injected interruption before pointer promotion')):
            self.assertRaises(OSError, publication.save_checkpoint, self.model, self.root, meta())
        final = self.root / 'checkpoint-000000032'
        before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in final.iterdir()}
        self.assertFalse((self.root / 'LATEST.json').exists())
        self.assertEqual(final, publication.save_checkpoint(self.model, self.root, meta()))
        after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in final.iterdir()}
        self.assertEqual(before, after)
        pointer = json.loads((self.root / 'LATEST.json').read_text())
        self.assertEqual(final.name, pointer['checkpoint'])
        self.assertEqual(before['COMMIT.json'], pointer['commitSHA256'])
        self.assertTrue(list(self.root.glob('LATEST.partial-*')))

    def test_abrupt_tensor_write_then_fresh_partial(self):
        with mock.patch.object(Serializer, 'save', side_effect=OSError('injected interrupted payload')):
            self.assertRaises(OSError, publication.save_checkpoint, self.model, self.root, meta())
        partials = list(self.root.glob('checkpoint-*.partial-*'))
        self.assertEqual(1, len(partials))
        final = publication.save_checkpoint(self.model, self.root, meta())
        self.assertTrue(final.is_dir())
        self.assertTrue(partials[0].is_dir())

    def test_resume_then_next_checkpoint_keeps_prior_commit(self):
        first = publication.save_checkpoint(self.model, self.root, meta())
        before = (first / 'COMMIT.json').read_bytes()
        resumed = Model()
        self.assertEqual(meta(), publication.load_checkpoint(resumed, first, {'sourceSHA256': 'a'*64}))
        resumed.weights[0].values[0][0] += .01
        second = publication.save_checkpoint(resumed, self.root, meta(64))
        self.assertNotEqual(first, second)
        self.assertEqual(before, (first / 'COMMIT.json').read_bytes())
        self.assertEqual(second.name, json.loads((self.root / 'LATEST.json').read_text())['checkpoint'])

    def test_changed_same_update_metadata_or_tensor_refuses(self):
        final = publication.save_checkpoint(self.model, self.root, meta())
        original = (final / 'COMMIT.json').read_bytes()
        self.assertRaises(ValueError, publication.save_checkpoint, self.model, self.root, {**meta(), 'sourceSHA256': 'b'*64})
        self.model.biases[0].values[0] += .01
        self.assertRaises(ValueError, publication.save_checkpoint, self.model, self.root, meta())
        self.assertEqual(original, (final / 'COMMIT.json').read_bytes())

    def test_committed_payload_tamper_cannot_be_promoted(self):
        final = publication.save_checkpoint(self.model, self.root, meta())
        (final / 'tensors.pt').write_bytes(b'bad')
        self.assertRaises(ValueError, publication.save_checkpoint, self.model, self.root, meta())

    def test_transient_windows_sharing_retries_same_owned_source_only(self):
        src,dst=self.root/'unique.partial',self.root/'committed'
        src.mkdir();original=publication.os.rename;calls=[];now=[0.]
        error=PermissionError('injected sharing failure');error.winerror=5
        def rename(a,b):
            calls.append((a,b))
            if len(calls)==1:raise error
            return original(a,b)
        def pause(seconds):now[0]+=seconds
        with mock.patch.object(publication.os,'name','nt'),mock.patch.object(publication.os,'rename',side_effect=rename):
            publication._publish_owned(src,dst,replace=False,clock=lambda:now[0],pause=pause)
        self.assertEqual([(src,dst),(src,dst)],calls)
        self.assertTrue(dst.is_dir())

    def test_persistent_sharing_failure_has_finite_deadline(self):
        src,dst=self.root/'unique.partial',self.root/'committed';src.mkdir();now=[0.]
        error=PermissionError('injected sharing failure');error.winerror=32
        def pause(seconds):now[0]+=seconds
        with mock.patch.object(publication.os,'name','nt'),mock.patch.object(publication.os,'rename',side_effect=error):
            self.assertRaises(PermissionError,publication._publish_owned,src,dst,replace=False,clock=lambda:now[0],pause=pause)
        self.assertLessEqual(now[0],.52)
        self.assertTrue(src.is_dir());self.assertFalse(dst.exists())


if __name__ == '__main__': unittest.main()
