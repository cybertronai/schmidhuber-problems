"""Tiny stdlib-only offline controls; never imports NumPy/torch or real MNIST."""
import copy
import gzip
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_online import PaperConfig, ReferenceMLP, bilinear, epoch_order, smoke, targets
from paper_inputs import idx


class OfflineControls(unittest.TestCase):
    def test_paper_parameters_and_disclosed_resize_difference(self):
        cfg = PaperConfig()
        cfg.validate()
        self.assertEqual(12115010, cfg.parameters)
        self.assertEqual(12107500, cfg.matrix_weights)
        self.assertEqual(7510, cfg.bias_parameters)
        self.assertEqual(11972510, PaperConfig(side=28).parameters)
        self.assertRaises(ValueError, PaperConfig(side=28).validate)

    def test_frozen_schedule_endpoints_and_bounds(self):
        cfg = PaperConfig()
        self.assertEqual(.001, cfg.learning_rate(0))
        self.assertAlmostEqual(.000001, cfg.learning_rate(799), places=20)
        self.assertTrue(all(cfg.learning_rate(e+1) < cfg.learning_rate(e) for e in range(799)))
        self.assertRaises(ValueError, cfg.learning_rate, -1)
        self.assertRaises(ValueError, cfg.learning_rate, 800)

    def test_config_cannot_silently_change_scientific_constants(self):
        self.assertRaises(ValueError, PaperConfig(alpha=8).validate)
        self.assertRaises(ValueError, PaperConfig(graph_chunk=64).validate)
        self.assertEqual(PaperConfig().digest(), PaperConfig().digest())
        self.assertNotEqual(PaperConfig().digest(), PaperConfig(seed=1).digest())

    def test_offline_smoke(self):
        result = smoke()
        self.assertTrue(result['offlineSmokePassed'])
        self.assertEqual(8, result['updates'])
        self.assertFalse(result['gpu'])

    def test_target_domain(self):
        self.assertEqual([-1., 1., -1.], targets(1, 3))
        self.assertRaises(ValueError, targets, -1)
        self.assertRaises(ValueError, targets, 10)

    def test_initialization_seed_and_bounds(self):
        a, b = ReferenceMLP([3, 4, 2], 9), ReferenceMLP([3, 4, 2], 9)
        self.assertEqual(a.weights, b.weights)
        self.assertEqual(a.biases, b.biases)
        self.assertNotEqual(a.weights, ReferenceMLP([3, 4, 2], 10).weights)
        self.assertTrue(all(-.05 <= v <= .05 for layer in a.weights for row in layer for v in row))
        self.assertTrue(all(-.05 <= v <= .05 for layer in a.biases for v in layer))

    def test_all_layers_include_scaled_tanh(self):
        m = ReferenceMLP([1, 1, 1])
        m.weights, m.biases = [[[2.]], [[3.]]], [[.1], [.2]]
        h = 1.7159 * math.tanh(.6666 * .7)
        y = 1.7159 * math.tanh(.6666 * (3*h + .2))
        self.assertEqual([.3], m.forward([.3])[0])
        self.assertAlmostEqual(h, m.forward([.3])[1][0])
        self.assertAlmostEqual(y, m.forward([.3])[2][0])

    def test_every_weight_and_bias_gradient_by_finite_difference(self):
        m = ReferenceMLP([2, 3, 2], 5)
        x, y, eps = [.31, -.43], [1., -1.], 1e-6
        dw, db = m.gradients(x, y)
        for l, layer in enumerate(m.weights):
            for i, row in enumerate(layer):
                for j, value in enumerate(row):
                    row[j] = value + eps
                    hi = m.loss(x, y)
                    row[j] = value - eps
                    lo = m.loss(x, y)
                    row[j] = value
                    self.assertAlmostEqual(dw[l][i][j], (hi-lo)/(2*eps), delta=3e-10)
        for l, bias in enumerate(m.biases):
            for i, value in enumerate(bias):
                bias[i] = value + eps
                hi = m.loss(x, y)
                bias[i] = value - eps
                lo = m.loss(x, y)
                bias[i] = value
                self.assertAlmostEqual(db[l][i], (hi-lo)/(2*eps), delta=3e-10)

    def test_step_uses_all_old_weights_for_deltas(self):
        m = ReferenceMLP([2, 3, 2], 3)
        original = copy.deepcopy(m)
        x, y, lr = [.7, -.2], targets(0, 2), .01
        dw, db = original.gradients(x, y)
        m.step(x, y, lr)
        for layer, old, grad in zip(m.weights, original.weights, dw):
            self.assertEqual(layer, [[v-lr*g for v, g in zip(row, drow)] for row, drow in zip(old, grad)])
        for b, old, grad in zip(m.biases, original.biases, db):
            self.assertEqual(b, [v-lr*g for v, g in zip(old, grad)])

    def test_sequential_examples_are_not_a_minibatch(self):
        a = ReferenceMLP([2, 2], 2)
        b = copy.deepcopy(a)
        data = [([.5, -.1], targets(0, 2)), ([-.3, .6], targets(1, 2))]
        for x, y in data:
            a.step(x, y, .1)
        for x, y in reversed(data):
            b.step(x, y, .1)
        self.assertNotEqual(a.weights, b.weights)

    def test_seed_cursor_resume_equals_uninterrupted(self):
        data = [([i/10, -i/20], targets(i % 2, 2)) for i in range(9)]
        order = epoch_order(3, 0, len(data))
        uninterrupted = ReferenceMLP([2, 3, 2], 3)
        for i in order:
            uninterrupted.step(*data[i], .001)
        resumed = ReferenceMLP([2, 3, 2], 3)
        for i in order[:4]:
            resumed.step(*data[i], .001)
        # Lossless pure scalar oracle checkpoint; real tensor serialization is
        # deliberately a distinct native pilot gate, not claimed by this test.
        state = json.loads(json.dumps({'weights': resumed.weights, 'biases': resumed.biases, 'seed': 3, 'epoch': 0, 'cursor': 4}))
        resumed.weights, resumed.biases = state['weights'], state['biases']
        for i in epoch_order(state['seed'], state['epoch'], len(data))[state['cursor']:]:
            resumed.step(*data[i], .001)
        self.assertEqual(uninterrupted.weights, resumed.weights)
        self.assertEqual(uninterrupted.biases, resumed.biases)

    def test_private_rng_order_is_deterministic_complete_and_epoch_distinct(self):
        order = epoch_order(0, 0, 50)
        self.assertEqual(list(range(50)), sorted(order))
        self.assertEqual(order, epoch_order(0, 0, 50))
        self.assertNotEqual(order, epoch_order(0, 1, 50))

    def test_bilinear_identity_middle_and_zero_background(self):
        image = [[0., 1.], [2., 3.]]
        self.assertEqual(3., bilinear(image, 1, 1))
        self.assertEqual(1.5, bilinear(image, .5, .5))
        self.assertEqual(0., bilinear(image, -2, .5))
        self.assertEqual(.5, bilinear(image, 1.5, 0))
        self.assertRaises(ValueError, bilinear, image, float('nan'), 0)

    def test_nonfinite_and_wrong_shape_controls(self):
        m = ReferenceMLP([2, 3, 2])
        self.assertRaises(ValueError, m.step, [1., float('inf')], [1., -1.], .001)
        self.assertRaises(ValueError, m.step, [1., 0.], [1.], .001)
        self.assertRaises(ValueError, m.step, [1., 0.], [1., -1.], float('nan'))
        self.assertRaises(ValueError, ReferenceMLP, [1000, 1000])

    def test_idx_exact_headers_payload_eof_and_label_domain(self):
        with tempfile.TemporaryDirectory(prefix='paper-control-', dir=HERE) as temp:
            path = Path(temp) / 'fixture.gz'
            for content, images, good in [
                (struct.pack('>IIII', 2051, 1, 28, 28) + bytes(784), True, True),
                (struct.pack('>IIII', 2051, 1, 28, 28) + bytes(785), True, False),
                (struct.pack('>IIII', 2051, 1, 29, 28) + bytes(784), True, False),
                (struct.pack('>II', 2049, 1) + bytes([9]), False, True),
                (struct.pack('>II', 2049, 1) + bytes([10]), False, False),
                (struct.pack('>II', 2049, 1), False, False),
            ]:
                path.write_bytes(gzip.compress(content))
                if good:
                    self.assertEqual(784 if images else 1, len(idx(path, 1, images)))
                else:
                    self.assertRaises(ValueError, idx, path, 1, images)

    def test_optional_cuda_module_import_does_not_load_torch(self):
        self.assertNotIn('torch', sys.modules)
        spec = importlib.util.spec_from_file_location('paper_cuda_optional', HERE / 'paper_cuda.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
