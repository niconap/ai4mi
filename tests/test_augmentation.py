import unittest
from unittest.mock import patch
from contextlib import ExitStack
import torch
import torch.nn.functional as F
from augmentation import SliceAugmentation
from dataset import SliceDataset


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        labels = torch.zeros(40, 56, dtype=torch.long)
        labels[10:30, 16:40] = 1
        self.target = F.one_hot(labels, 3).permute(2, 0, 1)
        self.image = labels[None].float()

    def test_baseline_is_exact_identity(self):
        image, target = SliceAugmentation('B0')(self.image, self.target)
        self.assertIs(image, self.image)
        self.assertIs(target, self.target)

    def test_presets_enable_exactly_the_requested_operations(self):
        operations = ('affine', 'adjust_contrast', 'adjust_gamma', 'randn_like',
                      'gaussian_blur', 'interpolate')
        expected_calls = {
            'B0': (0, 0, 0, 0, 0, 0),
            'B1': (2, 0, 0, 0, 0, 0),
            'B2': (0, 1, 1, 0, 0, 0),
            'B3': (0, 0, 0, 1, 0, 0),
            'B4': (0, 0, 0, 0, 1, 2),
        }
        import augmentation

        owners = (augmentation.TF,) * 3 + (torch, augmentation.TF, F)
        for experiment, expected in expected_calls.items():
            with self.subTest(experiment=experiment), ExitStack() as stack:
                stack.enter_context(patch('augmentation.torch.rand',
                                          return_value=torch.tensor(0.0)))
                spies = [stack.enter_context(patch.object(owner, name,
                         wraps=getattr(owner, name)))
                         for owner, name in zip(owners, operations)]
                SliceAugmentation(experiment)(self.image, self.target)
                self.assertEqual(tuple(spy.call_count for spy in spies), expected)

    def test_dataset_records_selected_experiment(self):
        with patch('dataset.make_dataset', return_value=[]):
            for experiment in ('B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'):
                dataset = SliceDataset('train', '/unused', experiment=experiment)
                self.assertEqual(dataset.experiment, experiment)
                self.assertEqual(dataset.augmentation, experiment not in ('B0', 'B5'))

    def test_all_operations_preserve_contract_and_alignment(self):
        # Force every random operation to execute, including both B4 operations.
        for experiment in ('B1', 'B2', 'B3', 'B4'):
            with patch('augmentation.torch.rand', return_value=torch.tensor(0.0)):
                torch.manual_seed(42)
                image, target = SliceAugmentation(experiment)(self.image, self.target)
            self.assertEqual(image.shape, self.image.shape)
            self.assertEqual(target.shape, self.target.shape)
            self.assertEqual(target.dtype, self.target.dtype)
            self.assertTrue(torch.equal(target.sum(0), torch.ones(40, 56)))
            self.assertTrue(((target == 0) | (target == 1)).all())
            self.assertTrue(torch.isfinite(image).all())
            self.assertTrue(((image >= 0) & (image <= 1)).all())
            if experiment == 'B1':
                self.assertLess(((image[0] > .5) != target[1].bool()).float().mean(), .02)
            torch.manual_seed(42)
            with patch('augmentation.torch.rand', return_value=torch.tensor(0.0)):
                image2, target2 = SliceAugmentation(experiment)(self.image, self.target)
            self.assertTrue(torch.equal(image, image2))
            self.assertTrue(torch.equal(target, target2))

    def test_intensity_operations_do_not_change_masks(self):
        for experiment in ('B2', 'B3', 'B4'):
            with patch('augmentation.torch.rand', return_value=torch.tensor(0.)):
                _, target = SliceAugmentation(experiment)(self.image, self.target)
            self.assertTrue(torch.equal(target, self.target))

    def test_nontraining_augmentation_rejected(self):
        for subset in ('val', 'test'):
            with self.assertRaises(ValueError):
                SliceDataset(subset, '/unused', experiment='B1')
        with self.assertRaises(ValueError):
            SliceAugmentation('B7')


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.labels = (torch.arange(256)[None, :].expand(256, 256) // 52).long()
        self.target = F.one_hot(self.labels, 5).permute(2, 0, 1)
        self.image = self.labels[None].float() / 4

    def test_baseline_legacy_bitwise_outputs_and_rng(self):
        from legacy_augmentation import SliceAugmentation as Legacy
        for level in range(1):
            for seed in range(20):
                torch.manual_seed(seed)
                expected = Legacy(f'B{level}')(self.image, self.target)
                rng = torch.get_rng_state()
                torch.manual_seed(seed)
                actual = SliceAugmentation(f'B{level}')(self.image, self.target)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_bilateral_deterministic_edge_preserving(self):
        from augmentation import bilateral_denoise
        rng = torch.get_rng_state()
        actual = bilateral_denoise(self.image)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(actual, bilateral_denoise(self.image)))
        self.assertEqual(actual.shape, self.image.shape)
        self.assertLess((actual - self.image).abs().max(), 1e-6)
        impulse = torch.full_like(self.image, .5)
        impulse[0, 128, 128] = .51
        self.assertLess(bilateral_denoise(impulse)[0, 128, 128], .51)

    def test_elastic_shared_grid_and_discrete_classes(self):
        import augmentation
        with patch.object(F, 'grid_sample', wraps=F.grid_sample) as sample:
            image, target = augmentation.mild_elastic(self.image, self.target)
        self.assertIs(sample.call_args_list[0].args[1], sample.call_args_list[1].args[1])
        self.assertEqual(sample.call_args_list[0].kwargs['mode'], 'bilinear')
        self.assertEqual(sample.call_args_list[1].kwargs['mode'], 'nearest')
        self.assertEqual(sample.call_args_list[0].kwargs['padding_mode'], 'border')
        self.assertEqual(sample.call_args_list[1].kwargs['padding_mode'], 'zeros')
        self.assertEqual(image.shape, self.image.shape)
        self.assertEqual(target.shape, self.target.shape)
        self.assertTrue(((target == 0) | (target == 1)).all())
        self.assertTrue((target.sum(0) == 1).all())
        self.assertTrue(set(target.argmax(0).unique().tolist()) <= set(range(5)))
        grid = sample.call_args_list[0].args[1][0]
        y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='ij')
        delta = torch.stack(((grid[..., 0] + 1) * 128 - .5 - x,
                             (grid[..., 1] + 1) * 128 - .5 - y))
        self.assertLessEqual(delta.square().sum(0).sqrt().max(), 1.0001)

    def test_dataset_order_and_evaluation_without_randomness(self):
        from augmentation import bilateral_denoise
        for experiment in ('B5', 'B6'):
            expected_image = bilateral_denoise(self.image) if experiment == 'B5' else self.image
            for subset in ('train', 'val', 'test'):
                with patch('dataset.make_dataset', return_value=[(__import__('pathlib').Path('ct.png'), 'gt.png')]), patch('dataset.Image.open'):
                    ds = SliceDataset(subset, '/unused', experiment=experiment,
                                      img_transform=lambda _: self.image,
                                      gt_transform=lambda _: self.target)
                    if subset == 'train':
                        with patch.object(ds, 'transform_pair', return_value=(self.image, self.target)) as paired:
                            ds[0]
                        self.assertTrue(torch.equal(paired.call_args.args[0], expected_image))
                    else:
                        rng = torch.get_rng_state()
                        with patch('augmentation.torch.rand', side_effect=AssertionError('random evaluation')):
                            result = ds[0]
                        self.assertFalse(ds.augmentation)
                        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                        self.assertTrue(torch.equal(result['images'], expected_image))
                        if subset == 'val':
                            self.assertTrue(torch.equal(result['gts'], self.target))

    def test_b56_full_pipeline_contract_and_order(self):
        import augmentation
        for experiment in ('B5', 'B6'):
            events = []
            with ExitStack() as stack:
                stack.enter_context(patch('augmentation.torch.rand', return_value=torch.tensor(0.)))
                for owner, name in ((augmentation.TF, 'affine'), (augmentation, 'mild_elastic'),
                                    (augmentation.TF, 'adjust_contrast'), (augmentation.TF, 'adjust_gamma'),
                                    (torch, 'randn_like'), (augmentation.TF, 'gaussian_blur'), (F, 'interpolate')):
                    original = getattr(owner, name)
                    def record(*args, _name=name, _original=original, **kwargs):
                        events.append(_name)
                        return _original(*args, **kwargs)
                    stack.enter_context(patch.object(owner, name, side_effect=record))
                image, target = SliceAugmentation(experiment)(self.image, self.target)
            expected = ['mild_elastic', 'interpolate'] if experiment == 'B6' else []
            self.assertEqual(events, expected)
            self.assertEqual(image.shape, self.image.shape)
            self.assertEqual(target.shape, self.target.shape)
            self.assertTrue(((image >= 0) & (image <= 1)).all())
            self.assertTrue(((target == 0) | (target == 1)).all())
            self.assertTrue((target.sum(0) == 1).all())


    def test_updated_probabilities_and_low_resolution_range(self):
        import augmentation
        operations = ((augmentation.TF, 'affine', .80, 'B1', 0),
                      (augmentation, 'mild_elastic', .05, 'B6', 0),
                      (augmentation.TF, 'adjust_contrast', .50, 'B2', 0),
                      (augmentation.TF, 'adjust_gamma', .50, 'B2', 1),
                      (torch, 'randn_like', .30, 'B3', 0),
                      (augmentation.TF, 'gaussian_blur', .20, 'B4', 0),
                      (F, 'interpolate', .20, 'B4', 1))
        for owner, name, probability, experiment, index in operations:
            for draw, enabled in ((probability - .001, True), (probability + .001, False)):
                draws = [torch.tensor(1.) for _ in range(2)]
                draws[index] = torch.tensor(draw)
                with patch('augmentation.torch.rand', side_effect=draws), patch.object(owner, name, wraps=getattr(owner, name)) as spy, patch('augmentation.uniform', wraps=augmentation.uniform) as uniform_spy:
                    SliceAugmentation(experiment)(self.image, self.target)
                self.assertEqual(spy.called, enabled, (name, draw))
                if name == 'interpolate' and enabled:
                    uniform_spy.assert_called_once_with(.65, .9)

    def test_elastic_caps_upsampled_diagonal_and_preserves_ct_border(self):
        import augmentation
        # Force an oversized final field to verify post-upsampling vector capping.
        with patch.object(F, 'interpolate', return_value=torch.full((1, 2, 256, 256), 2.)), patch.object(F, 'grid_sample', wraps=F.grid_sample) as sample:
            image, target = augmentation.mild_elastic(torch.ones_like(self.image), self.target)
        self.assertTrue(torch.allclose(image, torch.ones_like(image)))
        grid = sample.call_args_list[0].args[1][0]
        y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='ij')
        dx = (grid[..., 0] + 1) * 128 - .5 - x
        dy = (grid[..., 1] + 1) * 128 - .5 - y
        self.assertTrue(torch.allclose((dx.square() + dy.square()).sqrt(), torch.ones_like(dx), atol=3e-5))
        self.assertTrue((target.argmax(0)[-1] == 0).all())
