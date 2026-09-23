import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from prepare_separate_labels import prepare
from plot_class_comparison import read_scores
from segthor_labels import decode_png, inspect_dataset, require_separate_volume


class SeparateLabelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / 'old'
        self.sources = self.root / 'source'
        self.correct = self.root / 'correct'
        self.output = self.root / 'separate'
        self.labels = np.broadcast_to(
            np.arange(5, dtype=np.uint8)[None, :, None], (6, 5, 2)).copy()
        merged = self.labels.copy()
        merged[merged == 4] = 1
        for split, patient in [('train', 'Patient_01'), ('val', 'Patient_02')]:
            for folder, array in [(self.sources, merged), (self.correct, self.labels)]:
                (folder / patient).mkdir(parents=True)
                nib.save(nib.Nifti1Image(array, np.eye(4)), folder / patient / 'GT.nii.gz')
            for sub in ['img', 'gt']:
                (self.old / split / sub).mkdir(parents=True)
            for index in range(2):
                name = f'{patient}_{index:04d}.png'
                Image.fromarray(merged[:, :, index] * 63).save(self.old / split / 'gt' / name)
                Image.fromarray(np.full((6, 5), 110, dtype=np.uint8)).save(
                    self.old / split / 'img' / name)

    def test_original_masks_restore_two_classes_and_preserve_cts(self):
        prepare(self.old, self.sources, self.correct, self.output)
        schema = inspect_dataset(self.output, require_separate=True)
        self.assertEqual(schema['schema'], 'separate')
        for split, patient in [('train', 'Patient_01'), ('val', 'Patient_02')]:
            for index in range(2):
                name = f'{patient}_{index:04d}.png'
                with Image.open(self.output / split / 'gt' / name) as image:
                    np.testing.assert_array_equal(decode_png(np.asarray(image)), self.labels[:, :, index])
                self.assertEqual((self.old / split / 'img' / name).read_bytes(),
                                 (self.output / split / 'img' / name).read_bytes())

    def test_merged_sources_fail_without_creating_output(self):
        with self.assertRaisesRegex(ValueError, 'Aorta'):
            prepare(self.old, self.sources, self.sources, self.output)
        self.assertFalse(self.output.exists())

    def test_one_missing_patient_blocks_all_output(self):
        (self.correct / 'Patient_02' / 'GT.nii.gz').unlink()
        with self.assertRaisesRegex(ValueError, 'Patient_02'):
            prepare(self.old, self.sources, self.correct, self.output)
        self.assertFalse(self.output.exists())

    def test_misaligned_source_rejected(self):
        affine = np.eye(4)
        affine[0, 3] = 10
        nib.save(nib.Nifti1Image(self.labels, affine), self.correct / 'Patient_01' / 'GT.nii.gz')
        with self.assertRaisesRegex(ValueError, 'affine'):
            prepare(self.old, self.sources, self.correct, self.output)
        self.assertFalse(self.output.exists())

    def test_check_only_does_not_write(self):
        prepare(self.old, self.sources, self.correct, self.output, check_only=True)
        self.assertFalse(self.output.exists())

    def test_training_check_rejects_merged_data(self):
        self.assertEqual(inspect_dataset(self.old)['schema'], 'merged')
        with self.assertRaisesRegex(ValueError, 'Separate'):
            inspect_dataset(self.old, require_separate=True)

    def test_bad_png_and_fractional_nifti_labels_rejected(self):
        with self.assertRaises(ValueError):
            decode_png(np.array([[0, 64, 252]], dtype=np.uint8))
        with self.assertRaises(ValueError):
            require_separate_volume(self.labels.astype(float) + .1)

    def test_plot_accepts_only_separate_organ_scores(self):
        run = self.root / 'run'
        run.mkdir()
        (run / 'best_epoch.txt').write_text('Improved dice at epoch 0: 0.0->0.8 DSC')
        dice = np.full((1, 2, 5), .8, dtype=np.float32)
        np.save(run / 'dice_val.npy', dice)
        with self.assertRaisesRegex(ValueError, 'Retrain'):
            read_scores(run)
        (run / 'label_schema.json').write_text(json.dumps(inspect_dataset(self.old)))
        with self.assertRaisesRegex(ValueError, 'not separate'):
            read_scores(run)
        prepare(self.old, self.sources, self.correct, self.output)
        (run / 'label_schema.json').write_text((self.output / 'label_schema.json').read_text())
        _, scores, _ = read_scores(run)
        np.testing.assert_allclose(scores, [.8] * 4)
        schema = json.loads((run / 'label_schema.json').read_text())
        schema['splits']['val']['pixels'][4] = 0
        (run / 'label_schema.json').write_text(json.dumps(schema))
        with self.assertRaisesRegex(ValueError, 'absent'):
            read_scores(run)


if __name__ == '__main__':
    unittest.main()
