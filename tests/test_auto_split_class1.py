import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from PIL import Image

from auto_split_class1 import check_geometry, process_patient, split_class1
from prepare_separate_labels import prepare
from segthor_labels import inspect_dataset


def example():
    labels = np.zeros((24, 20, 12), dtype=np.uint8)
    # Two directly touching organs, with a deliberately unpredicted boundary band.
    labels[2:22, 3:17, 2:10] = 1
    labels[0, 0, :] = 2
    labels[-1, -1, :] = 3
    esophagus, aorta = np.zeros_like(labels), np.zeros_like(labels)
    esophagus[2:11, 3:17, 2:10] = 1
    aorta[13:22, 3:17, 2:10] = 1
    return labels, esophagus, aorta


class SplitTests(unittest.TestCase):
    def test_touching_organs_preserve_union_other_classes_and_identity(self):
        labels, esophagus, aorta = example()
        output, report = split_class1(labels, esophagus, aorta, (1, 1, 2))
        self.assertEqual(output.shape, labels.shape)
        self.assertEqual(output.dtype, np.uint8)
        np.testing.assert_array_equal(np.isin(output, [1, 4]), labels == 1)
        np.testing.assert_array_equal(output[labels != 1], labels[labels != 1])
        self.assertTrue((output[4:9, 5:15, 4:8] == 1).all())
        self.assertTrue((output[15:20, 5:15, 4:8] == 4).all())
        self.assertEqual(set(np.unique(output)), {0, 1, 2, 3, 4})
        self.assertTrue(report['review_recommended'])
        second, _ = split_class1(labels, esophagus, aorta, (1, 1, 2))
        np.testing.assert_array_equal(output, second)

    def test_missing_overlapping_and_unseeded_regions_fail(self):
        labels, esophagus, aorta = example()
        with self.assertRaisesRegex(ValueError, 'aorta.*seed'):
            split_class1(labels, esophagus, np.zeros_like(aorta), (1, 1, 1))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            split_class1(labels, esophagus, esophagus, (1, 1, 1))
        labels[0, 10, 5] = 1
        with self.assertRaisesRegex(ValueError, 'no seeds'):
            split_class1(labels, esophagus, aorta, (1, 1, 1))

    def test_physical_seed_margin_respects_spacing(self):
        labels, esophagus, aorta = example()
        _, small_spacing = split_class1(labels, esophagus, aorta, (.5, .5, .5))
        _, large_spacing = split_class1(labels, esophagus, aorta, (2, 2, 2))
        self.assertGreater(large_spacing['seed_voxels']['esophagus'],
                           small_spacing['seed_voxels']['esophagus'])

    def test_invalid_labels_geometry_and_parameters(self):
        labels, esophagus, aorta = example()
        with self.assertRaises(ValueError):
            split_class1(labels.astype(float) + .1, esophagus, aorta, (1, 1, 1))
        with self.assertRaises(ValueError):
            split_class1(labels, esophagus, aorta, (1, 0, 1))
        with self.assertRaises(ValueError):
            split_class1(labels, esophagus, aorta, (1, 1, 1), -1)
        reference = nib.Nifti1Image(labels, np.eye(4))
        affine = np.eye(4)
        affine[0, 3] = 1
        with self.assertRaisesRegex(ValueError, 'affine'):
            check_geometry(nib.Nifti1Image(labels, affine), reference, 'prediction')


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.predictions, self.output = [self.root / name
                                                     for name in ('source', 'predictions', 'pseudo')]
        self.old, self.dataset = self.root / 'old', self.root / 'dataset'
        self.labels, self.esophagus, self.aorta = example()
        self.affine = np.diag([1., 1., 2., 1.])
        self.affine[0, 3] = -100
        for split, patient in [('train', 'Patient_01'), ('val', 'Patient_02')]:
            source, predictions = self.source / patient, self.predictions / patient
            source.mkdir(parents=True)
            predictions.mkdir(parents=True)
            for name, array in [('GT', self.labels), (patient, self.labels.astype(np.int16))]:
                image = nib.Nifti1Image(array, self.affine)
                image.header.set_xyzt_units('mm')
                nib.save(image, source / f'{name}.nii.gz')
            for name, array in [('esophagus', self.esophagus), ('aorta', self.aorta)]:
                nib.save(nib.Nifti1Image(array, self.affine), predictions / f'{name}.nii.gz')
            for sub in ('gt', 'img'):
                (self.old / split / sub).mkdir(parents=True)
            name = f'{patient}_0005.png'
            Image.fromarray(self.labels[:, :, 5] * 63).save(self.old / split / 'gt' / name)
            Image.fromarray(np.full(self.labels.shape[:2], 110, dtype=np.uint8)).save(
                self.old / split / 'img' / name)

    def test_cached_predictions_export_and_prepare_with_provenance(self):
        source_bytes = (self.source / 'Patient_01' / 'GT.nii.gz').read_bytes()
        with patch('auto_split_class1.generate_predictions', side_effect=AssertionError('inference')):
            for patient in ('Patient_01', 'Patient_02'):
                report = process_patient(self.source, patient, self.output, self.predictions)
                self.assertEqual(report['label_source'], 'pseudo')
        output = nib.load(self.output / 'Patient_01' / 'GT_pseudo.nii.gz')
        np.testing.assert_allclose(output.affine, self.affine)
        self.assertEqual(output.header.get_data_dtype(), np.uint8)
        self.assertEqual(source_bytes, (self.source / 'Patient_01' / 'GT.nii.gz').read_bytes())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            process_patient(self.source, 'Patient_01', self.output, self.predictions)
        with self.assertRaisesRegex(ValueError, 'label-source pseudo'):
            prepare(self.old, self.source, self.output, self.dataset, 'GT_pseudo.nii.gz')
        self.assertFalse(self.dataset.exists())
        prepare(self.old, self.source, self.output, self.dataset,
                'GT_pseudo.nii.gz', label_source='pseudo')
        report = inspect_dataset(self.dataset, require_separate=True)
        self.assertEqual(report['label_source'], 'pseudo')
        for split, patient in [('train', 'Patient_01'), ('val', 'Patient_02')]:
            name = f'{patient}_0005.png'
            self.assertEqual((self.old / split / 'img' / name).read_bytes(),
                             (self.dataset / split / 'img' / name).read_bytes())
            with Image.open(self.dataset / split / 'gt' / name) as png:
                pixels = np.asarray(png)
                self.assertEqual(pixels.shape, self.labels.shape[:2])
                np.testing.assert_array_equal(np.isin(pixels, [63, 252]), self.labels[:, :, 5] == 1)

    def test_failed_prediction_does_not_write_pseudo_mask(self):
        path = self.predictions / 'Patient_01' / 'aorta.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros_like(self.aorta), self.affine), path)
        with self.assertRaisesRegex(ValueError, 'aorta.*seed'):
            process_patient(self.source, 'Patient_01', self.output, self.predictions)
        self.assertFalse((self.output / 'Patient_01' / 'GT_pseudo.nii.gz').exists())

    def test_modified_pseudo_mask_fails_provenance_check(self):
        process_patient(self.source, 'Patient_01', self.output, self.predictions)
        path = self.output / 'Patient_01' / 'pseudo_label_report.json'
        report = json.loads(path.read_text())
        report['output_mask_sha256'] = 'wrong'
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            prepare(self.old, self.source, self.output, self.dataset,
                    'GT_pseudo.nii.gz', label_source='pseudo')
        self.assertFalse(self.dataset.exists())

    def test_cli_records_failed_patient_and_continues_without_overwriting(self):
        (self.predictions / 'Patient_01' / 'aorta.nii.gz').unlink()
        script = Path(__file__).resolve().parents[1] / 'auto_split_class1.py'
        command = [sys.executable, str(script), '--source-dir', str(self.source),
                   '--predictions-dir', str(self.predictions), '--output-dir', str(self.output)]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(summary['patients']['Patient_01']['status'], 'failed')
        self.assertEqual(summary['patients']['Patient_02']['status'], 'generated')
        self.assertFalse((self.output / 'Patient_01' / 'GT_pseudo.nii.gz').exists())
        saved = (self.output / 'Patient_02' / 'GT_pseudo.nii.gz').read_bytes()
        retry = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(retry.returncode, 0)
        self.assertEqual(saved, (self.output / 'Patient_02' / 'GT_pseudo.nii.gz').read_bytes())


if __name__ == '__main__':
    unittest.main()
