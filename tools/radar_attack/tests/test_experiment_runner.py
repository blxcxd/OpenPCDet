import json
import tempfile
import unittest
from pathlib import Path

from tools.radar_attack.experiments.runner import (
    Campaign,
    Experiment,
    build_attack_command,
    load_campaign,
    result_matches_experiment,
    result_to_row,
    write_summaries,
)


class ExperimentRunnerTest(unittest.TestCase):
    def test_load_campaign_rejects_duplicate_names(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            campaign_path = Path(temporary_dir) / 'campaign.yaml'
            campaign_path.write_text(
                '\n'.join(
                    (
                        'version: 1',
                        'name: duplicate_test',
                        'common:',
                        '  cfg_file: cfgs/model.yaml',
                        '  ckpt: model.pth',
                        'experiments:',
                        '  - name: same',
                        '  - name: same',
                    )
                ),
                encoding='utf-8',
            )

            with self.assertRaisesRegex(ValueError, 'must be unique'):
                load_campaign(campaign_path)

    def test_build_attack_command_places_set_last(self):
        experiment = Experiment(
            name='pgd_xyz',
            parameters={
                'cfg_file': 'cfgs/model.yaml',
                'ckpt': '/tmp/model.pth',
                'attack_type': 'pgd',
                'epsilon_xyz': 0.02,
                'random_start': True,
                'no_vod_eval': True,
                'set': ['MODEL.POST_PROCESSING.SCORE_THRESH', '0.0'],
            },
            extra_tag='campaign/pgd_xyz',
            output_dir=Path('/tmp/output'),
        )

        command = build_attack_command(
            experiment,
            python_executable='/env/python',
            attack_entry=Path('/repo/run_attack.py'),
        )

        self.assertEqual(command[:2], ['/env/python', '/repo/run_attack.py'])
        self.assertIn('--random_start', command)
        self.assertIn('--no_vod_eval', command)
        self.assertEqual(
            command[-3:],
            ['--set', 'MODEL.POST_PROCESSING.SCORE_THRESH', '0.0'],
        )
        extra_tag_index = command.index('--extra_tag')
        self.assertLess(extra_tag_index, command.index('--set'))

    def test_result_summary_includes_vod_map(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / 'campaign' / 'fgsm_xyz'
            output_dir.mkdir(parents=True)
            result = {
                'attack': {
                    'extra_tag': 'campaign/fgsm_xyz',
                    'attack_domain': 'point',
                'attack_type': 'fgsm',
                'attack_feature': 'xyz',
                'epsilon': 0.05,
                'epsilon_xyz': 0.01,
                    'seed': 1024,
                },
                'metrics': {
                    'total_samples': 1296,
                    'original_recall': 0.8,
                    'attacked_recall': 0.6,
                    'recall_drop': 0.2,
                    'vod_official': {
                        'clean': {
                            'entire_area': {'3d': {'mAP': 45.4}},
                            'roi': {'3d': {'mAP': 67.2}},
                        },
                        'adversarial': {
                            'entire_area': {'3d': {'mAP': 30.0}},
                            'roi': {'3d': {'mAP': 44.0}},
                        },
                        'absolute_drop': {
                            'entire_area': {'3d': {'mAP': 15.4}},
                            'roi': {'3d': {'mAP': 23.2}},
                        },
                    },
                },
            }
            (output_dir / 'attack_results.json').write_text(
                json.dumps(result), encoding='utf-8'
            )
            experiment = Experiment(
                name='fgsm_xyz',
                parameters={},
                extra_tag='campaign/fgsm_xyz',
                output_dir=output_dir,
            )

            row = result_to_row(experiment)

            self.assertEqual(row['status'], 'completed')
            self.assertEqual(row['total_samples'], 1296)
            self.assertEqual(row['epsilon_default'], 0.05)
            self.assertEqual(row['epsilon_xyz'], 0.01)
            self.assertEqual(row['entire_3d_map_drop'], 15.4)
            self.assertEqual(row['roi_adversarial_3d_map'], 44.0)

            self.assertTrue(result_matches_experiment(experiment))
            changed_experiment = Experiment(
                name='fgsm_xyz',
                parameters={'epsilon_xyz': 0.02},
                extra_tag='campaign/fgsm_xyz',
                output_dir=output_dir,
            )
            self.assertFalse(result_matches_experiment(changed_experiment))
            self.assertEqual(
                result_to_row(changed_experiment)['status'],
                'stale',
            )

    def test_write_summaries_creates_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            campaign_dir = Path(temporary_dir) / 'campaign'
            experiment = Experiment(
                name='pending',
                parameters={'attack_type': 'fgsm'},
                extra_tag='campaign/pending',
                output_dir=campaign_dir / 'pending',
            )
            campaign = Campaign(
                name='campaign',
                source_path=Path('/tmp/campaign.yaml'),
                cfg_file='cfgs/model.yaml',
                output_dir=campaign_dir,
                experiments=[experiment],
            )

            rows = write_summaries(campaign)

            self.assertEqual(rows[0]['status'], 'pending')
            self.assertTrue((campaign_dir / 'summary.csv').is_file())
            self.assertTrue((campaign_dir / 'summary.md').is_file())


if __name__ == '__main__':
    unittest.main()
