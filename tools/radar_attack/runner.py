"""OpenPCDet experiment runner for 4D-radar adversarial attacks."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from radar_attack.adapters.openpcdet import (
    PointCloudVoxelizer,
    get_feature_names,
    get_voxel_settings,
)
from radar_attack.attacks.gradient import point_cloud_attack
from radar_attack.attacks.voxel import voxel_attack
from radar_attack.evaluation import (
    AdversarialPointCloudWriter,
    DetectionAttackMetrics,
)


def parse_config():
    parser = argparse.ArgumentParser(description='FGSM Attack on PointPillars for 4D Radar')
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for attack')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='fgsm_attack_radar', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint to load')
    parser.add_argument('--epsilon', type=float, default=0.05, help='FGSM epsilon (perturbation size)')
    parser.add_argument('--attack_domain', type=str, default='voxel',
                        choices=['voxel', 'point'],
                        help='attack voxel tensor or raw 4D-radar points')
    parser.add_argument('--attack_feature', type=str, default='all',
                        choices=['all', 'xyz', 'doppler', 'intensity', 'rcs', 'time'],
                        help='which features to perturb')
    parser.add_argument('--attack_type', type=str, default='fgsm', choices=['fgsm', 'pgd'],
                        help='attack type: fgsm or pgd')
    parser.add_argument('--pgd_steps', type=int, default=5, help='PGD steps')
    parser.add_argument('--step_size', type=float, default=None,
                        help='point PGD step size (default: 2 * epsilon / steps)')
    parser.add_argument('--epsilon_xyz', type=float, default=None,
                        help='point attack xyz budget, overriding epsilon')
    parser.add_argument('--epsilon_rcs', type=float, default=None,
                        help='point attack RCS budget, overriding epsilon')
    parser.add_argument('--epsilon_doppler', type=float, default=None,
                        help='point attack Doppler budget, overriding epsilon')
    parser.add_argument('--epsilon_time', type=float, default=None,
                        help='point attack timestamp budget, overriding epsilon')
    parser.add_argument('--random_start', action='store_true',
                        help='use a random PGD start for point attack')
    parser.add_argument('--voxel_mode', type=str, default='fixed',
                        choices=['fixed', 'revoxelize'],
                        help='point topology: fixed pillars or per-step revoxelization')
    parser.add_argument('--num_samples', type=int, default=None, help='number of samples to attack')
    parser.add_argument('--save_adv', action='store_true', default=False, help='save adversarial samples')
    parser.add_argument('--adv_dir', type=str, default=None,
                        help='directory for adversarial raw points (default: experiment output/adversarial_points)')
    parser.add_argument('--adv_format', choices=['npy', 'bin'], default='npy',
                        help='saved adversarial point-cloud format')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='confidence threshold used by sample-level attack success rate')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    if args.epsilon < 0:
        parser.error('--epsilon must be non-negative')
    if args.pgd_steps <= 0:
        parser.error('--pgd_steps must be positive')
    if args.step_size is not None and args.step_size <= 0:
        parser.error('--step_size must be positive')
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error('--num_samples must be positive')
    if not 0 <= args.score_threshold <= 1:
        parser.error('--score_threshold must be in [0, 1]')
    if args.save_adv and args.attack_domain != 'point':
        parser.error('--save_adv requires --attack_domain point')
    for name in ['epsilon_xyz', 'epsilon_rcs', 'epsilon_doppler', 'epsilon_time']:
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f'--{name} must be non-negative')

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def evaluate_attack(model, dataloader, args, logger, output_dir):
    model.eval()
    metrics = DetectionAttackMetrics(score_threshold=args.score_threshold)
    writer = None
    feature_names = get_feature_names(cfg.DATA_CONFIG)

    if args.attack_domain == 'point':
        voxel_size, max_points, max_voxels = get_voxel_settings(cfg.DATA_CONFIG)
        epsilon_overrides = {
            'xyz': args.epsilon_xyz,
            'rcs': args.epsilon_rcs,
            'doppler': args.epsilon_doppler,
            'time': args.epsilon_time,
        }
        if args.save_adv:
            adversarial_dir = (
                Path(args.adv_dir)
                if args.adv_dir is not None
                else output_dir / 'adversarial_points'
            )
            writer = AdversarialPointCloudWriter(
                output_dir=adversarial_dir,
                feature_names=feature_names,
                file_format=args.adv_format,
                run_metadata={
                    'attack_domain': args.attack_domain,
                    'attack_type': args.attack_type,
                    'attack_feature': args.attack_feature,
                    'epsilon': args.epsilon,
                    'epsilon_xyz': args.epsilon_xyz,
                    'epsilon_rcs': args.epsilon_rcs,
                    'epsilon_doppler': args.epsilon_doppler,
                    'epsilon_time': args.epsilon_time,
                    'pgd_steps': args.pgd_steps,
                    'step_size': args.step_size,
                    'random_start': args.random_start,
                    'voxel_mode': args.voxel_mode,
                },
            )
            logger.info('Adversarial raw points will be saved to %s', adversarial_dir)

    if args.num_samples is not None:
        total_samples_expected = min(args.num_samples, len(dataloader.dataset))
    else:
        total_samples_expected = len(dataloader.dataset)

    progress_bar = tqdm.tqdm(
        total=total_samples_expected,
        leave=True,
        desc='Attack Evaluation',
        dynamic_ncols=True,
    )

    for batch_dict in dataloader:
        if (
            args.num_samples is not None
            and metrics.total_samples >= args.num_samples
        ):
            break

        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            pred_dicts_original, ret_dict_original = model(dict(batch_dict))

        if args.attack_domain == 'point':
            original_points = batch_dict['points'].detach().clone()
            voxelizer = PointCloudVoxelizer(
                cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                voxel_size,
                max_points,
                max_voxels,
                int(batch_dict['batch_size']),
            )
            attack_output = point_cloud_attack(
                model=model,
                batch_dict=batch_dict,
                voxelizer=voxelizer,
                feature_names=feature_names,
                attack_type=args.attack_type,
                attack_feature=args.attack_feature,
                epsilon=args.epsilon,
                epsilon_overrides=epsilon_overrides,
                pgd_steps=args.pgd_steps,
                step_size=args.step_size,
                random_start=args.random_start,
                voxel_mode=args.voxel_mode,
            )
            batch_dict['points'] = attack_output.adv_points
            batch_dict.update(attack_output.model_inputs)
            metrics.update_perturbation(attack_output.stats)
            if writer is not None:
                writer.save_batch(
                    original_points,
                    attack_output.adv_points,
                    batch_dict,
                )
        else:
            batch_dict['voxels'] = voxel_attack(
                model=model,
                batch_dict=batch_dict,
                epsilon=args.epsilon,
                attack_type=args.attack_type,
                attack_feature=args.attack_feature,
                steps=args.pgd_steps,
                point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                feature_names=feature_names,
            )

        with torch.no_grad():
            pred_dicts_attacked, ret_dict_attacked = model(dict(batch_dict))

        metrics.update_predictions(
            pred_dicts_original,
            pred_dicts_attacked,
            ret_dict_original,
            ret_dict_attacked,
        )
        progress_bar.update(len(pred_dicts_original))

    progress_bar.close()
    results = metrics.compute()

    logger.info('=' * 70)
    logger.info('Attack Results for 4D Radar Data:')
    logger.info(f'Attack Domain: {args.attack_domain}')
    logger.info(f'Attack Type: {args.attack_type.upper()}')
    logger.info(f'Epsilon: {args.epsilon}')
    logger.info(f'Attack Feature: {args.attack_feature}')
    if args.attack_type == 'pgd':
        logger.info(f'PGD Steps: {args.pgd_steps}')
    if args.attack_domain == 'point':
        logger.info(f'Voxel Mode: {args.voxel_mode}')
        logger.info(f'Max |delta|: {results["max_abs_perturbation"]:.6f}')
        logger.info(f'Mean |delta|: {results["mean_abs_perturbation"]:.6f}')
        if writer is not None:
            logger.info(f'Saved adversarial point clouds: {writer.saved_samples}')
            logger.info(f'Adversarial manifest: {writer.manifest_path}')
    logger.info(f'Total Samples: {results["total_samples"]}')
    logger.info(f'Original Recall@0.5: {results["original_recall"]:.4f}')
    logger.info(f'Attacked Recall@0.5: {results["attacked_recall"]:.4f}')
    logger.info(f'Attack Success Rate (Sample): {results["attack_success_rate_sample"]:.4f}')
    logger.info(f'Attack Success Rate (Target): {results["attack_success_rate_target"]:.4f}')
    logger.info(f'Recall Drop: {results["recall_drop"]:.4f}')
    logger.info('=' * 70)
    return results


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args, cfg = parse_config()

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            18888, args.local_rank, backend='nccl'
        )
        dist_test = True

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_attack_%s_%s.txt' % (args.attack_type, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start 4D Radar Attack Logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger)
    model.cuda()

    logger.info('Starting attack evaluation on 4D radar data...')
    results = evaluate_attack(model, test_loader, args, logger, output_dir)

    with open(output_dir / 'attack_results.txt', 'w') as f:
        f.write('4D Radar Attack Results\n')
        f.write('=' * 40 + '\n')
        f.write(f'Attack Domain: {args.attack_domain}\n')
        f.write(f'Attack Type: {args.attack_type}\n')
        f.write(f'Epsilon: {args.epsilon}\n')
        f.write(f'Attack Feature: {args.attack_feature}\n')
        if args.attack_type == 'pgd':
            f.write(f'PGD Steps: {args.pgd_steps}\n')
        if args.attack_domain == 'point':
            f.write(f'Voxel Mode: {args.voxel_mode}\n')
        f.write(f'Total Samples: {results["total_samples"]}\n')
        f.write(f'Original Recall@0.5: {results["original_recall"]:.4f}\n')
        f.write(f'Attacked Recall@0.5: {results["attacked_recall"]:.4f}\n')
        f.write(f'Attack Success Rate (Sample): {results["attack_success_rate_sample"]:.4f}\n')
        f.write(f'Attack Success Rate (Target): {results["attack_success_rate_target"]:.4f}\n')
        f.write(f'Recall Drop: {results["recall_drop"]:.4f}\n')
        if args.attack_domain == 'point':
            f.write(f'Max |delta|: {results["max_abs_perturbation"]:.6f}\n')
            f.write(f'Mean |delta|: {results["mean_abs_perturbation"]:.6f}\n')

    logger.info('Attack evaluation finished. Results saved to %s' % output_dir)


if __name__ == '__main__':
    main()
