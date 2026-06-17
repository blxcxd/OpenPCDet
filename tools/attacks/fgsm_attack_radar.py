import _init_path
import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='FGSM Attack on PointPillars for 4D Radar')
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for attack')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='fgsm_attack_radar', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint to load')
    parser.add_argument('--epsilon', type=float, default=0.05, help='FGSM epsilon (perturbation size)')
    parser.add_argument('--attack_feature', type=str, default='all', 
                        choices=['all', 'xyz', 'doppler', 'intensity'],
                        help='which features to perturb')
    parser.add_argument('--attack_type', type=str, default='fgsm', choices=['fgsm', 'pgd'], 
                        help='attack type: fgsm or pgd')
    parser.add_argument('--pgd_steps', type=int, default=5, help='PGD steps')
    parser.add_argument('--num_samples', type=int, default=None, help='number of samples to attack')
    parser.add_argument('--save_adv', action='store_true', default=False, help='save adversarial samples')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def apply_perturbation_mask(points, perturbation, attack_feature):
    mask = torch.ones_like(points)
    
    if attack_feature == 'xyz':
        mask[:, 3:] = 0.0
    elif attack_feature == 'doppler':
        mask[:, :3] = 0.0
        mask[:, 4:] = 0.0
    elif attack_feature == 'intensity':
        mask[:, :4] = 0.0
    
    return perturbation * mask


def fgsm_attack_voxel(model, batch_dict, epsilon, attack_feature='all'):
    model.train()
    
    voxels = batch_dict['voxels'].clone().detach().requires_grad_(True)
    batch_dict['voxels'] = voxels
    
    model.zero_grad()
    
    ret_dict, _, _ = model(batch_dict)
    loss = ret_dict['loss']
    
    loss.backward()
    
    grad = voxels.grad.data
    
    perturbation = epsilon * grad.sign()
    perturbation = apply_perturbation_mask(voxels, perturbation, attack_feature)
    
    perturbed_voxels = voxels + perturbation
    
    point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    min_vals = torch.tensor([point_cloud_range[0], point_cloud_range[1], point_cloud_range[2], -10.0, 0.0], 
                            device=perturbed_voxels.device)
    max_vals = torch.tensor([point_cloud_range[3], point_cloud_range[4], point_cloud_range[5], 10.0, 1.0], 
                            device=perturbed_voxels.device)
    
    for i in range(min(perturbed_voxels.shape[-1], 5)):
        perturbed_voxels[..., i] = torch.clamp(perturbed_voxels[..., i], min_vals[i], max_vals[i])
    
    return perturbed_voxels.detach()


def pgd_attack_voxel(model, batch_dict, epsilon, attack_feature='all', steps=5):
    model.train()
    
    original_voxels = batch_dict['voxels'].clone().detach()
    perturbed_voxels = original_voxels.clone().requires_grad_(True)
    
    alpha = epsilon / steps
    
    for _ in range(steps):
        batch_dict['voxels'] = perturbed_voxels
        
        model.zero_grad()
        
        ret_dict, _, _ = model(batch_dict)
        loss = ret_dict['loss']
        
        loss.backward()
        
        grad = perturbed_voxels.grad.data
        
        perturbation = alpha * grad.sign()
        perturbation = apply_perturbation_mask(perturbed_voxels, perturbation, attack_feature)
        
        perturbed_voxels = perturbed_voxels + perturbation
        
        point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        min_vals = torch.tensor([point_cloud_range[0], point_cloud_range[1], point_cloud_range[2], -10.0, 0.0], 
                                device=perturbed_voxels.device)
        max_vals = torch.tensor([point_cloud_range[3], point_cloud_range[4], point_cloud_range[5], 10.0, 1.0], 
                                device=perturbed_voxels.device)
        
        for i in range(min(perturbed_voxels.shape[-1], 5)):
            perturbed_voxels[..., i] = torch.clamp(perturbed_voxels[..., i], min_vals[i], max_vals[i])
        
        perturbation = torch.clamp(perturbed_voxels - original_voxels, -epsilon, epsilon)
        perturbed_voxels = original_voxels + perturbation
        
        perturbed_voxels = perturbed_voxels.detach().requires_grad_(True)
    
    return perturbed_voxels.detach()


def evaluate_attack(model, dataloader, args, logger):
    model.eval()
    
    total_samples = 0
    attack_success = 0
    original_recall = 0
    attacked_recall = 0
    gt_count = 0
    
    if args.num_samples is not None:
        total_iters = min(args.num_samples, len(dataloader))
    else:
        total_iters = len(dataloader)
    
    progress_bar = tqdm.tqdm(total=total_iters, leave=True, desc='Attack Evaluation', dynamic_ncols=True)
    
    for i, batch_dict in enumerate(dataloader):
        if args.num_samples is not None and i >= args.num_samples:
            break
            
        load_data_to_gpu(batch_dict)
        
        with torch.no_grad():
            pred_dicts_original, ret_dict_original = model(batch_dict)
        
        if args.attack_type == 'fgsm':
            perturbed_voxels = fgsm_attack_voxel(model, batch_dict, args.epsilon, args.attack_feature)
        else:
            perturbed_voxels = pgd_attack_voxel(model, batch_dict, args.epsilon, args.attack_feature, args.pgd_steps)
        
        batch_dict['voxels'] = perturbed_voxels
        
        with torch.no_grad():
            pred_dicts_attacked, ret_dict_attacked = model(batch_dict)
        
        original_recall += ret_dict_original.get('rcnn_0.5', 0)
        attacked_recall += ret_dict_attacked.get('rcnn_0.5', 0)
        gt_count += ret_dict_original.get('gt', 0)
        
        original_boxes = pred_dicts_original[0]['pred_boxes']
        attacked_boxes = pred_dicts_attacked[0]['pred_boxes']
        
        original_scores = pred_dicts_original[0]['pred_scores'] if len(pred_dicts_original[0]['pred_scores']) > 0 else torch.tensor([0.0])
        attacked_scores = pred_dicts_attacked[0]['pred_scores'] if len(pred_dicts_attacked[0]['pred_scores']) > 0 else torch.tensor([0.0])
        
        orig_detected = len(original_boxes) > 0 and original_scores.max() >= 0.5
        attack_success_cond = len(attacked_boxes) == 0 or attacked_scores.max() < 0.5
        
        if orig_detected and attack_success_cond:
            attack_success += 1
        
        total_samples += 1
        progress_bar.update()
    
    progress_bar.close()
    
    original_recall_rate = original_recall / max(gt_count, 1)
    attacked_recall_rate = attacked_recall / max(gt_count, 1)
    attack_success_rate = attack_success / max(total_samples, 1)
    
    logger.info('=' * 70)
    logger.info('Attack Results for 4D Radar Data:')
    logger.info(f'Attack Type: {args.attack_type.upper()}')
    logger.info(f'Epsilon: {args.epsilon}')
    logger.info(f'Attack Feature: {args.attack_feature}')
    if args.attack_type == 'pgd':
        logger.info(f'PGD Steps: {args.pgd_steps}')
    logger.info(f'Total Samples: {total_samples}')
    logger.info(f'Original Recall@0.5: {original_recall_rate:.4f}')
    logger.info(f'Attacked Recall@0.5: {attacked_recall_rate:.4f}')
    logger.info(f'Attack Success Rate: {attack_success_rate:.4f}')
    logger.info(f'Recall Drop: {(original_recall_rate - attacked_recall_rate):.4f}')
    logger.info('=' * 70)
    
    return {
        'original_recall': original_recall_rate,
        'attacked_recall': attacked_recall_rate,
        'attack_success_rate': attack_success_rate,
        'recall_drop': original_recall_rate - attacked_recall_rate
    }


def main():
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
    results = evaluate_attack(model, test_loader, args, logger)

    with open(output_dir / 'attack_results.txt', 'w') as f:
        f.write('4D Radar Attack Results\n')
        f.write('=' * 40 + '\n')
        f.write(f'Attack Type: {args.attack_type}\n')
        f.write(f'Epsilon: {args.epsilon}\n')
        f.write(f'Attack Feature: {args.attack_feature}\n')
        if args.attack_type == 'pgd':
            f.write(f'PGD Steps: {args.pgd_steps}\n')
        f.write(f'Total Samples: {results["total_samples"] if "total_samples" in results else total_samples}\n')
        f.write(f'Original Recall@0.5: {results["original_recall"]:.4f}\n')
        f.write(f'Attacked Recall@0.5: {results["attacked_recall"]:.4f}\n')
        f.write(f'Attack Success Rate: {results["attack_success_rate"]:.4f}\n')
        f.write(f'Recall Drop: {results["recall_drop"]:.4f}\n')

    logger.info('Attack evaluation finished. Results saved to %s' % output_dir)


if __name__ == '__main__':
    main()