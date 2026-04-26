# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from PIL import Image

from attack_toolkit.src.utils.utils_attack import (
    setup_dirs, denormalize_img,
    get_asymmetry_anchors, get_target_boundary_pts, find_best_matching_boundary,
    sample_boundary_at_interval, generate_sampled_points,
    chamfer_distance, outward_inward_loss_interpolated,
    visualize_attack_results
)
from attack_toolkit.src.utils.utils_blind_attack import (
    calculate_combined_score, generate_lens_flare
)

np.set_printoptions(precision=3, suppress=True)


def single_gpu_attack_camera_blind(model,
                    data_loader,
                    cfg,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        cfg (dict): Configuration dictionary containing attack parameters.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.
        show_score_thr (float): Score threshold for visualization.
            Default: 0.3.

    Returns:
        list[dict]: The prediction results.
    """
    
    ''' init '''
    # init dirs 
    cams_dir = os.path.join(out_dir, 'cams')
    vis_seg_dir = os.path.join(out_dir, 'vis_seg')
    
    map_results_dir = os.path.join(out_dir, 'results', 'map')
    gt_dir = os.path.join(map_results_dir, 'gt')
    clean_dir = os.path.join(map_results_dir, 'clean')
    attack_dir = os.path.join(map_results_dir, 'attack')
    
    setup_dirs([out_dir,cams_dir, vis_seg_dir, map_results_dir, gt_dir, clean_dir, attack_dir])
    
    
    ''' settings '''
    device = torch.device('cuda:{}'.format(model.device_ids[0]))  # set the same device with model
    
    # vis
    pc_range = cfg.point_cloud_range
    car_img = Image.open('./figs/lidar_car.png')
    colors_plt = ['orange', 'b', 'g']  # divider->r, ped->b, boundary->g
    
    
    ''' loop '''
    results = []
    clean_results = []
    losses = []
    clean_losses = []
    best_attack_locs = {}
    
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        
        ''' debug '''
        sample_token = data['img_metas'][0].data[0][0]['sample_idx']
        # print(f'sample_token: {sample_token}')
        
        ''' data init '''
        # data: dict, keys: ['img', 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d']
        #  - img: list[torch.tensor], len: 1, shape: (1, 3, H, W)
        #  - img_metas: list[dict], len: 1
        #       - ['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 
        #           'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 
        #           'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_token', 'prev_idx', 
        #           'next_idx', 'pcd_scale_factor', 'pts_filename', 'scene_token', 'can_bus', 
        #           'lidar2global', 'camera2ego', 'camera_intrinsics', 'img_aug_matrix', 
        #           'lidar2ego', 'global2img']
        imgs = data['img'][0].data[0]  # shape (1, 6, 3, 480, 800), ori_shape (450, 800, 3)
        img_metas = data['img_metas'][0].data[0][0]
        
        ### image related ###
        img_h, img_w, img_c = img_metas['img_shape'][0]  # (h, w, c)
        img_shape = (3, img_h, img_w)  # (c, h, w)
        ori_h, ori_w, _ = img_metas['ori_shape'][0]  # (h, w, c)
        ori_img_shape = (3, ori_h, ori_w)  # (c, h, w)
        
        img_norm_cfg = img_metas['img_norm_cfg']
        ### image related ###
        
        ### map related ###
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0][0]  # lidarinstancelines object
        gt_labels_3d = data['gt_labels_3d'].data[0][0]  # tensor
            
        # load road boundary data from the pre-processed JSON file
        dataset_dir = os.path.basename(os.path.dirname(os.path.dirname(out_dir)))
        scene_data_path = f'{dataset_dir}/scenes_{cfg.attack.dataset}/{sample_token}.json'
        with open(scene_data_path, 'r') as f:
            scene_label = json.load(f)
            
        # extract left and right boundary coordinates from map elements
        left_boundary_pts = None
        right_boundary_pts = None
        for boundary in scene_label['map_elements']:
            if boundary['tag'] == 'left':
                left_boundary_pts = np.array(boundary['coordinates'])  # (50, 2)
            elif boundary['tag'] == 'right':
                right_boundary_pts = np.array(boundary['coordinates'])  # (50, 2)
        
        # determine which boundary is the diverge boundary (the one being attacked)
        # and which is the reference boundary based on the pre-computed tag
        diverge_boundary_tag, _, _, _ = scene_label['diverge_boundary_tag']
        
        # set up boundary points and their tensor representations based on diverge_boundary_tag
        if diverge_boundary_tag == 'left':
            diverge_boundary_pts = left_boundary_pts
            reference_boundary_pts = right_boundary_pts
        else:
            diverge_boundary_pts = right_boundary_pts
            reference_boundary_pts = left_boundary_pts
        diverge_boundary_pts_tensor = torch.tensor(diverge_boundary_pts).to(device)
        reference_boundary_pts_tensor = torch.tensor(reference_boundary_pts).to(device)
                
        # identify diverge points
        asymmetry_anchors = get_asymmetry_anchors(diverge_boundary_pts, reference_boundary_pts, CURVATURE_DIFF_THRESHOLD=0.1, top_k=5)
        asymmetry_anchors = np.hstack([asymmetry_anchors, np.ones((asymmetry_anchors.shape[0], 1)) * -1.84])
        ### map related ###
        
        ### attack related ###    
        # identify target boundary
        if cfg.attack.loss == 'rsa':
            target_boundary_pts = get_target_boundary_pts(diverge_boundary_pts, reference_boundary_pts, diverge_boundary_tag, cfg.attack.dataset, step=5)
            target_boundary_pts = torch.tensor(target_boundary_pts).to(device)
        elif cfg.attack.loss == 'eta':
            with open(f'{dataset_dir}/diverge_route_centerlines_{cfg.attack.dataset}/{sample_token}.json', 'r') as f:
                diverge_route_centerlines = json.load(f)
            target_boundary_pts = torch.tensor(diverge_route_centerlines).to(device)  # (20, 2)
        ### attack related ###
        
        
        ''' inference wo attack '''
        with torch.no_grad():
            # forward
            clean_result = model(return_loss=False, rescale=True, **data)
            
            # calculate clean loss
            pred_pts_3d = clean_result[0]['pts_bbox']['pts_3d'].to(device)  # (50, 20, 2)
            scores_3d = clean_result[0]['pts_bbox']['scores_3d']
            labels_3d = clean_result[0]['pts_bbox']['labels_3d']
            keep = (scores_3d > show_score_thr) & (labels_3d == 2)
            pred_pts_3d = pred_pts_3d[keep]
            
            diverge_boundary_pts_pred_clean = find_best_matching_boundary(
                pred_pts_3d,
                diverge_boundary_pts_tensor,
                device=device
            )
            
            if diverge_boundary_pts_pred_clean is None:
                clean_loss = torch.tensor(20, dtype=torch.float32, device=device)
            else:
                if diverge_boundary_pts_pred_clean[:, 1].max() > 0:
                    diverge_boundary_pts_pred_clean = diverge_boundary_pts_pred_clean[diverge_boundary_pts_pred_clean[:, 1] > 0]
                
                if cfg.attack.loss == 'rsa':
                    clean_loss = chamfer_distance(diverge_boundary_pts_pred_clean, target_boundary_pts, device=device)
                elif cfg.attack.loss == 'eta':
                    clean_loss = outward_inward_loss_interpolated(diverge_boundary_pts_pred_clean, diverge_boundary_pts_tensor, target_boundary_pts, reference_boundary_pts_tensor, visualize=False)
                else:
                    raise ValueError(f'Unknown loss type: {cfg.attack.loss}')
            
            clean_losses.append(clean_loss.detach().item())
            # print(f'\nClean loss: {clean_loss.item():.4f}')
        
        
        ''' inference w attack '''
        ### attack settings ###
        total_locs = cfg.attack.blind.total_locs
        sample_interval = cfg.attack.blind.sample_interval
        locs_height_num = cfg.attack.blind.locs_height_num
        
        samples_per_loc = cfg.attack.blind.samples_per_loc -1
        sample_range = cfg.attack.blind.sample_range
        ### attack settings ###
        
        ### attack location generation ###
        # sample attack locations densely along the boundary
        dense_attack_locs = sample_boundary_at_interval(diverge_boundary_pts, interval=sample_interval)  # 1m interval
        
        # sampled points around if configured
        if cfg.attack.blind.sample:
            all_attack_locs = dense_attack_locs.copy()
            for attack_loc in dense_attack_locs:
                sampled_points = generate_sampled_points(attack_loc, sample_range, samples_per_loc, mode='random')
                all_attack_locs = np.vstack([all_attack_locs, sampled_points])

        # attack position ranking
        attack_loc_candidates = []
        for attack_loc in all_attack_locs:
            # sample heights evenly from [0, 1.84]
            heights = np.linspace(-1.84, 0, locs_height_num)
            for height in heights:
                attack_loc_3d = np.append(attack_loc, height)
                
                # Calculate physical feasibility score (still valuable)
                combined_score = calculate_combined_score(attack_loc_3d, img_metas, asymmetry_anchors, max_beam_angle=np.radians(40))
                
                # Store point with its feasibility score
                attack_loc_candidates.append((attack_loc_3d, combined_score))
        
        # Sort by feasibility and take top candidates
        attack_loc_candidates.sort(key=lambda x: x[1], reverse=True)
        top_attack_locs = np.array([p for p, _ in attack_loc_candidates[:total_locs]])
            
        # ### debug ###
        # # Create figure and axis
        # plt.figure(figsize=(8, 16))
        # # Plot diverge boundary
        # plt.plot(diverge_boundary_pts[:, 0], diverge_boundary_pts[:, 1], 
        #         'b-', label='Diverge Boundary', linewidth=2)
        # # Plot reference boundary 
        # plt.plot(reference_boundary_pts[:, 0], reference_boundary_pts[:, 1],
        #         'g-', label='Reference Boundary', linewidth=2)
        # # Plot position candidates
        # plt.scatter(asymmetry_anchors_sampled[:, 0], asymmetry_anchors_sampled[:, 1],
        #             color='r', label='Position Candidates', marker='x', s=10)
        # # plot all asymmetry_anchors_sampled
        # plt.scatter(asymmetry_anchors[:, 0], asymmetry_anchors[:, 1],
        #             color='g', label='Diverge Points', marker='o', s=100)
        # # Plot target boundary if using targeted attack
        # if cfg.attack.loss == 'targeted':
        #     target_boundary_pts_np = target_boundary_pts.cpu().numpy()
        #     plt.plot(target_boundary_pts_np[:, 0], target_boundary_pts_np[:, 1],
        #             'r--', label='Target Boundary', linewidth=2)
        # plt.xlim(-15, 15)
        # plt.ylim(-30, 30)
        # plt.xlabel('X (m)')
        # plt.ylabel('Y (m)') 
        # plt.title('Road Boundaries')
        # plt.legend()
        # plt.savefig(f'test.png')
        # plt.close()
        # raise ValueError('stop here')
        # ### debug: plot target boundaries ###
        ### attack location generation ###
        
        
        ''' Attack '''
        if cfg.attack.loss in ['rsa', 'eta']:
            best_loss = 1e10
        else:
            raise ValueError(f'Unknown loss type: {cfg.attack.loss}')
        best_attack_loc = None
        
        print(f'\nSearching {len(top_attack_locs)} attack locations for sample {sample_token}...\n')
        
        for loc_idx, attack_loc in enumerate(top_attack_locs):
            
            # apply patch
            imgs_adv = imgs.clone().detach().to(device)  # (1, 6, 3, 480, 800)
            
            light_source_params = {
                'position': attack_loc,  # 3D position in LiDAR coordinates
                'power': 3000.0,  # lumens for flashlight (typical range: 100-2000)
                'beam_angle': np.radians(40),  # typical flashlight beam angle: 20-40 degrees
            }
            for cam_idx in range(6):
                imgs_adv = generate_lens_flare(imgs_adv, img_metas, light_source_params, img_norm_cfg, cam_idx, (ori_h, ori_w))
            
            # forward
            data['img'][0].data[0] = imgs_adv.detach().cpu()
            result = model(return_loss=False, rescale=True, **data)
            
            # calculate the loss
            pred_pts_3d = result[0]['pts_bbox']['pts_3d'].to(device)  # (50, 20, 2)
            scores_3d = result[0]['pts_bbox']['scores_3d']
            labels_3d = result[0]['pts_bbox']['labels_3d']
            keep = (scores_3d > show_score_thr) & (labels_3d == 2)
            pred_pts_3d = pred_pts_3d[keep]
            
            diverge_boundary_pts_pred_atk = find_best_matching_boundary(
                pred_pts_3d,
                diverge_boundary_pts_tensor,
                device=device
            )
            
            # handle the case that the predicted boundary is None
            if diverge_boundary_pts_pred_atk is None:
                if cfg.attack.loss == 'rsa':
                    loss = torch.tensor(20, dtype=torch.float32, device=device)
                elif cfg.attack.loss == 'eta':
                    loss = torch.tensor(-20, dtype=torch.float32, device=device)
                    best_loss = loss
                    best_attack_loc = attack_loc
                
                # Dynamic progress update in one line
                progress_msg = f'\rLocation {loc_idx+1}/{len(top_attack_locs)}: Loss={loss.item():.4f} (No boundary), Best={best_loss.item():.4f}, Clean={clean_loss.item():.4f}'
                print(progress_msg, end='', flush=True)
                break  # no need to continue
            
            # calc loss
            if diverge_boundary_pts_pred_atk[:, 1].max() > 0:
                diverge_boundary_pts_pred_atk = diverge_boundary_pts_pred_atk[diverge_boundary_pts_pred_atk[:, 1] > 0]  
            
            if cfg.attack.loss == 'rsa':
                loss = chamfer_distance(diverge_boundary_pts_pred_atk, target_boundary_pts, device=device)
            elif cfg.attack.loss == 'eta':
                loss = outward_inward_loss_interpolated(diverge_boundary_pts_pred_atk, diverge_boundary_pts_tensor, target_boundary_pts, reference_boundary_pts_tensor, visualize=False)
            else:
                raise ValueError(f'Unknown loss type: {cfg.attack.loss}')
            
            if loss < best_loss:
                best_loss = loss
                best_attack_loc = attack_loc
          
            # Dynamic progress update in one line
            progress_msg = f'\rLocation {loc_idx+1}/{len(top_attack_locs)}: Loss={loss.item():.4f}, Best={best_loss.item():.4f}, Clean={clean_loss.item():.4f}'
            print(progress_msg, end='', flush=True)

        # Print final summary for this sample
        print(f'\n\nSample {sample_token} completed - Best loss: {best_loss.item():.4f}')
        
        losses.append(best_loss.detach().item())
        
        
        ''' Evaluation '''
        with torch.no_grad():
            
            # apply patch
            imgs_adv = imgs.clone().detach().to(device)  # (1, 6, 3, 480, 800)
            
            light_source_params = {
                'position': best_attack_loc,  # 3D position in LiDAR coordinates (numpy array with shape (3,))
                'power': 3000.0,  # lumens for flashlight (typical range: 100-2000)
                'beam_angle': np.radians(40),  # typical flashlight beam angle: 20-40 degrees
            }
            for cam_idx in range(6):
                imgs_adv = generate_lens_flare(imgs_adv, img_metas, light_source_params, img_norm_cfg, cam_idx, (ori_h, ori_w))
            
            # forward
            data['img'][0].data[0] = imgs_adv.detach().cpu()
            result = model(return_loss=False, rescale=True, **data)
            
            # save combined adv image
            combined_adv_imgs = []
            for img_idx in range(6):
                img_adv_save = imgs_adv[0, img_idx].clone()
                img_adv_save = denormalize_img(img_adv_save, img_norm_cfg).permute(1, 2, 0)
                img_adv_save = img_adv_save[:ori_h, :ori_w, :]
                img_adv_save = cv2.cvtColor(img_adv_save.detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
                combined_adv_imgs.append(img_adv_save)
            row1 = np.hstack([combined_adv_imgs[2], combined_adv_imgs[0], combined_adv_imgs[1]])
            row2 = np.hstack([combined_adv_imgs[4], combined_adv_imgs[3], combined_adv_imgs[5]])
            combined_adv_imgs = np.vstack([row1, row2])
            adv_imgs_path = os.path.join(cams_dir, f'{sample_token}.png')
            cv2.imwrite(adv_imgs_path, combined_adv_imgs)
            print(f'Save adv images to: {adv_imgs_path}')


        ''' Visualization '''
        # visualize attack results
        gt_data = (gt_bboxes_3d, gt_labels_3d)
        clean_data = clean_result[0]
        attack_data = result[0]

        visualize_attack_results(gt_data, clean_data, attack_data, \
            vis_seg_dir, f'{sample_token}', pc_range, car_img, colors_plt, segment=target_boundary_pts.detach().cpu().numpy(), show_score_thr=show_score_thr)


        ''' save '''
        # save attack result
        best_attack_locs[sample_token] = best_attack_loc.tolist()
        
        # save map result
        results.extend(result)
        clean_results.extend(clean_result)
        
        # save ground truth map for planning
        gt_bboxes = [bbox.tolist() for bbox in gt_bboxes_3d.fixed_num_sampled_points]
        gt_labels = [label.item() for label in gt_labels_3d]
        gt_data = {
            'bboxes': gt_bboxes,
            'labels': gt_labels
        }
        with open(os.path.join(gt_dir, f'{sample_token}.json'), 'w') as f:
            json.dump(gt_data, f)
        
        ''' update progress bar '''
        batch_size = len(clean_result)
        for _ in range(batch_size):
            prog_bar.update()
            
        # break
    
    # save best diverge points
    with open(os.path.join(attack_dir, 'best_attack_locs.json'), 'w') as f:
        json.dump(best_attack_locs, f)
        
    # # print average loss
    # print(f'Average Clean Loss: {np.mean(clean_losses)}')
    # print(f'Average Attack Loss: {np.mean(losses)}')
    # with open(os.path.join(map_results_dir, 'losses.txt'), 'w') as f:
    #     f.write(f'Average Clean Loss: {np.mean(clean_losses)}\n')
    #     f.write(f'Average Attack Loss: {np.mean(losses)}\n')

    return results, clean_results
