# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import pickle
import sys

import mmcv
import torch
import cv2
import numpy as np
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from shapely.geometry import LineString

from attack_toolkit.src.utils.utils_attack import (
    setup_dirs, denormalize_img,
    get_asymmetry_anchors, get_target_boundary_pts, find_best_matching_boundary,
    sample_boundary_at_interval, generate_sampled_points,
    chamfer_distance, outward_inward_loss_interpolated,
    visualize_attack_results
)
from attack_toolkit.src.utils.utils_patch_attack import (
    get_patch_heading_facing_ego, get_proj_scale, create_pseudo_area,
    init_patch_mask, get_phy_patch_mask, apply_patch
)
from attack_toolkit.src.utils.utils_blind_attack import calculate_combined_score


def single_gpu_attack_patch(model,
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
    vis_seg_dir = os.path.join(out_dir, 'vis_seg')    
    cams_dir = os.path.join(out_dir, 'cams')
    
    map_results_dir = os.path.join(out_dir, 'results', 'map')
    clean_dir = os.path.join(map_results_dir, 'clean')
    attack_dir = os.path.join(map_results_dir, 'attack')
    gt_dir = os.path.join(map_results_dir, 'gt')
    
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
    clean_losses = []
    losses = []
    best_patches = {}
    
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
        #           'lidar2ego']
        imgs = data['img'][0].data[0]  # shape (1, 6, 3, 480, 800), ori_shape (450, 800, 3)
        data['img'][0].data[0] = data['img'][0].data[0].to(device)  # to cuda
        img_metas = data['img_metas'][0].data[0][0]
        
        ### image related ###
        img_h, img_w, img_c = img_metas['img_shape'][0]  # (h, w, c)
        img_shape = (3, img_h, img_w)  # (c, h, w)
        ori_h, ori_w, _ = img_metas['ori_shape'][0]  # (h, w, c)
        ori_img_shape = (3, ori_h, ori_w)  # (c, h, w)
        
        img_norm_cfg = img_metas['img_norm_cfg']
        lidar2img = torch.tensor(img_metas['lidar2img']).to(device)
        lidar2global = img_metas['lidar2global']
        global2img = img_metas['global2img']
        
        lidar2global = torch.tensor(lidar2global, dtype=imgs.dtype).to(device)
        global2img = torch.tensor(global2img, dtype=imgs.dtype).to(device)
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
        # convert boundary points to tensors and move to the correct device
        diverge_boundary_pts_tensor = torch.tensor(diverge_boundary_pts).to(device)
        reference_boundary_pts_tensor = torch.tensor(reference_boundary_pts).to(device)
        
        # identify asymmetry anchors
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
        patch_num = 1
        sample_range = cfg.attack.patch.sample_range  # e.g., 2.0 meters
        sampled_locs = cfg.attack.patch.samples_per_loc - 1

        total_locs = cfg.attack.patch.total_locs // cfg.attack.patch.step_per_loc
        sample_interval = cfg.attack.patch.sample_interval
        
        max_step = cfg.attack.patch.step_per_loc
        lr = 255 / max_step
        lr_decay = 0.9
        
        
        ### attack location generation ###
        # 1. sample attack locations densely along the boundary      
        dense_attack_locs = sample_boundary_at_interval(diverge_boundary_pts, interval=sample_interval)  # 1m interval
        
        # 2. filter valid locations along the boundary
        patch_center_valid = []
        patch_heading_valid = []
        for attack_loc in dense_attack_locs:

            patch_center = attack_loc.copy()
            is_left_boundary = diverge_boundary_tag == 'left'
            patch_center[0] = attack_loc[0] - cfg.attack.patch.width/2 if is_left_boundary else attack_loc[0] + cfg.attack.patch.width/2
            
            patch_cfg = {
                'type': 'vertical',  # 'vertical', 'ground'
                'lat': patch_center[0],
                'long': patch_center[1],  # to the vehicle front
                'width': cfg.attack.patch.width,
                'height': cfg.attack.patch.height,
                'heading': get_patch_heading_facing_ego(patch_center),  # lidar x-axis
                'lidar2vehfront': 0.94,  # 0.9m
                'lidar2ground': 1.84  # 1.84m
            }
            
            proj_scale = get_proj_scale(patch_cfg['lat'], patch_cfg['long'], ori_w)
            pseudo_area = create_pseudo_area(patch_cfg, ori_img_shape, proj_scale)
            
            # skip if pseudo_area is invalid
            if pseudo_area is None:
                # print(f"Skipping attack location {attack_loc} - patch would appear too large in camera view")
                continue
            
            patch_center_valid.append(patch_center)
            patch_heading_valid.append(patch_cfg['heading'])
        
        # 3. sample additional locations if configured
        patch_center_candidates = patch_center_valid.copy()
        patch_heading_candidates = patch_heading_valid.copy()

        if cfg.attack.patch.sample:    
            for patch_center, base_heading in zip(patch_center_valid, patch_heading_valid):
                # sample around the attack location
                sampled_points = generate_sampled_points(patch_center, sample_range, sampled_locs, mode=diverge_boundary_tag)
                
                # sample heading angles within ±30 degrees (±0.52 radians)
                offset_range = np.radians(30)  # 30 degrees
                heading_offsets = np.random.rand(len(sampled_points)) * offset_range * 2 - offset_range
                sampled_headings = base_heading + heading_offsets
                
                # add sampled points and headings to candidates
                for point, heading in zip(sampled_points, sampled_headings):
                    patch_center_candidates.append(point)
                    patch_heading_candidates.append(heading)

        # 4. attack location ranking
        attack_loc_candidates = []
        for patch_center, patch_heading in zip(patch_center_candidates, patch_heading_candidates):
            # create 3D point with height component
            patch_center_3d = np.append(patch_center, -1.84 + cfg.attack.patch.height/2)

            # calculate score based on visibility and other factors
            combined_score = calculate_combined_score(
                patch_center_3d, 
                img_metas, 
                asymmetry_anchors, 
                max_beam_angle=np.radians(20)
            )
            attack_loc_candidates.append((patch_center_3d, patch_heading, combined_score))

        # 5. sort and select the top k candidates
        attack_loc_candidates.sort(key=lambda x: x[2], reverse=True)
        top_attack_locs = attack_loc_candidates[:total_locs]

        # extract points and headings from the top k candidates
        top_patch_centers = np.array([p for p, _, _ in top_attack_locs])  # (n, 3)
        top_patch_headings = np.array([h for _, h, _ in top_attack_locs])  # (n,)
        ### attack location generation ###

        # ### debug ###
        # plt.figure(figsize=(5, 10))
        # # Plot diverge boundary
        # plt.plot(diverge_boundary_pts[:, 0], diverge_boundary_pts[:, 1], 
        #         'b-', label='Diverge Boundary', linewidth=2)
        # # Plot reference boundary 
        # plt.plot(reference_boundary_pts[:, 0], reference_boundary_pts[:, 1],
        #         'g-', label='Reference Boundary', linewidth=2)
        # plt.scatter(diverge_points[:, 0], diverge_points[:, 1],
        #             color='g', label='Diverge Points', marker='o', s=10)
        # # plot all diverge_points_sampled
        # top_patch_centers = np.array(top_patch_centers)
        # plt.scatter(top_patch_centers[:, 0], top_patch_centers[:, 1],
        #             color='r', label='Diverge Points', marker='x', s=10)
        # # Plot target boundary if using rsa attack
        # if cfg.attack.loss == 'rsa':
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
        # raise ValueError('stop')
        # ### debug ###
            
            
        ''' Attack '''
        if cfg.attack.loss in ['rsa', 'eta']:
            best_loss = 1e10
        else:
            raise ValueError(f'Unknown loss type: {cfg.attack.loss}')
        
        best_patch = None
        best_mask = None
        best_patch_cfg = None
        best_pseudo_area = None
        
        print(f'\nOptimizing patches at {len(top_patch_centers)} locations over {max_step} steps for sample {sample_token}...\n')
        
        for loc_idx, (patch_center, patch_heading) in enumerate(zip(top_patch_centers, top_patch_headings)):
            
            # init patches and masks
            patches = []
            masks = []
            pseudo_areas = []
            patch_cfgs = []
            for _ in range(patch_num):
                
                patch_cfg = {
                    'type': 'vertical',  # 'vertical', 'ground'
                    'lat': patch_center[0],
                    'long': patch_center[1],  # to the vehicle front
                    'width': cfg.attack.patch.width,
                    'height': cfg.attack.patch.height,
                    'heading': patch_heading,  # lidar x-axis
                    'lidar2vehfront': 0.94,  # 0.9m
                    'lidar2ground': 1.84  # 1.84m
                }
                
                proj_scale = get_proj_scale(patch_cfg['lat'], patch_cfg['long'], ori_w)
                pseudo_area = create_pseudo_area(patch_cfg, ori_img_shape, proj_scale)
                
                # skip if pseudo_area is invalid
                if pseudo_area is None:
                    # print(f"Skipping attack location {patch_center} - patch would appear too large in camera view")
                    continue
                
                # init patch and mask
                patch, mask = init_patch_mask(ori_img_shape, img_norm_cfg, device, pseudo_area, mode='random')
                
                patch_cfgs.append(patch_cfg)
                patches.append(patch)
                masks.append(mask)
                pseudo_areas.append(pseudo_area)
            
            # skip if no valid patches
            if len(patches) == 0:
                continue

            # init optimizer and scheduler for attack
            optimizer = optim.Adam(patches, lr, betas=(0.5, 0.9))
            scheduler = StepLR(optimizer, 10, lr_decay)
        
        
            ''' Attack '''
            # track loss for early stopping
            loss_history = []
            stagnation_counter = 0
            stagnation_threshold = 5  # number of iterations with minimal change to trigger early stopping
            min_loss_change = 0.0001   # minimum change in loss to be considered progress
            
            # Track best loss for this location
            location_best_loss = 1e10
            
            for step in range(max_step):
                
                # prepare for attack
                optimizer.zero_grad()
                
                for patch in patches:
                    patch.data.clamp_(0, 255)
                
                imgs_adv = imgs.clone().detach().to(device)  # (1, 6, 3, 480, 800)
                
                # apply patches
                for patch_idx, patch_cfg in enumerate(patch_cfgs):
                    # get phy patch and mask
                    patch_trans_list, mask_trans_list, visible_cam_indices = get_phy_patch_mask(
                        patches[patch_idx], 
                        masks[patch_idx], 
                        patch_cfg, 
                        pseudo_areas[patch_idx], 
                        lidar2global, 
                        global2img, 
                        (ori_h, ori_w), 
                        ground=True
                    )
                    
                    # if visible_cam_idx is None, the patch is not visible on any camera
                    if visible_cam_indices is None:
                        continue
                    
                    # apply patch to all visible cameras
                    imgs_adv = apply_patch(
                        imgs_adv, 
                        patch_trans_list, 
                        mask_trans_list, 
                        img_norm_cfg, 
                        visible_cam_indices
                    )
                
                # ### debug ###
                # combined_adv_imgs = []
                # for img_idx in range(6):
                #     img_adv_save = imgs_adv[0, img_idx].clone()
                #     img_adv_save = denormalize_img(img_adv_save, img_norm_cfg).permute(1, 2, 0)
                #     img_adv_save = img_adv_save[:ori_h, :ori_w, :]
                #     img_adv_save = cv2.cvtColor(img_adv_save.detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
                #     combined_adv_imgs.append(img_adv_save)
                    
                # # Combine images
                # row1 = np.hstack([combined_adv_imgs[2], combined_adv_imgs[0], combined_adv_imgs[1]])
                # row2 = np.hstack([combined_adv_imgs[4], combined_adv_imgs[3], combined_adv_imgs[5]])
                # combined_adv_imgs = np.vstack([row1, row2])

                # # Save combined image
                # cv2.imwrite(os.path.join(cams_dir, f'{sample_token}.png'), combined_adv_imgs)
                # print(f'save adv image', os.path.join(cams_dir, f'{sample_token}.png'))
                # ### debug ###


                data['img'][0].data[0].detach_()
                data['img'][0].data[0] = imgs_adv
                
                # forward
                result = model(return_loss=False, rescale=True, **data)
                
                # calculate loss
                pred_pts_3d = result[0]['pts_bbox']['pts_3d'].to(device)  # (50, 20, 2)
                scores_3d = result[0]['pts_bbox']['scores_3d']
                labels_3d = result[0]['pts_bbox']['labels_3d']
                
                keep = (scores_3d > show_score_thr) & (labels_3d == 2)
                pred_pts_3d = pred_pts_3d[keep]
                scores_3d = scores_3d[keep]
                    
                # find the predicted boundary points that best match the ground truth
                diverge_boundary_pts_pred = find_best_matching_boundary(
                    pred_pts_3d,
                    diverge_boundary_pts_tensor,
                    device=device
                )

                # -------------------------------------------------------------------------
                # CASE 1: No boundary detected - Complete detection failure
                # -------------------------------------------------------------------------
                if diverge_boundary_pts_pred is None:
                    if cfg.attack.loss == 'rsa':  # for rsa attacks, detection failure is penalized
                        loss = torch.tensor(20.0, dtype=torch.float32, device=device)
                    elif cfg.attack.loss == 'eta':  # for eta attacks, detection failure is rewarded
                        loss = torch.tensor(-20.0, dtype=torch.float32, device=device)
                        # save this as the best result
                        best_loss = loss
                        best_patch = patches.copy()
                        best_mask = masks.copy()
                        best_patch_cfg = patch_cfgs.copy()
                        best_pseudo_area = pseudo_areas.copy()
                        location_best_loss = loss
                    else:
                        raise ValueError(f'Unknown loss type: {cfg.attack.loss}')
                    
                    progress_msg = f'\rLoc {loc_idx+1}/{len(top_patch_centers)}, Step {step+1}/{max_step}: Loss={loss.item():.4f} (No boundary), Best(global)={best_loss.item():.4f}, Clean={clean_loss.item():.4f} [Complete]'
                    print(progress_msg)
                    break

                # -------------------------------------------------------------------------
                # CASE 2: Boundary too short - Partial detection failure
                # -------------------------------------------------------------------------
                if diverge_boundary_pts_pred[:, 1].max() > 0:
                    diverge_boundary_pts_pred = diverge_boundary_pts_pred[diverge_boundary_pts_pred[:, 1] > 0]  
                
                is_boundary_too_short = (
                    diverge_boundary_pts_pred.shape[0] < 2 or 
                    LineString(diverge_boundary_pts_pred).length < 5
                )

                if cfg.attack.loss == 'eta' and is_boundary_too_short:  # for eta attacks, a too-short boundary is also a success
                    loss = torch.tensor(-20.0, dtype=torch.float32, device=device)
                    # save this as the best result
                    best_loss = loss
                    best_patch = patches.copy()
                    best_mask = masks.copy()
                    best_patch_cfg = patch_cfgs.copy()
                    best_pseudo_area = pseudo_areas.copy()
                    location_best_loss = loss
                    
                    progress_msg = f'\rLoc {loc_idx+1}/{len(top_patch_centers)}, Step {step+1}/{max_step}: Loss={loss.item():.4f} (Short boundary), Best(global)={best_loss.item():.4f}, Clean={clean_loss.item():.4f} [Complete]'
                    print(progress_msg)
                    break

                # -------------------------------------------------------------------------
                # CASE 3: Valid boundary detected - Calculate appropriate loss
                # -------------------------------------------------------------------------
                if cfg.attack.loss == 'rsa':
                    loss = chamfer_distance(diverge_boundary_pts_pred, target_boundary_pts, device=device)
                elif cfg.attack.loss == 'eta':
                    loss = outward_inward_loss_interpolated(
                        diverge_boundary_pts_pred, 
                        diverge_boundary_pts_tensor, 
                        target_boundary_pts, 
                        reference_boundary_pts_tensor,
                        visualize=False
                    )
                else:
                    raise ValueError(f'Unknown loss type: {cfg.attack.loss}')

                # update best result
                if loss < best_loss:
                    best_loss = loss
                    best_patch = patches.copy()
                    best_mask = masks.copy()
                    best_patch_cfg = patch_cfgs.copy()
                    best_pseudo_area = pseudo_areas.copy()
                
                # update location best loss
                if loss < location_best_loss:
                    location_best_loss = loss
                
                # early stopping check
                loss_value = loss.item()
                loss_history.append(loss_value)
                
                if step > 0 and len(loss_history) > 1:
                    loss_change = abs(loss_history[-1] - loss_history[-2])
                    if loss_change < min_loss_change:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0  # reset counter if we see progress
                        
                    if stagnation_counter >= stagnation_threshold:
                        progress_msg = f'\rLoc {loc_idx+1}/{len(top_patch_centers)}, Step {step+1}/{max_step}: Loss={loss_value:.4f}, Best(loc)={location_best_loss.item():.4f}, Best(global)={best_loss.item():.4f}, Clean={clean_loss.item():.4f} [Early Stop]'
                        print(progress_msg)
                        break
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Dynamic progress update in one line
                progress_msg = f'\rLoc {loc_idx+1}/{len(top_patch_centers)}, Step {step+1}/{max_step}: Loss={loss_value:.4f}, Best(loc)={location_best_loss.item():.4f}, Best(global)={best_loss.item():.4f}, Clean={clean_loss.item():.4f}'
                print(progress_msg, end='', flush=True)
            
        # Print final summary for this sample
        print(f'\n\nSample {sample_token} completed - Best loss: {best_loss.item():.4f}')
        
        losses.append(best_loss.detach().item())
        
        
        ''' Evaluation '''
        with torch.no_grad():
            
            imgs_adv = imgs.clone().detach().to(device)  # (1, 6, 3, 480, 800)
            
            # apply patches
            for patch_idx, patch_cfg in enumerate(patch_cfgs): 

                patch_trans_list, mask_trans_list, visible_cam_indices = get_phy_patch_mask(
                    best_patch[patch_idx], # tensor
                    best_mask[patch_idx],  # tensor
                    best_patch_cfg[patch_idx],   # dict
                    best_pseudo_area[patch_idx],  # tuple
                    lidar2global, 
                    global2img, 
                    (ori_h, ori_w), 
                    ground=True
                )
                
                if visible_cam_indices is None:
                    continue
                
                # apply patch to all visible cameras
                imgs_adv = apply_patch(
                    imgs_adv, 
                    patch_trans_list, 
                    mask_trans_list, 
                    img_norm_cfg, 
                    visible_cam_indices
                )

            data['img'][0].data[0].detach_()
            data['img'][0].data[0] = imgs_adv
            
            # forward
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
        # visualize prediction results
        gt_data = (gt_bboxes_3d, gt_labels_3d)
        orig_data = clean_result[0]
        attack_data = result[0]
        visualize_attack_results(gt_data, orig_data, attack_data, \
            vis_seg_dir, f'{sample_token}', pc_range, car_img, colors_plt, segment=target_boundary_pts.detach().cpu().numpy(), show_score_thr=show_score_thr)
        
        
        ''' save '''
        # save attack result
        best_patches[sample_token] = {
            'patch_cfg': best_patch_cfg,
            'patch': best_patch,
            'mask': best_mask,
            'pseudo_area': best_pseudo_area
        }
        
        # save map result
        results.extend(result)
        clean_results.extend(clean_result)
        
        # save ground truth for planning
        gt_bboxes = [bbox.tolist() for bbox in gt_bboxes_3d.fixed_num_sampled_points]
        gt_labels = [label.item() for label in gt_labels_3d]
        gt_data = {
            'bboxes': gt_bboxes,
            'labels': gt_labels
        }
        with open(os.path.join(gt_dir, f'{sample_token}.json'), 'w') as f:
            json.dump(gt_data, f)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
            
            
    # save best patches
    with open(os.path.join(attack_dir, 'best_patches.pkl'), 'wb') as f:
        pickle.dump(best_patches, f)
    
    # # print average loss
    # print(f'Average Clean Loss: {np.mean(clean_losses)}')
    # print(f'Average Attack Loss: {np.mean(losses)}')
    # with open(os.path.join(map_results_dir, 'losses.txt'), 'w') as f:
    #     f.write(f'Average Clean Loss: {np.mean(clean_losses)}\n')
    #     f.write(f'Average Attack Loss: {np.mean(losses)}\n')

    return results, clean_results
