#!/usr/bin/env python3
"""
Visualize previously saved Human3R outputs without re-running the model.

Loads the saved depth, conf, color, camera, and smpl data from an output
directory and launches the SceneHumanViewer.

Example:
    python visualize_saved.py --output_dir output \
        --vis_threshold 2 --downsample_factor 1
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import roma

from add_ckpt_path import add_path_to_dust3r


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize saved Human3R outputs."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to the saved output directory (contains depth/, conf/, color/, camera/, smpl/ folders).",
    )
    parser.add_argument(
        "--model_path", type=str, default="src/human3r_896L.pth",
        help="Path to the model checkpoint (needed only to set up import paths, model is NOT loaded).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for SMPL computation ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--vis_threshold", type=float, default=1.5,
        help="Visualization threshold for the viewer (1 to INF).",
    )
    parser.add_argument(
        "--msk_threshold", type=float, default=0.1,
        help="Mask threshold (0 to 1).",
    )
    parser.add_argument(
        "--downsample_factor", type=int, default=10,
        help="Point cloud downsample factor for the viewer.",
    )
    parser.add_argument(
        "--smpl_downsample", type=int, default=1,
        help="SMPL sequence downsample factor for the viewer.",
    )
    parser.add_argument(
        "--camera_downsample", type=int, default=1,
        help="Camera motion downsample factor for the viewer.",
    )
    parser.add_argument(
        "--mask_morph", type=int, default=10,
        help="Mask morphology for the viewer.",
    )
    parser.add_argument(
        "--size", type=int, default=512,
        help="Image size used during inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = args.output_dir

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Verify the output directory structure
    required_dirs = ["depth", "conf", "color", "camera", "smpl"]
    for d in required_dirs:
        p = os.path.join(outdir, d)
        if not os.path.isdir(p):
            print(f"Error: Missing required subdirectory '{d}' in {outdir}")
            sys.exit(1)

    # Set up import paths (no model is actually loaded)
    add_path_to_dust3r(args.model_path)

    from src.dust3r.utils.geometry import geotrf
    from src.dust3r.utils import SMPL_Layer
    from viser_utils import SceneHumanViewer
    import imageio.v2 as iio

    # Discover frames by looking at the depth folder
    depth_files = sorted(glob.glob(os.path.join(outdir, "depth", "*.npy")))
    B = len(depth_files)
    if B == 0:
        print(f"No saved frames found in {outdir}/depth/")
        sys.exit(1)
    print(f"Found {B} saved frames in {outdir}")

    # Load all per-frame data
    print("Loading saved data...")
    depths = []
    confs = []
    colors = []
    cam2worlds = []
    intrinsics_list = []
    smpl_shapes = []
    smpl_rotvecs = []
    smpl_transls = []
    smpl_expressions = []
    msks_list = []

    for i in range(B):
        frame_id = f"{i:06d}"

        depths.append(np.load(os.path.join(outdir, "depth", f"{frame_id}.npy")))
        confs.append(np.load(os.path.join(outdir, "conf", f"{frame_id}.npy")))
        color = iio.imread(os.path.join(outdir, "color", f"{frame_id}.png"))
        colors.append(color.astype(np.float32) / 255.0)

        cam_data = np.load(os.path.join(outdir, "camera", f"{frame_id}.npz"))
        cam2worlds.append(cam_data["pose"])
        intrinsics_list.append(cam_data["intrinsics"])

        smpl_data = np.load(
            os.path.join(outdir, "smpl", f"{frame_id}.npz"), allow_pickle=True
        )
        smpl_shapes.append(torch.from_numpy(smpl_data["shape"]).float())
        smpl_rotvecs.append(torch.from_numpy(smpl_data["rotvec"]).float())
        smpl_transls.append(torch.from_numpy(smpl_data["transl"]).float())

        expr = smpl_data["expression"]
        if expr is None or (isinstance(expr, np.ndarray) and expr.ndim == 0):
            smpl_expressions.append(None)
        else:
            smpl_expressions.append(torch.from_numpy(expr).float())

        msk = smpl_data["msk"]
        if msk is None or (isinstance(msk, np.ndarray) and msk.ndim == 0):
            H, W = depths[-1].shape
            msks_list.append(torch.zeros(1, H, W))
        else:
            msks_list.append(torch.from_numpy(msk).float())

    # Stack arrays
    depths = np.stack(depths)          # B, H, W
    confs = np.stack(confs)            # B, H, W
    colors_np = np.stack(colors)       # B, H, W, 3
    cam2worlds_np = np.stack(cam2worlds)  # B, 4, 4
    intrinsics_np = np.stack(intrinsics_list)  # B, 3, 3

    H, W = depths.shape[1], depths.shape[2]

    # Reconstruct pts3d_self from depth + intrinsics
    # pts3d_self[..., 2] = depth, and x,y are computed from pixel coords + intrinsics
    print("Reconstructing 3D points from depth maps...")
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))  # W, H grid
    pts3ds_self = np.zeros((B, H, W, 3), dtype=np.float32)
    for i in range(B):
        fx = intrinsics_np[i, 0, 0]
        fy = intrinsics_np[i, 1, 1]
        cx = intrinsics_np[i, 0, 2]
        cy = intrinsics_np[i, 1, 2]
        z = depths[i]
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        pts3ds_self[i] = np.stack([x, y, z], axis=-1)

    # Transform to world coordinates
    pts3ds_world = []
    for i in range(B):
        pose = torch.from_numpy(cam2worlds_np[i]).float().unsqueeze(0)  # 1, 4, 4
        pts = torch.from_numpy(pts3ds_self[i]).float().unsqueeze(0)    # 1, H, W, 3
        pts3ds_world.append(geotrf(pose, pts))

    # Build cam_dict
    focal = np.array([intrinsics_np[i, 0, 0] for i in range(B)])
    pp = np.stack([
        [intrinsics_np[i, 0, 2], intrinsics_np[i, 1, 2]] for i in range(B)
    ])
    R = cam2worlds_np[:, :3, :3]
    t = cam2worlds_np[:, :3, 3]
    cam_dict = {"focal": focal, "pp": pp, "R": R, "t": t}

    # Run SMPL forward pass to get vertices
    print("Computing SMPL meshes...")
    num_betas = smpl_shapes[0].shape[-1] if smpl_shapes[0].shape[0] > 0 else 10
    smpl_layer = SMPL_Layer(
        type="smplx", gender="neutral",
        num_betas=num_betas, kid=False, person_center="head"
    )
    smpl_faces = smpl_layer.bm_x.faces

    all_verts = []
    smpl_ids = []
    for i in range(B):
        n_humans = smpl_shapes[i].shape[0]
        if n_humans > 0:
            pose_t = torch.from_numpy(cam2worlds_np[i]).float().unsqueeze(0)
            intrins_t = torch.from_numpy(intrinsics_np[i]).float().unsqueeze(0).expand(n_humans, -1, -1)
            with torch.no_grad():
                smpl_out = smpl_layer(
                    smpl_rotvecs[i], smpl_shapes[i], smpl_transls[i],
                    None, None, K=intrins_t,
                    expression=smpl_expressions[i]
                )
            # Transform to world coordinates
            all_verts.append(
                geotrf(pose_t, smpl_out["smpl_v3d"].unsqueeze(0))[0]
            )
        else:
            all_verts.append(torch.empty(0))
        # smpl_id wasn't saved; use zeros (all same identity color)
        smpl_ids.append(torch.zeros(n_humans).long())

    # Prepare visualization arrays
    pts3ds_to_vis = [p.squeeze(0).numpy() for p in pts3ds_world]
    colors_to_vis = [colors_np[i:i+1] for i in range(B)]
    conf_to_vis = [confs[i:i+1] for i in range(B)]
    msks_to_vis = [m.numpy() for m in msks_list]
    edge_colors = [None] * B
    verts_to_vis = [v.numpy() if isinstance(v, torch.Tensor) and v.numel() > 0 else v.numpy() for v in all_verts]

    # Launch viewer
    print("Launching Human3R viewer...")
    viewer = SceneHumanViewer(
        pts3ds_to_vis,
        colors_to_vis,
        conf_to_vis,
        cam_dict,
        verts_to_vis,
        smpl_faces,
        smpl_ids,
        msks_to_vis,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        msk_threshold=args.msk_threshold,
        mask_morph=args.mask_morph,
        size=args.size,
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=args.smpl_downsample,
        camera_downsample_factor=args.camera_downsample,
    )
    viewer.run()


if __name__ == "__main__":
    main()
