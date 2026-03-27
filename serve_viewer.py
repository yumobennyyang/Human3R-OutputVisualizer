#!/usr/bin/env python3
"""
Serve Human3R output data for the HTML viewer.

Loads the saved depth, conf, color, camera, and SMPL data from an output
directory, preprocesses it (3D reconstruction + SMPL forward pass), and
serves the data as JSON over HTTP for the Three.js viewer.

Usage:
    python serve_viewer.py --output_dir <path_to_output_folder> [--port 8888]
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import roma
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import struct

from add_ckpt_path import add_path_to_dust3r


def parse_args():
    parser = argparse.ArgumentParser(description="Serve Human3R output for HTML viewer.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the saved output directory.")
    parser.add_argument("--model_path", type=str, default="src/human3r_896L.pth",
                        help="Path to model checkpoint (for import paths only).")
    parser.add_argument("--port", type=int, default=8888,
                        help="HTTP server port.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for SMPL computation.")
    parser.add_argument("--vis_threshold", type=float, default=1.5,
                        help="Default confidence threshold.")
    parser.add_argument("--downsample_factor", type=int, default=8,
                        help="Default point cloud downsample factor.")
    parser.add_argument("--size", type=int, default=512,
                        help="Image size used during inference.")
    return parser.parse_args()


def load_and_preprocess(args):
    """Load all output data and preprocess into viewer-ready format."""
    outdir = args.output_dir
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Verify structure
    required_dirs = ["depth", "conf", "color", "camera", "smpl"]
    for d in required_dirs:
        p = os.path.join(outdir, d)
        if not os.path.isdir(p):
            print(f"Error: Missing required subdirectory '{d}' in {outdir}")
            sys.exit(1)

    # Set up imports
    add_path_to_dust3r(args.model_path)

    from src.dust3r.utils.geometry import geotrf
    from src.dust3r.utils import SMPL_Layer
    import imageio.v2 as iio

    # Discover frames
    depth_files = sorted(glob.glob(os.path.join(outdir, "depth", "*.npy")))
    B = len(depth_files)
    if B == 0:
        print(f"No saved frames found in {outdir}/depth/")
        sys.exit(1)
    print(f"Found {B} saved frames in {outdir}")

    # Load SMPL colors
    root_dir = os.path.dirname(os.path.abspath(__file__))
    colors_path = os.path.join(root_dir, "src", "models", "smpl_colors.txt")
    smpl_colors = np.loadtxt(colors_path).astype(int)

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
            msks_list.append(np.zeros((H, W), dtype=np.float32))
        else:
            msks_list.append(msk.astype(np.float32) if msk.ndim == 2 else msk[0].astype(np.float32))

    # Stack arrays
    depths = np.stack(depths)
    confs = np.stack(confs)
    colors_np = np.stack(colors)
    cam2worlds_np = np.stack(cam2worlds)
    intrinsics_np = np.stack(intrinsics_list)

    H, W = depths.shape[1], depths.shape[2]

    # Reconstruct 3D points from depth maps
    print("Reconstructing 3D points from depth maps...")
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
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
    print("Transforming to world coordinates...")
    pts3ds_world = np.zeros_like(pts3ds_self)
    for i in range(B):
        R = cam2worlds_np[i, :3, :3]
        t = cam2worlds_np[i, :3, 3]
        pts = pts3ds_self[i].reshape(-1, 3)
        pts3ds_world[i] = (pts @ R.T + t).reshape(H, W, 3)

    # SMPL forward pass
    print("Computing SMPL meshes...")
    num_betas = smpl_shapes[0].shape[-1] if smpl_shapes[0].shape[0] > 0 else 10
    smpl_layer = SMPL_Layer(
        type="smplx", gender="neutral",
        num_betas=num_betas, kid=False, person_center="head"
    )
    smpl_faces = smpl_layer.bm_x.faces.tolist()

    all_frame_data = []

    for i in range(B):
        print(f"  Processing frame {i+1}/{B}...", end="\r")
        n_humans = smpl_shapes[i].shape[0]

        # Point cloud: apply confidence threshold and downsample
        pts_flat = pts3ds_world[i].reshape(-1, 3)
        col_flat = colors_np[i].reshape(-1, 3)
        conf_flat = confs[i].reshape(-1)
        msk_flat = msks_list[i].reshape(-1)

        # SMPL meshes
        mesh_verts = []
        mesh_colors_list = []
        if n_humans > 0:
            pose_t = torch.from_numpy(cam2worlds_np[i]).float().unsqueeze(0)
            intrins_t = torch.from_numpy(intrinsics_np[i]).float().unsqueeze(0).expand(n_humans, -1, -1)
            with torch.no_grad():
                smpl_out = smpl_layer(
                    smpl_rotvecs[i], smpl_shapes[i], smpl_transls[i],
                    None, None, K=intrins_t,
                    expression=smpl_expressions[i]
                )
            # Transform to world coords
            world_verts = geotrf(pose_t, smpl_out["smpl_v3d"].unsqueeze(0))[0]
            for j in range(n_humans):
                mesh_verts.append(world_verts[j].numpy().tolist())
                c = smpl_colors[j % len(smpl_colors)]
                mesh_colors_list.append([int(c[0]), int(c[1]), int(c[2])])

        # Camera data
        focal = float(intrinsics_np[i, 0, 0])
        pp = [float(intrinsics_np[i, 0, 2]), float(intrinsics_np[i, 1, 2])]
        R = cam2worlds_np[i, :3, :3]
        t = cam2worlds_np[i, :3, 3]

        frame_data = {
            "pts": pts_flat.tolist(),
            "colors": col_flat.tolist(),
            "conf": conf_flat.tolist(),
            "msk": msk_flat.tolist(),
            "meshes": mesh_verts,
            "mesh_colors": mesh_colors_list,
            "cam_R": R.tolist(),
            "cam_t": t.tolist(),
            "focal": focal,
            "pp": pp,
        }
        all_frame_data.append(frame_data)

    print(f"\nDone preprocessing {B} frames.")

    return {
        "num_frames": B,
        "height": int(H),
        "width": int(W),
        "smpl_faces": smpl_faces,
        "frames": all_frame_data,
        "defaults": {
            "vis_threshold": args.vis_threshold,
            "downsample_factor": args.downsample_factor,
        }
    }


def create_handler(viewer_data, viewer_html_path):
    """Create an HTTP request handler with access to preloaded data."""

    data_json = json.dumps(viewer_data)
    data_bytes = data_json.encode("utf-8")
    print(f"Data payload size: {len(data_bytes) / 1024 / 1024:.1f} MB")

    class ViewerHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(viewer_html_path, "rb") as f:
                    self.wfile.write(f.read())

            elif path == "/api/data":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data_bytes)))
                self.end_headers()
                self.wfile.write(data_bytes)

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress default logging for cleaner output
            pass

    return ViewerHandler


def main():
    args = parse_args()

    print("=" * 60)
    print("  Human3R HTML Viewer Server")
    print("=" * 60)
    print(f"  Output dir: {args.output_dir}")
    print(f"  Port:       {args.port}")
    print()

    viewer_data = load_and_preprocess(args)

    viewer_html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")
    if not os.path.isfile(viewer_html):
        print(f"Error: viewer.html not found at {viewer_html}")
        sys.exit(1)

    handler = create_handler(viewer_data, viewer_html)
    server = HTTPServer(("0.0.0.0", args.port), handler)

    print(f"\n  Viewer ready at: http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
