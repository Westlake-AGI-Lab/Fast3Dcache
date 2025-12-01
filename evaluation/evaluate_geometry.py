# evaluate_geometry.py

# ref:https://github.com/RuiningLi/dso/blob/main/evaluation/evaluate_geometry.py

import os
import os.path as osp
from glob import glob
import numpy as np
import trimesh
from tqdm import tqdm
import argparse
from collections import defaultdict

def icp_align(source, target, max_iterations=50):
    try:
        matrix, transformed_source, cost = trimesh.registration.icp(
            source,
            target,
            max_iterations=max_iterations
        )
        return transformed_source
    except Exception as e:
        print(f"Warning: ICP alignment failed for a sample: {e}")
        return source

def compute_metrics(gen_model, gt_model, n_points=10000, tau=0.05):
    if not hasattr(gen_model, 'vertices') or len(gen_model.vertices) == 0 or \
       not hasattr(gt_model, 'vertices') or len(gt_model.vertices) == 0:
        return None, None

    gen_points, _ = trimesh.sample.sample_surface(gen_model, n_points)
    gt_points, _ = trimesh.sample.sample_surface(gt_model, n_points)
    
    if len(gen_points) == 0 or len(gt_points) == 0:
        return None, None

    gen_points = icp_align(gen_points, gt_points)
    
    from scipy.spatial import cKDTree as KDTree
    gt_tree = KDTree(gt_points)
    pred_tree = KDTree(gen_points)

    pred_to_gt_distances, _ = gt_tree.query(gen_points)
    gt_to_pred_distances, _ = pred_tree.query(gt_points)
    
    precision = np.mean((pred_to_gt_distances < tau).astype(float))
    recall = np.mean((gt_to_pred_distances < tau).astype(float))

    fscore = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    chamfer_distance = (np.mean(pred_to_gt_distances) + np.mean(gt_to_pred_distances)) / 2

    return fscore, chamfer_distance

def process_single_mesh(mesh_path, up_dir, ground_truth_base_path):
    try:
        gen_model = trimesh.load(mesh_path, force='mesh')
        
        if up_dir == "z":
            gen_model.vertices[:, [1, 2]] = gen_model.vertices[:, [2, 1]]
        elif up_dir != "y":
            raise ValueError(f"Invalid up_dir: {up_dir}")

        filename = osp.basename(mesh_path)
        parts = osp.splitext(filename)[0].split('_')
        if len(parts) >= 3:
            object_id = f"{parts[0]}_{parts[2]}"
        else:
            return None, None, None

        gt_dir_path = osp.join(ground_truth_base_path, object_id)
        ground_truth_path = osp.join(gt_dir_path, "mesh.obj")

        if not osp.exists(ground_truth_path):
            return object_id, None, None
        
        gt_model = trimesh.load(ground_truth_path, force='mesh')

        for model in [gen_model, gt_model]:
            if len(model.vertices) == 0: continue
            bounds = model.bounds
            if bounds is None: continue
            center = (bounds[1] + bounds[0]) / 2
            model.apply_translation(-center)
            scale = 1.0 / np.max(model.extents)
            model.apply_scale(scale)

        fscore, chamfer_distance = compute_metrics(gen_model, gt_model)
        return object_id, fscore, chamfer_distance

    except Exception as e:
        filename = osp.basename(mesh_path)
        parts = osp.splitext(filename)[0].split('_')
        object_id = f"{parts[0]}_{parts[2]}" if len(parts) >= 3 else "unknown"
        return object_id, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="--")
    parser.add_argument("--mesh_dir", type=str, required=True, help="your mesh")
    parser.add_argument("--ground_truth_dir", type=str, required=True, help="ground truth")
    parser.add_argument("--up_dir", type=str, default="y", choices=["x", "y", "z"])
    args = parser.parse_args()

    mesh_paths = sorted(glob(osp.join(args.mesh_dir, "*.*")))
    if not mesh_paths:
        exit()

    print(f"Finding {len(mesh_paths)} meshes: Evaluating...")
    
    results_by_object = defaultdict(list)
    
    for mesh_path in tqdm(mesh_paths, desc="Evaluating Meshes"):
        object_id, fscore, chamfer_distance = process_single_mesh(mesh_path, args.up_dir, args.ground_truth_dir)

        if object_id and fscore is not None and chamfer_distance is not None:
            results_by_object[object_id].append((fscore, chamfer_distance))

    print("\n\n--- Results ---")
    
    sorted_object_ids = sorted(results_by_object.keys())
    
    for object_id in sorted_object_ids:
        scores = results_by_object[object_id]
        if scores:
            avg_fscore = np.mean([s[0] for s in scores])
            avg_cd = np.mean([s[1] for s in scores])
            print(f"\n ID: {object_id} (Evaluate {len(scores)} objects)")
            print(f"  - F-Score (x100 scale): {avg_fscore * 100:.4f}")
            print(f"  - CD: {avg_cd:.6f}")
    
    all_results_flat = [score for scores_list in results_by_object.values() for score in scores_list]
    
    if all_results_flat:
        all_fscores = [s[0] for s in all_results_flat]
        all_chamfer_distances = [s[1] for s in all_results_flat]
        
        avg_fscore_total = np.mean(all_fscores)
        avg_cd_total = np.mean(all_chamfer_distances)
        
        print("\n\n--- Final Results ---")
        print(f"Successful: {len(all_results_flat)} / {len(mesh_paths)}")
        print(f"Final F-Score (x100 scale): {avg_fscore_total * 100:.4f}")
        print(f"Final CD: {avg_cd_total:.6f}")
        print("--- Finished! ---")
    else:
        print("\n ERROR!")