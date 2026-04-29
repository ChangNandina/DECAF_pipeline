import requests
import json
import nibabel as nib
import numpy as np
import gzip
import base64
import os
import time


def decode_response(response_json, shape):
    result_gz  = base64.b64decode(response_json["result"])
    result_raw = gzip.decompress(result_gz)
    return np.frombuffer(result_raw, dtype=np.int8).reshape(shape)


def compute_dice(mask_a, mask_b):
    a, b = (mask_a > 0).astype(bool), (mask_b > 0).astype(bool)
    inter = (a & b).sum()
    total = a.sum() + b.sum()
    return 2.0 * inter / total if total > 0 else 0.0


def run_partb(nifti_dir, mask_path, output_dir,
              skip_segments=None, n_phases=25, file_prefix="",
              server="http://localhost:8911"):
    """Plan B+ Part B: batch inference for all phases.

    Requires Part A to have been run first (segments.json, segment_seeds.json,
    dist_maps.npz, soft_regions.npz must exist in output_dir).

    Args:
        nifti_dir:     directory with cropped 480³ NIfTI phase files
        mask_path:     vessel mask NIfTI (used for Phase 1 Dice only)
        output_dir:    Part A output directory (inputs + outputs go here)
        skip_segments: list of segment IDs to exclude
        n_phases:      number of cardiac phases
        file_prefix:   prefix on NIfTI filenames (e.g. "" or "r_")
        server:        nninteractive server URL
    """
    if skip_segments is None:
        skip_segments = []

    print("=" * 60)
    print("Plan B+ Part B")
    print(f"  nifti_dir     : {nifti_dir}")
    print(f"  output_dir    : {output_dir}")
    print(f"  skip_segments : {skip_segments}")
    print("=" * 60)

    mask_nib   = nib.load(mask_path)
    mask_data  = mask_nib.get_fdata()
    mask_shape = mask_data.shape

    with open(os.path.join(output_dir, "segments.json")) as f:
        seg_data = json.load(f)
    segments = {int(sid): np.array(coords) for sid, coords in seg_data.items()}

    with open(os.path.join(output_dir, "segment_seeds.json")) as f:
        seeds_data = json.load(f)
    segment_seeds = {int(k): np.array(v) for k, v in seeds_data['seeds'].items()}

    for sid in skip_segments:
        segment_seeds.pop(sid, None)
    print(f"  Active segments: {len(segment_seeds)}")

    dm_npz    = np.load(os.path.join(output_dir, "dist_maps.npz"))
    dist_maps = {int(k): dm_npz[k] for k in dm_npz.files}

    sr_npz       = np.load(os.path.join(output_dir, "soft_regions.npz"))
    soft_regions = {int(k): sr_npz[k] for k in sr_npz.files}

    found_phases = [p for p in range(1, n_phases + 1)
                    if os.path.exists(os.path.join(nifti_dir,
                                                   f"{file_prefix}phase{p}.nii.gz"))]
    print(f"  Found {len(found_phases)}/{n_phases} phases")

    all_volumes = {}
    seg_ids     = sorted(segment_seeds.keys())

    for phase in found_phases:
        print(f"\n--- Phase {phase:02d} ---")
        t_start = time.time()

        nifti_path = os.path.join(nifti_dir, f"{file_prefix}phase{phase}.nii.gz")
        img_nib    = nib.load(nifti_path)
        img_data   = img_nib.get_fdata().astype(np.float32)
        affine     = img_nib.affine
        voxel_vol  = np.abs(np.linalg.det(affine[:3, :3]))

        img_gz     = gzip.compress(img_data.tobytes(), compresslevel=4)
        dimensions = [int(d) for d in img_data.shape[::-1]]

        r          = requests.get(f"{server}/start_session")
        session_id = r.json()["session_id"]
        r = requests.post(f"{server}/upload_raw/{session_id}",
                          files={"file": ("image.raw.gz", img_gz, "application/octet-stream")},
                          data={"metadata": json.dumps({"dimensions": dimensions})})
        del img_gz
        if r.status_code != 200:
            print(f"  Upload FAILED")
            all_volumes[phase] = {sid: 0.0 for sid in seg_ids}
            continue

        clipped_masks = {}
        for sid, seeds in segment_seeds.items():
            requests.get(f"{server}/reset_interactions/{session_id}")
            last_response = None
            for pt in seeds:
                x, y, z = int(pt[2]), int(pt[1]), int(pt[0])
                r = requests.get(f"{server}/process_point_interaction/{session_id}",
                                 params={"x": x, "y": y, "z": z, "foreground": True})
                if r.status_code == 200:
                    last_response = r.json()

            if last_response and last_response.get("status") == "success":
                raw_fg           = (decode_response(last_response, mask_shape) > 0)
                clipped          = raw_fg & soft_regions[sid]
                clipped_masks[sid] = clipped
                print(f"    Seg {sid}: {raw_fg.sum()} -> {clipped.sum()} kept")
            else:
                print(f"    Seg {sid}: FAILED")
                clipped_masks[sid] = np.zeros(mask_shape, dtype=bool)

        requests.get(f"{server}/end_session/{session_id}")

        multilabel     = np.zeros(mask_shape, dtype=np.uint8)
        coverage_count = np.zeros(mask_shape, dtype=np.int32)
        for sid, mask in clipped_masks.items():
            coverage_count[mask] += 1
        for sid, mask in clipped_masks.items():
            multilabel[mask & (coverage_count == 1)] = sid

        overlap_zone = coverage_count >= 2
        if int(overlap_zone.sum()) > 0:
            best_dist = np.full(mask_shape, np.inf, dtype=np.float32)
            for sid, mask in clipped_masks.items():
                contest = mask & overlap_zone
                closer  = dist_maps[sid] < best_dist
                update  = contest & closer
                multilabel[update] = sid
                best_dist[update]  = dist_maps[sid][update]

        binary_combined = (multilabel > 0)
        binary_voxels   = int(binary_combined.sum())
        print(f"  Merge: {binary_voxels} voxels, {binary_voxels*voxel_vol:.2f} mm³")

        nib.save(nib.Nifti1Image(binary_combined.astype(np.uint8), affine),
                 os.path.join(output_dir, f"p{phase:02d}_binary.nii.gz"))
        nib.save(nib.Nifti1Image(multilabel, affine),
                 os.path.join(output_dir, f"p{phase:02d}_multilabel.nii.gz"))

        if phase == 1:
            dice = compute_dice(binary_combined, mask_data)
            print(f"  Phase 1 Dice vs GT: {dice:.4f}")

        phase_volumes = {sid: int((multilabel == sid).sum()) * voxel_vol
                         for sid in seg_ids}
        all_volumes[phase] = phase_volumes
        print(f"  Phase {phase:02d} done ({time.time()-t_start:.1f}s)")

    csv_path = os.path.join(output_dir, "volumes.csv")
    with open(csv_path, 'w') as f:
        f.write("Phase," + ",".join([f"Seg{sid}" for sid in seg_ids]) + ",Total\n")
        for phase in sorted(all_volumes.keys()):
            vals  = [all_volumes[phase].get(sid, 0) for sid in seg_ids]
            total = sum(vals)
            f.write(f"{phase}," + ",".join(f"{v:.2f}" for v in vals) + f",{total:.2f}\n")

    print(f"\nPart B complete  ->  {output_dir}")
    print(f"Volumes CSV: {csv_path}")
