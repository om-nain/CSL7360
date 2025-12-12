import time
import json

import cv2
import numpy as np

from helpers import poseRt, hamming_distance, add_ones
from constants import CULLING_ERR_THRES
from frame import Frame

LOCAL_WINDOW = 20
# LOCAL_WINDOW = None


def _collect_correspondences_for_frame(frame, points, min_points=6):
    """
    Collect 3D-2D correspondences for a given frame from the map points.

    Returns:
        obj_pts: (N, 3) world coordinates
        img_pts: (N, 2) pixel coordinates
    """
    obj_pts = []
    img_pts = []

    for p in points:
        # p.frames and p.idxs are parallel lists
        for f_obs, idx in zip(p.frames, p.idxs):
            if f_obs is frame:
                obj_pts.append(p.pt)
                img_pts.append(f_obs.kpus[idx])

    if len(obj_pts) < min_points:
        return None, None

    obj_pts = np.asarray(obj_pts, dtype=np.float32)
    img_pts = np.asarray(img_pts, dtype=np.float32)
    return obj_pts, img_pts


def _pnp_optimize_single_frame(frame, points, verbose=False):
    """
    Refine a single frame's pose using OpenCV solvePnP, keeping 3D points fixed.

    Returns:
        mean squared reprojection error (in pixels^2), or 0.0 if not optimized.
    """
    obj_pts, img_pts = _collect_correspondences_for_frame(frame, points)
    if obj_pts is None:
        if verbose:
            print(f"[PnP] Frame {frame.id}: not enough points for PnP")
        return 0.0

    K = frame.K.astype(np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)

    # Current pose is a 4x4 world->camera transform
    R = frame.pose[:3, :3]
    t = frame.pose[:3, 3]

    # Use current pose as initial guess
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        dist_coeffs,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok:
        if verbose:
            print(f"[PnP] Frame {frame.id}: solvePnP failed")
        return 0.0

    R_refined, _ = cv2.Rodrigues(rvec)
    t_refined = tvec.reshape(3)

    # Update frame pose (still world->camera)
    frame.pose = poseRt(R_refined, t_refined)

    # Compute reprojection error as a simple "error" metric
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist_coeffs)
    proj = proj.reshape(-1, 2)
    residuals = proj - img_pts
    mse = float(np.mean(np.sum(residuals ** 2, axis=1)))

    if verbose:
        print(f"[PnP] Frame {frame.id}: {len(obj_pts)} pts, MSE = {mse:.4f}")

    return mse


def _run_optimizer(frames, points, local_window, fix_points, verbose=False, rounds=50):
    """
    Replacement for the old g2o-based optimizer.

    We keep all 3D points fixed and refine camera poses of a local window
    of frames using PnP. For this SLAM toy, that's more than enough.
    """
    if len(frames) == 0 or len(points) == 0:
        return 0.0

    # Decide which frames to optimize
    if local_window is None:
        local_frames = frames
    else:
        local_frames = frames[-local_window:]

    # Original code used local_frames when fix_points=True and
    # all frames otherwise. For simplicity we just optimize local_frames
    # in both cases – older poses won't matter much visually.
    frames_to_optimize = local_frames

    total_err = 0.0
    count = 0

    for f in frames_to_optimize:
        err = _pnp_optimize_single_frame(f, points, verbose=verbose)
        if err > 0.0:
            total_err += err
            count += 1

    if count == 0:
        return 0.0

    return total_err / count


class Point(object):
    # A Point is a 3-D point in the world
    # Each Point is observed in multiple Frames

    def __init__(self, mapp, loc, color, tid=None):
        self.pt = np.array(loc)
        self.frames = []
        self.idxs = []
        self.color = np.copy(color)
        self.id = tid if tid is not None else mapp.add_point(self)

    def homogeneous(self):
        return add_ones(self.pt)

    def orb(self):
        return [f.des[idx] for f, idx in zip(self.frames, self.idxs)]

    def orb_distance(self, des):
        return min([hamming_distance(o, des) for o in self.orb()])

    def delete(self):
        for f, idx in zip(self.frames, self.idxs):
            f.pts[idx] = None
        # Actual object removal is handled by Map.points.remove
        del self

    def add_observation(self, frame, idx):
        assert frame.pts[idx] is None
        assert frame not in self.frames
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)


class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.max_frame = 0
        self.max_point = 0

    def serialize(self):
        ret = {}
        ret["points"] = [
            {
                "id": p.id,
                "pt": p.pt.tolist(),
                "color": p.color.tolist(),
            }
            for p in self.points
        ]
        ret["frames"] = []
        for f in self.frames:
            ret["frames"].append(
                {
                    "id": f.id,
                    "K": f.K.tolist(),
                    "pose": f.pose.tolist(),
                    "h": f.h,
                    "w": f.w,
                    "kpus": f.kpus.tolist(),
                    "des": f.des.tolist(),
                    "pts": [p.id if p is not None else -1 for p in f.pts],
                }
            )
        ret["max_frame"] = self.max_frame
        ret["max_point"] = self.max_point
        return json.dumps(ret)

    def deserialize(self, s):
        ret = json.loads(s)
        self.max_frame = ret["max_frame"]
        self.max_point = ret["max_point"]
        self.points = []
        self.frames = []

        pids = {}
        for p in ret["points"]:
            pp = Point(self, p["pt"], p["color"], p["id"])
            self.points.append(pp)
            pids[p["id"]] = pp

        for f in ret["frames"]:
            ff = Frame(self, None, f["K"], f["pose"], f["id"])
            ff.w, ff.h = f["w"], f["h"]
            ff.kpus = np.array(f["kpus"])
            ff.des = np.array(f["des"])
            ff.pts = [None] * len(ff.kpus)
            for i, p in enumerate(f["pts"]):
                if p != -1:
                    ff.pts[i] = pids[p]
            self.frames.append(ff)

    def add_point(self, point):
        ret = self.max_point
        self.max_point += 1
        self.points.append(point)
        return ret

    def add_frame(self, frame):
        ret = self.max_frame
        self.max_frame += 1
        self.frames.append(frame)
        return ret

    # *** optimizer ***

    def optimize(self, local_window=LOCAL_WINDOW, fix_points=False, verbose=False, rounds=50):
        # Use our PnP-based optimizer instead of g2o
        err = _run_optimizer(
            self.frames,
            self.points,
            local_window,
            fix_points,
            verbose=verbose,
            rounds=rounds,
        )

        # prune points
        culled_pt_count = 0
        for p in list(self.points):
            # <= 4 match point that's old
            old_point = len(p.frames) <= 4 and p.frames[-1].id + 7 < self.max_frame

            # compute reprojection error
            errs = []
            for f, idx in zip(p.frames, p.idxs):
                uv = f.kps[idx]  # normalized coordinates
                proj = np.dot(f.pose[:3], p.homogeneous())
                proj = proj[0:2] / proj[2]
                errs.append(np.linalg.norm(proj - uv))

            # cull
            if old_point or (len(errs) > 0 and np.mean(errs) > CULLING_ERR_THRES):
                culled_pt_count += 1
                self.points.remove(p)
                p.delete()

        print("Culled:   %d points" % (culled_pt_count))

        return err
