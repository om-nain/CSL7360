import g2o
import numpy as np
from helpers import poseRt


def _make_solver():
    """
    Create a g2o solver that works with both g2opy and g2o-python.

    We try a few common solver types in order of preference.
    """
    # Try CSparse first (classic g2opy)
    if hasattr(g2o, "LinearSolverCSparseSE3"):
        linear_solver = g2o.LinearSolverCSparseSE3()
    # Then Cholmod (often present in newer builds)
    elif hasattr(g2o, "LinearSolverCholmodSE3"):
        linear_solver = g2o.LinearSolverCholmodSE3()
    # Fallback to dense solver (always slower but safe for this toy project)
    elif hasattr(g2o, "LinearSolverDenseSE3"):
        linear_solver = g2o.LinearSolverDenseSE3()
    else:
        raise RuntimeError(
            "No suitable g2o LinearSolver*SE3 found. "
            "Checked: LinearSolverCSparseSE3, LinearSolverCholmodSE3, "
            "LinearSolverDenseSE3."
        )

    block_solver = g2o.BlockSolverSE3(linear_solver)
    algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)
    return algorithm


def _get_point_vertex_class():
    """
    Get the correct 3D point vertex class depending on g2o build.

    - Old / g2opy:   VertexSBAPointXYZ
    - New / g2o-python: VertexPointXYZ
    """
    if hasattr(g2o, "VertexSBAPointXYZ"):
        return g2o.VertexSBAPointXYZ
    if hasattr(g2o, "VertexPointXYZ"):
        return g2o.VertexPointXYZ

    raise RuntimeError(
        "g2o has neither VertexSBAPointXYZ nor VertexPointXYZ; "
        "cannot add 3D point vertices."
    )


def optimize(frames, points, local_window, fix_points, verbose=False, rounds=50):
    # select frames to optimize
    if local_window is None:
        local_frames = frames
    else:
        local_frames = frames[-local_window:]

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    opt.set_algorithm(_make_solver())

    # add normalized camera (we use normalized image coords in f.kps)
    cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
    cam.set_id(0)
    opt.add_parameter(cam)

    robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
    graph_frames, graph_points = {}, {}

    PointVertex = _get_point_vertex_class()

    # --- add frame vertices ---
    for f in (local_frames if fix_points else frames):
        pose = f.pose
        se3 = g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3])
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)

        v_se3.set_id(f.id * 2)
        # fix very early frames and anything outside the local window
        v_se3.set_fixed(f.id <= 1 or f not in local_frames)
        opt.add_vertex(v_se3)

        # sanity check
        est = v_se3.estimate()
        assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
        assert np.allclose(pose[0:3, 3], est.translation())

        graph_frames[f] = v_se3

    # --- add point vertices and observation edges ---
    for p in points:
        # skip points not observed in the current local window
        if not any((f in local_frames) for f in p.frames):
            continue

        v_p = PointVertex()
        v_p.set_id(p.id * 2 + 1)
        v_p.set_estimate(p.pt[0:3])
        v_p.set_marginalized(True)
        v_p.set_fixed(fix_points)
        opt.add_vertex(v_p)
        graph_points[p] = v_p

        # add edges for each observation
        for f, idx in zip(p.frames, p.idxs):
            if f not in graph_frames:
                continue
            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_parameter_id(0, 0)
            edge.set_vertex(0, v_p)
            edge.set_vertex(1, graph_frames[f])
            edge.set_measurement(f.kps[idx])
            edge.set_information(np.eye(2))
            edge.set_robust_kernel(robust_kernel)
            opt.add_edge(edge)

    if verbose:
        opt.set_verbose(True)

    opt.initialize_optimization()
    opt.optimize(rounds)

    # --- write back optimized poses ---
    for f, v_se3 in graph_frames.items():
        est = v_se3.estimate()
        R = est.rotation().matrix()
        t = est.translation()
        f.pose = poseRt(R, t)

    # --- write back optimized points ---
    if not fix_points:
        for p, v_p in graph_points.items():
            p.pt = np.array(v_p.estimate())

    return opt.active_chi2()
