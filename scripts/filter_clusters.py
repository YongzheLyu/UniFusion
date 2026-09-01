#!/usr/bin/env python3
"""Cluster-size threshold filtering for a large PLY mesh (CPU-only open3d).

Keeps connected triangle clusters with at least --min-faces triangles,
instead of post_process_mesh's top-N heuristic, so large background walls
survive while small floaters are removed.
"""
import argparse
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--min-faces", type=int, default=2000)
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.input)
    print(f"loaded: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    print(f"clusters: {len(cluster_n_triangles)}")

    keep_clusters = np.where(cluster_n_triangles >= args.min_faces)[0]
    print(f"keeping {len(keep_clusters)} clusters with >= {args.min_faces} faces")
    sizes = np.sort(cluster_n_triangles)[::-1]
    print("top-20 cluster sizes:", sizes[:20].tolist())
    for thr in (100, 500, 1000, 2000, 5000, 10000):
        print(f"  clusters >= {thr}: {(cluster_n_triangles >= thr).sum()}")

    keep_mask = np.isin(triangle_clusters, keep_clusters)
    mesh.remove_triangles_by_mask(~keep_mask)
    mesh.remove_unreferenced_vertices()
    print(f"after filter: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")

    o3d.io.write_triangle_mesh(args.output, mesh)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
