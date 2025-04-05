import g2o
import numpy as np
from g2o import OptimizationAlgorithmWithHessian
from numpy.typing import NDArray

from asrl.slam.icp import icp


class GraphICPSLAM2D:
    def __init__(self) -> None:
        self._scans = []
        self._last_transform = np.eye(3)

        self._optimizer = g2o.SparseOptimizer()
        self._solver = g2o.BlockSolverSE2(g2o.LinearSolverEigenSE2())
        self._algorithm = g2o.OptimizationAlgorithmLevenberg(self._solver)
        self._optimizer.set_algorithm(self._algorithm)

    def step(self, scan: NDArray[np.floating]) -> None:
        vertex_id = len(self._scans)

        if 0 == vertex_id:
            self._scans.append(scan)

            # this is the first vertex in the graph so we can't add an edge
            vertex = g2o.VertexSE2()
            vertex.set_id(vertex_id)
            vertex.set_estimate(g2o.SE2(0.0, 0.0, 0.0))
            vertex.set_fixed(True)
            self._optimizer.add_vertex(vertex)
            return

        previous_vertex_id = vertex_id - 1
        previous_scan = self._scans[previous_vertex_id]
        transform, _, _ = icp(previous_scan, scan, self._last_transform)

        if np.linalg.norm(transform[:2]) < 0.1:
            self._last_transform = transform
            return

        self._last_transform = np.eye(3)
        self._scans.append(scan)

        vertex = g2o.VertexSE2()
        vertex.set_id(vertex_id)

        previous_vertex = self._optimizer.vertex(previous_vertex_id)

        measurement = g2o.SE2(transform)

        vertex.set_estimate(measurement)
        self._optimizer.add_vertex(vertex)

        edge = g2o.EdgeSE2()

        edge.set_vertex(0, vertex)
        edge.set_vertex(1, previous_vertex)
        edge.set_measurement(measurement)
        edge.set_information(np.eye(3))

        self._optimizer.add_edge(edge)

    def poses(self) -> NDArray[np.floating]:
        self._optimizer.initialize_optimization()
        self._optimizer.set_verbose(True)
        self._optimizer.optimize(100)
        vertices = len(self._scans)
        poses = np.zeros((vertices, 3))
        for vertex_id in range(vertices):
            poses[vertex_id, :] = self._optimizer.vertex(vertex_id).estimate().to_vector()
        return poses

    def hessian(self) -> NDArray[np.floating]:
        # print(self._optimizer.vertices()[0].hessian_index())
        # print(self._optimizer.vertices()[1].hessian_index())
        # print(self._optimizer.vertices()[2].hessian_index())
        # print(self._optimizer.vertices()[0].get_id())

        # Generate all (i, j) pairs for the Hessian
        # index_pairs = [(i, j) for i in vertex_ids for j in vertex_ids]

        # Compute the marginals (blocks of Hessian)
        # self._optimizer.compute_active_errors()
        hessian_blocks, success = self._optimizer.compute_marginals(self._optimizer.vertices()[3])

        if not success:
            raise RuntimeError("Unable to retrieve hessian blocks")

        row_block_indices = hessian_blocks.row_block_indices()
        col_block_indices = hessian_blocks.col_block_indices()

        row_block_indices = np.r_[0, row_block_indices]
        col_block_indices = np.r_[0, col_block_indices]

        total_rows = row_block_indices[-1]
        total_cols = col_block_indices[-1]

        H = np.zeros((total_rows, total_cols))

        # Iterate through the block entries
        for r in range(row_block_indices.shape[0] - 1):
            for c in range(col_block_indices.shape[0] - 1):
                row_start = row_block_indices[r]
                row_end = row_block_indices[r + 1]
                col_start = col_block_indices[c]
                col_end = col_block_indices[c + 1]

                H[row_start:row_end, col_start:col_end] = hessian_blocks.block(r, c)


        return H
