import g2o
import numpy as np

optimizer = g2o.SparseOptimizer()
solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
optimizer.set_algorithm(algorithm)

a = g2o.VertexSE2()
a.set_id(0)
a.set_estimate(g2o.SE2(np.array([0, 0, 0])))
a.set_fixed(True)
optimizer.add_vertex(a)

b = g2o.VertexSE2()
b.set_id(1)
b.set_estimate(g2o.SE2(np.array([0, 0, 0])))
optimizer.add_vertex(b)

c = g2o.VertexSE2()
c.set_id(2)
c.set_estimate(g2o.SE2(np.array([0, 0, 0])))
optimizer.add_vertex(c)

ab = g2o.EdgeSE2()
ab.set_vertex(0, b)
ab.set_vertex(1, a)
ab.set_measurement(g2o.SE2(np.array([1, 0, np.pi / 2])))
ab.set_information(np.eye(3))
optimizer.add_edge(ab)

bc = g2o.EdgeSE2()
bc.set_vertex(0, b)
bc.set_vertex(1, c)
bc.set_measurement(g2o.SE2(np.array([50, 0, np.pi / 2])))
bc.set_information(np.eye(3))
optimizer.add_edge(bc)

optimizer.initialize_optimization()
optimizer.optimize(100)

print(a.estimate().to_vector())
print(b.estimate().to_vector())
print(c.estimate().to_vector())



# bc = g2o.EdgeSE2()
