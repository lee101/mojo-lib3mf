import numpy as np

import mojo_lib3mf as m3


vertices = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    dtype=np.float32,
)
triangles = np.array(
    [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    dtype=np.int32,
)

mesh = m3.Mesh(vertices, triangles)
assert mesh.check_sanity()
assert mesh.is_manifold_and_oriented()

transform = m3.identity_transform()
transform[:3, 3] = [10, 20, 30]
model = m3.Model(
    objects=[m3.MeshObject(1, mesh, name="tetrahedron")],
    build_items=[m3.BuildItem(1, transform)],
)
model.write("tetrahedron.3mf")

loaded = m3.Model.read("tetrahedron.3mf")
print(loaded.objects[0].mesh.triangle_normals())
