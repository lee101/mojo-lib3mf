"""Helpers that ask the installed upstream ``lib3mf`` 2.5.0 binding directly.

	Every helper here is a thin wrapper around the real upstream library, so a test
	that uses one compares the port against lib3mf itself rather than against a
	second Python implementation of the same idea. No test imports this module yet
	(see ``bench/verify.md``); ``tests/test_kernels.py`` and ``tests/test_io.py``
	call the binding directly. Where upstream's public API does
	not expose a kernel, the test cites ``tests/upstream_fixtures.py`` instead.
"""

from __future__ import annotations

import io
import zipfile

import lib3mf
import numpy as np

CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
BEAM = "http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02"
BALLS = "http://schemas.microsoft.com/3dmanufacturing/beamlattice/balls/2020/07"

#: Numeric agreement required between the port and upstream. Upstream stores
#: positions, radii and transforms as ``nfFloat`` (binary32), so anything that
#: passes through an XML attribute is matched to within one part in a million
#: of a Float32, and kernel output is compared at Float32 resolution.
FLOAT32 = np.finfo(np.float32).eps
RTOL = 1e-6
ATOL = 1e-6


def wrapper():
    return lib3mf.get_wrapper()


def upstream_model():
    return wrapper().CreateModel()


def _positions(vertices):
    positions = []
    for coordinates in np.asarray(vertices, dtype=np.float32):
        position = lib3mf.Position()
        position.Coordinates[:] = [float(value) for value in coordinates]
        positions.append(position)
    return positions


def _triangles(triangles):
    faces = []
    for indices in np.asarray(triangles, dtype=np.int32):
        face = lib3mf.Triangle()
        face.Indices[:] = [int(value) for value in indices]
        faces.append(face)
    return faces


def reference_mesh(vertices, triangles, name=""):
    """An upstream mesh object with the same geometry, or an exception if upstream rejects it."""
    model = upstream_model()
    mesh = model.AddMeshObject()
    if name:
        mesh.SetName(name)
    mesh.SetGeometry(_positions(vertices), _triangles(triangles))
    return model, mesh


def upstream_manifold_and_oriented(vertices, triangles):
    """``CModelMeshObject::isManifoldAndOriented`` for the same geometry.

    Returns ``None`` when upstream refuses the geometry before the check
    (``SetGeometry`` validates, see ``MeshObject::IsManifoldAndOriented``).
    """
    try:
        _model, mesh = reference_mesh(vertices, triangles)
    except lib3mf.Lib3MF.ELib3MFException:
        return None
    return bool(mesh.IsManifoldAndOriented())


def upstream_transform(fields):
    """Build an upstream ``NMATRIX3`` from a 4x4 row-major matrix."""
    transform = lib3mf.Transform()
    for column in range(4):
        for row in range(3):
            transform.Fields[column][row] = float(fields[row][column])
    return transform


def upstream_outbox(vertices, component_transform=None, build_transform=None):
    """The outbox lib3mf computes from components and build items.

    Upstream applies the accumulated matrix with ``fnMATRIX3_apply`` inside
    ``CMesh::extendOutbox``, so this exercises matrix composition as well.
    """
    model = upstream_model()
    _model, mesh = reference_mesh(vertices, [])
    if component_transform is None:
        model.AddBuildItem(mesh, upstream_transform(build_transform))
    else:
        components = model.AddComponentsObject()
        components.AddComponent(mesh, upstream_transform(component_transform))
        model.AddBuildItem(
            components,
            upstream_transform(build_transform if build_transform is not None else np.eye(4)),
        )
    box = model.GetOutbox()
    return np.array(box.MinCoordinate, np.float32), np.array(box.MaxCoordinate, np.float32)


def _template_package() -> bytes:
    """A package written by upstream itself; used as the shell for hand-built XML."""
    return bytes(upstream_model().QueryWriter("3mf").WriteToBuffer())


def package(model_xml: str, drop: tuple[str, ...] = ()) -> bytes:
    """An OPC package whose model part is ``model_xml`` and whose other parts come from upstream."""
    source = zipfile.ZipFile(io.BytesIO(_template_package()))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for info in source.infolist():
            if info.filename in drop:
                continue
            data = source.read(info.filename)
            if info.filename.endswith("3dmodel.model"):
                data = model_xml.encode()
            archive.writestr(info.filename, data)
    return buffer.getvalue()


def model_xml(resources: str, build: str = "", attributes: str = "", metadata: str = "") -> str:
    """A Core 2015 model part with the beam-lattice and balls namespaces declared."""
    return (
        '<?xml version="1.0"?>'
        f'<model xmlns="{CORE}" xmlns:b="{BEAM}" xmlns:b2="{BALLS}" '
        f'unit="millimeter" {attributes}>'
        f"{metadata}<resources>{resources}</resources><build>{build}</build></model>"
    )


def mesh_object(object_id: int, vertices_xml: str, triangles_xml: str = "", extra: str = "") -> str:
    return (
        f'<object id="{object_id}" type="model"><mesh><vertices>{vertices_xml}</vertices>'
        f"<triangles>{triangles_xml}</triangles>{extra}</mesh></object>"
    )


def beam_lattice(inner: str, attributes: str = 'radius="0.1"') -> str:
    return f"<b:beamlattice {attributes}>{inner}</b:beamlattice>"


def beams(beams_xml: str) -> str:
    return f"<b:beams>{beams_xml}</b:beams>"


def balls(balls_xml: str) -> str:
    return f"<b2:balls>{balls_xml}</b2:balls>"


def read_with_upstream(data: bytes):
    """Read a package with the real lib3mf reader: ``(model, reader)``."""
    model = upstream_model()
    reader = model.QueryReader("3mf")
    reader.ReadFromBuffer(list(data))
    return model, reader


def upstream_beam_lattice(data: bytes):
    """The upstream beam lattice of object 1, plus its reader and ball options."""
    model, reader = read_with_upstream(data)
    lattice = model.GetMeshObjectByID(1).BeamLattice()
    mode, default_radius = lattice.GetBallOptions()
    beam_rows = []
    for index in range(lattice.GetBeamCount()):
        beam = lattice.GetBeam(index)
        beam_rows.append(
            (
                [int(value) for value in beam.Indices],
                [float(value) for value in beam.Radii],
                [int(value) for value in beam.CapModes],
            )
        )
    ball_rows = [
        (int(lattice.GetBall(index).Index), float(lattice.GetBall(index).Radius))
        for index in range(lattice.GetBallCount())
    ]
    return {
        "model": model,
        "reader": reader,
        "lattice": lattice,
        "beams": beam_rows,
        "balls": ball_rows,
        "ball_mode": int(mode),
        "default_ball_radius": float(default_radius),
        "minimum_length": float(lattice.GetMinLength()),
    }


def ball_mode_value(name: str) -> int:
    """Upstream's numeric ``eModelBeamLatticeBallMode`` for a name."""
    return {
        "none": int(lib3mf.BeamLatticeBallMode.BeamLatticeBallModeNone),
        "mixed": int(lib3mf.BeamLatticeBallMode.Mixed),
        "all": int(lib3mf.BeamLatticeBallMode.All),
    }[name]


def cap_mode_name(value: int) -> str:
    """Upstream's ``eModelBeamLatticeCapMode`` as the port's string."""
    return {0: "sphere", 1: "hemisphere", 2: "butt"}[int(value)]
