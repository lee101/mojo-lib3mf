from __future__ import annotations

import io
import zipfile
from xml.etree import ElementTree as ET

import lib3mf
import numpy as np
import pytest

import mojo_lib3mf as m3


def sample_model(with_beams: bool = False) -> m3.Model:
    vertices = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], np.float32
    )
    triangles = np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], np.int32
    )
    lattice = None
    if with_beams:
        lattice = m3.BeamLattice(
            [[0, 1], [1, 2]],
            [[0.1, 0.2], [0.3, 0.3]],
            [("butt", "hemisphere"), ("sphere", "sphere")],
            minimum_length=0.0001,
            default_radius=0.1,
            ball_mode="mixed",
            default_ball_radius=0.2,
            ball_indices=np.array([0], np.int32),
            ball_radii=np.array([0.25]),
        )
    transform = m3.identity_transform()
    transform[:3, 3] = [4, 5, 6]
    return m3.Model(
        [m3.MeshObject(1, m3.Mesh(vertices, triangles, lattice), "tetra")],
        [m3.BuildItem(1, transform, "part-7")],
        metadata={"Title": "Parity tetrahedron"},
    )


def test_opc_and_model_xml_roundtrip_preserves_core_data():
    expected = sample_model()
    stream = io.BytesIO()
    m3.write(expected, stream)
    stream.seek(0)
    actual = m3.read(stream)
    assert actual.unit == expected.unit
    assert actual.metadata == expected.metadata
    assert actual.objects[0].name == "tetra"
    np.testing.assert_array_equal(
        actual.objects[0].mesh.vertices, expected.objects[0].mesh.vertices
    )
    np.testing.assert_array_equal(
        actual.objects[0].mesh.triangles, expected.objects[0].mesh.triangles
    )
    np.testing.assert_array_equal(
        actual.build_items[0].transform, expected.build_items[0].transform
    )
    assert actual.build_items[0].part_number == "part-7"


@pytest.mark.parametrize(
    "unit", ["micron", "millimeter", "centimeter", "inch", "foot", "meter"]
)
def test_all_supported_core_units_roundtrip(unit):
    expected = sample_model()
    expected.unit = unit
    expected.language = "fr-CA"
    stream = io.BytesIO()
    expected.write(stream)
    stream.seek(0)
    actual = m3.read(stream)
    assert actual.unit == unit
    assert actual.language == "fr-CA"


def test_same_upstream_binding_reads_our_opc_mesh_and_transform(tmp_path):
    path = tmp_path / "ours.3mf"
    m3.write(sample_model(), path)
    wrapper = lib3mf.get_wrapper()
    reference = wrapper.CreateModel()
    reference.QueryReader("3mf").ReadFromFile(str(path))
    mesh = reference.GetMeshObjectByID(1)
    assert mesh.GetName() == "tetra"
    assert mesh.GetVertexCount() == 4
    assert mesh.GetTriangleCount() == 4
    assert mesh.IsManifoldAndOriented()
    vertices = np.array(
        [list(position.Coordinates) for position in mesh.GetVertices()], np.float32
    )
    np.testing.assert_array_equal(vertices, sample_model().objects[0].mesh.vertices)
    items = reference.GetBuildItems()
    assert items.MoveNext()
    item = items.GetCurrent()
    assert item.HasObjectTransform()
    fields = item.GetObjectTransform().Fields
    matrix = np.eye(4, dtype=np.float32)
    for column in range(4):
        for row in range(3):
            matrix[row, column] = fields[column][row]
    np.testing.assert_array_equal(matrix, sample_model().build_items[0].transform)


def test_our_reader_reads_package_written_by_same_upstream(tmp_path):
    wrapper = lib3mf.get_wrapper()
    reference = wrapper.CreateModel()
    reference.SetLanguage("de-DE")
    mesh = reference.AddMeshObject()
    mesh.SetName("upstream")
    positions = []
    for xyz in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
        position = lib3mf.Position()
        position.Coordinates[:] = xyz
        positions.append(position)
    triangle = lib3mf.Triangle()
    triangle.Indices[:] = (0, 1, 2)
    mesh.SetGeometry(positions, [triangle])
    transform = wrapper.GetTranslationTransform(2, 3, 4)
    reference.AddBuildItem(mesh, transform)
    path = tmp_path / "upstream.3mf"
    reference.QueryWriter("3mf").WriteToFile(str(path))

    actual = m3.read(path)
    assert actual.language == "de-DE"
    assert actual.objects[0].name == "upstream"
    np.testing.assert_array_equal(
        actual.objects[0].mesh.vertices,
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
    )
    np.testing.assert_array_equal(
        actual.build_items[0].transform[:3, 3], [2, 3, 4]
    )


def test_beam_lattice_roundtrip_and_same_upstream_parity(tmp_path):
    expected = sample_model(with_beams=True)
    path = tmp_path / "beam.3mf"
    expected.write(path)
    actual = m3.read(path).objects[0].mesh.beam_lattice
    np.testing.assert_array_equal(actual.beams, [[0, 1], [1, 2]])
    np.testing.assert_allclose(actual.radii, [[0.1, 0.2], [0.3, 0.3]], rtol=1e-6)
    assert actual.caps == (("butt", "hemisphere"), ("sphere", "sphere"))
    np.testing.assert_array_equal(actual.ball_indices, [0])
    np.testing.assert_allclose(actual.ball_radii, [0.25])

    wrapper = lib3mf.get_wrapper()
    reference = wrapper.CreateModel()
    reference.QueryReader("3mf").ReadFromFile(str(path))
    lattice = reference.GetMeshObjectByID(1).BeamLattice()
    assert lattice.GetBeamCount() == 2
    assert lattice.GetBallCount() == 1
    beams = lattice.GetBeams()
    np.testing.assert_allclose(
        [list(beam.Radii) for beam in beams], expected.objects[0].mesh.beam_lattice.radii
    )
    assert [int(mode) for mode in beams[0].CapModes] == [2, 1]


def test_our_reader_reads_beams_written_by_same_upstream(tmp_path):
    wrapper = lib3mf.get_wrapper()
    reference = wrapper.CreateModel()
    mesh = reference.AddMeshObject()
    positions = []
    for xyz in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
        position = lib3mf.Position()
        position.Coordinates[:] = xyz
        positions.append(position)
    triangle = lib3mf.Triangle()
    triangle.Indices[:] = (0, 1, 2)
    mesh.SetGeometry(positions, [triangle])
    lattice = mesh.BeamLattice()
    lattice.SetMinLength(0.01)
    beam = lib3mf.Beam()
    beam.Indices[:] = (0, 1)
    beam.Radii[:] = (0.1, 0.2)
    beam.CapModes[:] = (
        lib3mf.BeamLatticeCapMode.Butt,
        lib3mf.BeamLatticeCapMode.HemiSphere,
    )
    lattice.AddBeam(beam)
    lattice.SetBallOptions(lib3mf.BeamLatticeBallMode.Mixed, 0.3)
    ball = lib3mf.Ball()
    ball.Index = 0
    ball.Radius = 0.25
    lattice.AddBall(ball)
    reference.AddBuildItem(mesh, wrapper.GetIdentityTransform())
    path = tmp_path / "upstream-beam.3mf"
    reference.QueryWriter("3mf").WriteToFile(str(path))

    actual = m3.read(path).objects[0].mesh.beam_lattice
    np.testing.assert_array_equal(actual.beams, [[0, 1]])
    np.testing.assert_allclose(actual.radii, [[0.1, 0.2]])
    assert actual.caps == (("butt", "hemisphere"),)
    assert actual.minimum_length == pytest.approx(0.01)
    assert actual.ball_mode == "mixed"
    assert actual.default_ball_radius == pytest.approx(0.3)
    np.testing.assert_array_equal(actual.ball_indices, [0])
    np.testing.assert_allclose(actual.ball_radii, [0.25])


def test_components_and_component_transform_roundtrip(tmp_path):
    base = sample_model()
    matrix = m3.identity_transform()
    matrix[0, 3] = 9
    base.objects.append(m3.ComponentsObject(2, [m3.Component(1, matrix)], "assembly"))
    base.build_items = [m3.BuildItem(2)]
    path = tmp_path / "components.3mf"
    m3.write(base, path)
    actual = m3.read(path)
    component_object = actual.objects[1]
    assert component_object.name == "assembly"
    assert component_object.components[0].object_id == 1
    np.testing.assert_array_equal(component_object.components[0].transform, matrix)


def test_opc_contains_required_parts_and_relationship():
    stream = io.BytesIO()
    m3.write(sample_model(), stream, compression=0)
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        assert set(archive.namelist()) == {
            "[Content_Types].xml",
            "_rels/.rels",
            "3D/3dmodel.model",
        }
        relationships = ET.fromstring(archive.read("_rels/.rels"))
        relationship = next(iter(relationships))
        assert relationship.attrib["Target"] == "/3D/3dmodel.model"
        assert relationship.attrib["Type"].endswith("/3dmodel")


def test_invalid_or_unsafe_opc_inputs_are_rejected():
    with pytest.raises(ValueError, match="OPC ZIP"):
        m3.read(io.BytesIO(b"not a zip"))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "_rels/.rels",
            b"""<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
            <Relationship Id="r" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" Target="../secret"/>
            </Relationships>""",
        )
    stream.seek(0)
    with pytest.raises(ValueError, match="relationship target"):
        m3.read(stream)


def test_writer_rejects_upstream_invalid_mesh_cases(tmp_path):
    duplicate_index = m3.Model(
        [
            m3.MeshObject(
                1,
                m3.Mesh(
                    np.array([[0, 0, 0], [1, 0, 0]], np.float32),
                    np.array([[0, 0, 1]], np.int32),
                ),
            )
        ]
    )
    with pytest.raises(ValueError, match="sanity"):
        duplicate_index.write(tmp_path / "bad.3mf")

    nonfinite = m3.Model(
        [m3.MeshObject(1, m3.Mesh(np.array([[np.nan, 0, 0]], np.float32), []))]
    )
    with pytest.raises(ValueError, match="non-finite"):
        nonfinite.write(tmp_path / "nan.3mf")


def test_writer_validates_before_replacing_destination(tmp_path):
    destination = tmp_path / "existing.3mf"
    destination.write_bytes(b"keep me")
    invalid = m3.Model(
        [
            m3.MeshObject(
                1,
                m3.Mesh(
                    np.array([[0, 0, 0]], np.float32),
                    np.array([[0, 0, 0]], np.int32),
                ),
            )
        ]
    )
    with pytest.raises(ValueError, match="sanity"):
        invalid.write(destination)
    assert destination.read_bytes() == b"keep me"


def test_reader_rejects_nonfinite_ball_radius():
    stream = io.BytesIO()
    model_xml = b"""<?xml version="1.0"?>
    <model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
      xmlns:b="http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02"
      xmlns:b2="http://schemas.microsoft.com/3dmanufacturing/beamlattice/balls/2020/07">
      <resources><object id="1"><mesh>
        <vertices><vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/></vertices>
        <triangles/>
        <b:beamlattice radius="0.1"><b:beams><b:beam v1="0" v2="1"/></b:beams>
          <b2:balls><b2:ball vindex="0" r="nan"/></b2:balls>
        </b:beamlattice>
      </mesh></object></resources><build/>
    </model>"""
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "_rels/.rels",
            b"""<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
            <Relationship Id="r" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" Target="/3D/3dmodel.model"/>
            </Relationships>""",
        )
        archive.writestr("3D/3dmodel.model", model_xml)
    stream.seek(0)
    with pytest.raises(ValueError, match="ball radius"):
        m3.read(stream)
