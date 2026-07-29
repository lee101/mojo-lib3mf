"""Core 3MF XML and OPC package reader/writer."""

from __future__ import annotations

import io
import math
import posixpath
import zipfile
from xml.etree import ElementTree as ET

import numpy as np

from .model import (
    BALL_MODES,
    BeamLattice,
    BuildItem,
    Component,
    ComponentsObject,
    Mesh,
    MeshObject,
    Model,
    transform_from_3mf,
    transform_to_3mf,
)

CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
BEAM = "http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02"
BALL = "http://schemas.microsoft.com/3dmanufacturing/beamlattice/balls/2020/07"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT = "http://schemas.openxmlformats.org/package/2006/content-types"
MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
MODEL_TYPE = "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"

ET.register_namespace("", CORE)
ET.register_namespace("b", BEAM)
ET.register_namespace("b2", BALL)


def _tag(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def _local(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _attribute(element: ET.Element, name: str, default=None):
    if name in element.attrib:
        return element.attrib[name]
    for key, value in element.attrib.items():
        if _local(key) == name:
            return value
    return default


def _float(value: float) -> str:
    return format(float(np.float32(value)), ".9g")


def _double(value: float) -> str:
    return format(float(value), ".17g")


def _validate_for_write(model: Model) -> None:
    ids = [obj.object_id for obj in model.objects]
    if len(ids) != len(set(ids)):
        raise ValueError("object IDs must be unique")
    known = set(ids)
    if any(item.object_id not in known for item in model.build_items):
        raise ValueError("build item refers to an unknown object")
    for obj in model.objects:
        if isinstance(obj, ComponentsObject) and any(
            component.object_id not in known for component in obj.components
        ):
            raise ValueError("component refers to an unknown object")
    for obj in model.objects:
        if not isinstance(obj, MeshObject):
            continue
        mesh = obj.mesh
        if not mesh.check_sanity():
            raise ValueError(f"mesh object {obj.object_id} failed lib3mf sanity checks")
        if not np.isfinite(mesh.vertices).all():
            raise ValueError(f"mesh object {obj.object_id} has non-finite coordinates")
        lattice = mesh.beam_lattice
        if lattice is None or not len(lattice.beams):
            continue
        if obj.object_type not in ("model", "solidsupport"):
            raise ValueError("beam lattices require a model or solidsupport object")
        if not 0 < lattice.default_radius <= 1e9:
            raise ValueError("beam default radius must be in (0, 1e9]")
        if not 0 < lattice.minimum_length <= 1e9:
            raise ValueError("beam minimum length must be in (0, 1e9]")
        if (
            not np.isfinite(lattice.radii).all()
            or np.any(lattice.radii < 0)
            or np.any(lattice.radii >= 1e9)
        ):
            raise ValueError("beam radii must be finite and in [0, 1e9)")
        if lattice.ball_mode != "none" and not 0 < lattice.default_ball_radius <= 1e9:
            raise ValueError("non-none ball mode requires a positive default radius")
        occupied = set(map(int, lattice.beams.ravel()))
        if any(int(index) not in occupied for index in lattice.ball_indices):
            raise ValueError("ball vertex must be occupied by a beam")
        if (
            not np.isfinite(lattice.ball_radii).all()
            or np.any(lattice.ball_radii < 0)
            or np.any(lattice.ball_radii >= 1e9)
        ):
            raise ValueError("ball radii must be finite and in [0, 1e9)")


def _model_xml(model: Model) -> bytes:
    _validate_for_write(model)
    has_beams = any(
        isinstance(obj, MeshObject)
        and obj.mesh.beam_lattice is not None
        and len(obj.mesh.beam_lattice.beams)
        for obj in model.objects
    )
    attributes = {"unit": model.unit, _tag("http://www.w3.org/XML/1998/namespace", "lang"): model.language}
    if has_beams:
        attributes["requiredextensions"] = "b"
    root = ET.Element(_tag(CORE, "model"), attributes)
    for name, value in model.metadata.items():
        metadata = ET.SubElement(root, _tag(CORE, "metadata"), {"name": name})
        metadata.text = value
    resources = ET.SubElement(root, _tag(CORE, "resources"))
    for obj in model.objects:
        obj_attrs = {"id": str(obj.object_id), "type": obj.object_type}
        if obj.name:
            obj_attrs["name"] = obj.name
        object_element = ET.SubElement(resources, _tag(CORE, "object"), obj_attrs)
        if isinstance(obj, ComponentsObject):
            components = ET.SubElement(object_element, _tag(CORE, "components"))
            for component in obj.components:
                attrs = {"objectid": str(component.object_id)}
                if component.transform is not None:
                    attrs["transform"] = transform_to_3mf(component.transform)
                ET.SubElement(components, _tag(CORE, "component"), attrs)
            continue
        mesh_element = ET.SubElement(object_element, _tag(CORE, "mesh"))
        vertices = ET.SubElement(mesh_element, _tag(CORE, "vertices"))
        for x, y, z in obj.mesh.vertices:
            ET.SubElement(
                vertices,
                _tag(CORE, "vertex"),
                {"x": _float(x), "y": _float(y), "z": _float(z)},
            )
        triangles = ET.SubElement(mesh_element, _tag(CORE, "triangles"))
        for a, b, c in obj.mesh.triangles:
            ET.SubElement(
                triangles,
                _tag(CORE, "triangle"),
                {"v1": str(int(a)), "v2": str(int(b)), "v3": str(int(c))},
            )
        lattice = obj.mesh.beam_lattice
        if lattice is not None and len(lattice.beams):
            attrs = {
                "radius": _float(lattice.default_radius),
                "minlength": _float(lattice.minimum_length),
                "cap": "sphere",
            }
            if lattice.ball_mode != "none":
                attrs[_tag(BALL, "ballmode")] = lattice.ball_mode
                attrs[_tag(BALL, "ballradius")] = _float(
                    lattice.default_ball_radius
                )
            lattice_element = ET.SubElement(
                mesh_element, _tag(BEAM, "beamlattice"), attrs
            )
            beams = ET.SubElement(lattice_element, _tag(BEAM, "beams"))
            for index, ((a, b), radii, caps) in enumerate(
                zip(lattice.beams, lattice.radii, lattice.caps, strict=True)
            ):
                beam_attrs = {"v1": str(int(a)), "v2": str(int(b))}
                r1, r2 = map(float, radii)
                if r1 != lattice.default_radius or r2 != r1:
                    beam_attrs["r1"] = _double(r1)
                if r2 != r1:
                    beam_attrs["r2"] = _double(r2)
                if caps[0] != "sphere":
                    beam_attrs["cap1"] = caps[0]
                if caps[1] != "sphere":
                    beam_attrs["cap2"] = caps[1]
                ET.SubElement(beams, _tag(BEAM, "beam"), beam_attrs)
            if len(lattice.ball_indices):
                balls = ET.SubElement(lattice_element, _tag(BALL, "balls"))
                for vertex, radius in zip(
                    lattice.ball_indices, lattice.ball_radii, strict=True
                ):
                    ball_attrs = {"vindex": str(int(vertex))}
                    if float(radius) != lattice.default_ball_radius:
                        ball_attrs["r"] = _double(radius)
                    ET.SubElement(balls, _tag(BALL, "ball"), ball_attrs)
    build = ET.SubElement(root, _tag(CORE, "build"))
    for item in model.build_items:
        attrs = {"objectid": str(item.object_id)}
        if item.transform is not None:
            attrs["transform"] = transform_to_3mf(item.transform)
        if item.part_number:
            attrs["partnumber"] = item.part_number
        ET.SubElement(build, _tag(CORE, "item"), attrs)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _content_types() -> bytes:
    root = ET.Element(_tag(CONTENT, "Types"))
    ET.SubElement(
        root,
        _tag(CONTENT, "Default"),
        {
            "Extension": "rels",
            "ContentType": "application/vnd.openxmlformats-package.relationships+xml",
        },
    )
    ET.SubElement(
        root,
        _tag(CONTENT, "Override"),
        {"PartName": "/3D/3dmodel.model", "ContentType": MODEL_TYPE},
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _relationships() -> bytes:
    root = ET.Element(_tag(REL, "Relationships"))
    ET.SubElement(
        root,
        _tag(REL, "Relationship"),
        {"Target": "/3D/3dmodel.model", "Id": "rel0", "Type": MODEL_REL},
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write(model: Model, destination, *, compression: int = 8) -> None:
    if not isinstance(model, Model):
        raise TypeError("model must be a Model")
    level = int(compression)
    if not 0 <= level <= 9:
        raise ValueError("compression must be between 0 and 9")
    method = zipfile.ZIP_STORED if level == 0 else zipfile.ZIP_DEFLATED
    model_xml = _model_xml(model)
    with zipfile.ZipFile(
        destination, "w", compression=method, compresslevel=level or None
    ) as archive:
        archive.writestr("[Content_Types].xml", _content_types())
        archive.writestr("_rels/.rels", _relationships())
        archive.writestr("3D/3dmodel.model", model_xml)


def _model_part(archive: zipfile.ZipFile) -> str:
    try:
        root = ET.fromstring(archive.read("_rels/.rels"))
    except KeyError as error:
        raise ValueError("3MF package has no root relationships part") from error
    for relationship in root:
        if relationship.get("Type") == MODEL_REL:
            target = relationship.get("Target", "").lstrip("/")
            normalized = posixpath.normpath(target)
            if (
                not normalized
                or normalized == "."
                or normalized.startswith("../")
                or normalized not in archive.namelist()
            ):
                raise ValueError("invalid 3MF model-part relationship target")
            return normalized
    raise ValueError("3MF package has no start-part relationship")


def _read_mesh(mesh_element: ET.Element) -> Mesh:
    vertices_element = mesh_element.find(_tag(CORE, "vertices"))
    triangles_element = mesh_element.find(_tag(CORE, "triangles"))
    if vertices_element is None or triangles_element is None:
        raise ValueError("mesh requires vertices and triangles")
    vertices = []
    for vertex in vertices_element:
        try:
            point = tuple(float(vertex.attrib[key]) for key in ("x", "y", "z"))
        except (KeyError, ValueError) as error:
            raise ValueError("invalid or missing vertex coordinate") from error
        if not all(math.isfinite(value) and abs(value) <= 1e9 for value in point):
            raise ValueError("invalid vertex coordinate")
        vertices.append(point)
    triangles = []
    for triangle in triangles_element:
        try:
            indices = tuple(int(triangle.attrib[key]) for key in ("v1", "v2", "v3"))
        except (KeyError, ValueError) as error:
            raise ValueError("invalid or missing triangle index") from error
        triangles.append(indices)
    lattice_element = mesh_element.find(_tag(BEAM, "beamlattice"))
    lattice = (
        _read_lattice(lattice_element, len(vertices))
        if lattice_element is not None
        else None
    )
    mesh = Mesh(vertices, triangles, lattice)
    if not mesh.check_sanity():
        raise ValueError("mesh failed lib3mf sanity checks")
    return mesh


def _cap(value: str | None) -> str:
    if value == "butt":
        return "butt"
    if value in ("hemisphere", "round"):
        return "hemisphere"
    return "sphere"


def _read_lattice(element: ET.Element, vertex_count: int) -> BeamLattice:
    try:
        default_radius = float(_attribute(element, "radius", "0.0001"))
        minimum_length = float(
            _attribute(element, "minlength", _attribute(element, "precision", "0.0001"))
        )
        default_ball_radius = float(_attribute(element, "ballradius", "0"))
    except ValueError as error:
        raise ValueError("invalid beam-lattice numeric attribute") from error
    if (
        not math.isfinite(default_radius)
        or not 0 < default_radius <= 1e9
        or not math.isfinite(minimum_length)
        or not 0 < minimum_length <= 1e9
    ):
        raise ValueError("invalid beam-lattice defaults")
    ball_mode = _attribute(element, "ballmode", "none")
    if ball_mode not in BALL_MODES:
        ball_mode = "none"
    if ball_mode != "none" and (
        not math.isfinite(default_ball_radius)
        or not 0 < default_ball_radius <= 1e9
    ):
        raise ValueError("invalid beam-lattice ball radius")
    default_cap = _cap(_attribute(element, "cap"))
    beams_element = element.find(_tag(BEAM, "beams"))
    beams: list[tuple[int, int]] = []
    radii: list[tuple[float, float]] = []
    caps: list[tuple[str, str]] = []
    if beams_element is not None:
        for beam in beams_element:
            try:
                a, b = int(beam.attrib["v1"]), int(beam.attrib["v2"])
                r1 = float(beam.get("r1", default_radius))
                r2 = float(beam.get("r2", r1))
            except (KeyError, ValueError) as error:
                raise ValueError("invalid beam") from error
            if (
                a < 0
                or b < 0
                or a >= vertex_count
                or b >= vertex_count
                or a == b
            ):
                raise ValueError("invalid beam vertex index")
            if (
                not math.isfinite(r1)
                or not math.isfinite(r2)
                or r1 < 0
                or r2 < 0
                or r1 >= 1e9
                or r2 >= 1e9
            ):
                raise ValueError("invalid beam radius")
            beams.append((a, b))
            radii.append((r1, r2))
            caps.append(
                (_cap(beam.get("cap1", default_cap)), _cap(beam.get("cap2", default_cap)))
            )
    ball_elements = element.find(_tag(BALL, "balls"))
    if ball_elements is None:
        ball_elements = element.find(_tag(BEAM, "balls"))
    ball_indices: list[int] = []
    ball_radii: list[float] = []
    if ball_elements is not None:
        occupied = {index for pair in beams for index in pair}
        for ball in ball_elements:
            try:
                vertex = int(ball.attrib["vindex"])
                radius = float(ball.get("r", default_ball_radius))
            except (KeyError, ValueError) as error:
                raise ValueError("invalid beam-lattice ball") from error
            if vertex not in occupied:
                raise ValueError("ball vertex must be occupied by a beam")
            if not math.isfinite(radius) or radius < 0 or radius >= 1e9:
                raise ValueError("invalid beam-lattice ball radius")
            ball_indices.append(vertex)
            ball_radii.append(radius)
    return BeamLattice(
        beams,
        radii,
        caps,
        minimum_length,
        default_radius,
        ball_mode,
        default_ball_radius,
        np.asarray(ball_indices, np.int32),
        np.asarray(ball_radii, np.float64),
    )


def _parse_model(data: bytes) -> Model:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise ValueError("invalid model XML") from error
    if _local(root.tag) != "model":
        raise ValueError("model part does not contain a model")
    unit = root.get("unit", "millimeter")
    language = root.get(_tag("http://www.w3.org/XML/1998/namespace", "lang"), "en-US")
    metadata = {
        element.get("name", ""): element.text or ""
        for element in root.findall(_tag(CORE, "metadata"))
    }
    resources = root.find(_tag(CORE, "resources"))
    if resources is None:
        raise ValueError("model has no resources")
    objects = []
    for element in resources.findall(_tag(CORE, "object")):
        try:
            object_id = int(element.attrib["id"])
        except (KeyError, ValueError) as error:
            raise ValueError("object has invalid or missing ID") from error
        common = {
            "object_id": object_id,
            "name": element.get("name", ""),
            "object_type": element.get("type", "model"),
        }
        mesh = element.find(_tag(CORE, "mesh"))
        components = element.find(_tag(CORE, "components"))
        if mesh is not None:
            objects.append(MeshObject(mesh=_read_mesh(mesh), **common))
        elif components is not None:
            values = []
            for component in components:
                try:
                    target = int(component.attrib["objectid"])
                except (KeyError, ValueError) as error:
                    raise ValueError("component has invalid object ID") from error
                text = component.get("transform")
                values.append(
                    Component(target, transform_from_3mf(text) if text else None)
                )
            objects.append(ComponentsObject(components=values, **common))
        else:
            raise ValueError("object is neither a mesh nor components object")
    build_element = root.find(_tag(CORE, "build"))
    build_items = []
    if build_element is not None:
        for item in build_element:
            try:
                target = int(item.attrib["objectid"])
            except (KeyError, ValueError) as error:
                raise ValueError("build item has invalid object ID") from error
            text = item.get("transform")
            build_items.append(
                BuildItem(
                    target,
                    transform_from_3mf(text) if text else None,
                    item.get("partnumber", ""),
                )
            )
    return Model(objects, build_items, unit, language, metadata)


def read(source) -> Model:
    if isinstance(source, (bytes, bytearray, memoryview)):
        source = io.BytesIO(source)
    try:
        with zipfile.ZipFile(source, "r") as archive:
            part = _model_part(archive)
            return _parse_model(archive.read(part))
    except zipfile.BadZipFile as error:
        raise ValueError("not a valid OPC ZIP package") from error
