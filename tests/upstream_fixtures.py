"""Upstream lib3mf 2.5.0 source excerpts, used as the parity reference.

The installed ``lib3mf`` 2.5.0 binding is the first reference for every test, but
its public API only goes through the writer/reader/properties layer. It does not
expose lib3mf's internal fixed-size math (``fnMATRIX3_multiply``,
``fnMATRIX3_apply``, ``fnMATRIX3_isIdentity``, ``fnMATRIX3_isplanar``,
``fnVEC3_calcTriangleNormal``), ``CMesh::checkSanity``, the oriented-manifold
routine, or the beam-length computation. For those kernels the reference is the
upstream source itself: the excerpts below are copied verbatim from
``github.com/3MFConsortium/lib3mf`` at tag ``v2.5.0``, so an assertion added over
them would cite upstream behaviour instead of the port's own expectations.

``test_upstream_fixture.py`` was meant to cross-check each excerpt against
``raw.githubusercontent.com`` and execute the arithmetic ones through
:func:`python_transcription`. That file does not exist and no test imports this
module, so nothing here is currently asserted; the live parity coverage is
``tests/test_kernels.py`` (NumPy transcriptions checked against the installed
binding where it exposes the operation) and ``tests/test_io.py`` (both read
directions against ``lib3mf``). See ``bench/verify.md``.
"""

from __future__ import annotations

import math
import re

RAW = "https://raw.githubusercontent.com/3MFConsortium/lib3mf/v2.5.0/"

#: path -> label -> (first line, last line, verbatim code, role)
#: role "executable": translated and run by test_upstream_fixture.py.
#: role "rule": a control-flow rule too tangled to translate; asserted
#:              structurally and exercised through the installed binding instead.
SOURCE: dict[str, dict[str, tuple[int, int, str, str]]] = {
    "Source/Common/Math/NMR_Matrix.cpp": {
        "fnMATRIX3_multiply": (
            348,
            361,
            """NMATRIX3 fnMATRIX3_multiply(_In_ const NMATRIX3 mMatrix1, _In_ const NMATRIX3 mMatrix2)
{
	NMATRIX3 mResult;
	nfInt32 i, j;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 4; j++)
			mResult.m_fields[i][j] = mMatrix1.m_fields[i][0] * mMatrix2.m_fields[0][j] +
			mMatrix1.m_fields[i][1] * mMatrix2.m_fields[1][j] +
			mMatrix1.m_fields[i][2] * mMatrix2.m_fields[2][j] +
			mMatrix1.m_fields[i][3] * mMatrix2.m_fields[3][j];

	return mResult;

}""",
            "executable",
        ),
        "fnMATRIX3_apply": (
            411,
            419,
            """NVEC3 fnMATRIX3_apply(_In_ const NMATRIX3 mMatrix, _In_ const NVEC3 vVector)
{
	NVEC3 vResult;
	nfInt32 i;
	for (i = 0; i < 3; i++)
		vResult.m_fields[i] = mMatrix.m_fields[i][0] * vVector.m_fields[0] + mMatrix.m_fields[i][1] * vVector.m_fields[1] + mMatrix.m_fields[i][2] * vVector.m_fields[2] + mMatrix.m_fields[i][3];

	return vResult;
}""",
            "executable",
        ),
        "fnMATRIX3_isIdentity": (
            269,
            282,
            """nfBool fnMATRIX3_isIdentity(_In_ const NMATRIX3 mMatrix)
{
	nfDouble dDelta = 0.0;
	nfInt32 i, j;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 4; j++) {
			nfDouble dId = (i == j) ? 1.0f : 0.0f;
			nfDouble dM = mMatrix.m_fields[i][j];
			nfDouble dX = dM - dId;
			dDelta += dX * dX;
		}

	return dDelta < NMR_MATRIX_IDENTITYTHRESHOLD;
}""",
            "executable",
        ),
        "fnMATRIX3_isplanar": (
            486,
            491,
            """nfBool fnMATRIX3_isplanar(_In_ const NMATRIX3 mMatrix)
{
	const double eps = 1e-7;
	return (fabs(mMatrix.m_fields[2][0]) < eps) && (fabs(mMatrix.m_fields[2][1]) < eps) && (fabs(mMatrix.m_fields[2][2] - 1 ) < eps)
			&& (fabs(mMatrix.m_fields[0][2]) < eps) && (fabs(mMatrix.m_fields[1][2]) < eps); 
}""",
            "executable",
        ),
    },
    "Source/Common/Math/NMR_Vector.cpp": {
        "fnVEC3_calcTriangleNormal": (
            335,
            346,
            """NVEC3 fnVEC3_calcTriangleNormal(_In_ const NVEC3 vVector1, _In_ const NVEC3 vVector2, _In_ const NVEC3 vVector3)
{
	NVEC3 vVectorU = fnVEC3_sub(vVector2, vVector1);
	NVEC3 vVectorV = fnVEC3_sub(vVector3, vVector1);
	NVEC3 vResult = fnVEC3_crossproduct(vVectorU, vVectorV);
	nfFloat fLength = fnVEC3_length(vResult);

	if (fLength > NMR_VECTOR_MINNORMALIZELENGTH)
		return fnVEC3_scale(vResult, 1.0f / fLength);
	else
		return fnVEC3_make(0.0f, 0.0f, 0.0f);
}""",
            "executable",
        ),
    },
    "Source/Common/Mesh/NMR_Mesh.cpp": {
        "CMesh::checkSanity": (
            450,
            506,
            """nfBool CMesh::checkSanity()
{
	nfUint32 nIdx, j;

	nfUint32 nNodeCount = getNodeCount();
	nfUint32 nFaceCount = getFaceCount();
	nfUint32 nBeamCount = getBeamCount();
	nfUint32 nBallCount = getBallCount();

	// max 2^31-1 billion Nodes/Faces
	if (nNodeCount > NMR_MESH_MAXNODECOUNT)
		return false;
	if (nFaceCount > NMR_MESH_MAXFACECOUNT)
		return false;
	if (nBeamCount > NMR_MESH_MAXBEAMCOUNT)
		return false;
	if (nBallCount > NMR_MESH_MAXBALLCOUNT)
		return false;

	for (nIdx = 0; nIdx < nNodeCount; nIdx++) {
		MESHNODE * node = getNode(nIdx);
		if (node->m_index != (nfInt32) nIdx)
			return false;
		for (j = 0; j < 3; j++)
			if (fabs(node->m_position.m_fields[j]) > NMR_MESH_MAXCOORDINATE)
				return false;
	}

	for (nIdx = 0; nIdx < nFaceCount; nIdx++) {
		MESHFACE * face = getFace(nIdx);
		for (j = 0; j < 3; j++)
			if ((face->m_nodeindices[j] < 0) || (((nfUint32)face->m_nodeindices[j]) >= nNodeCount))
				return false;

		if ((face->m_nodeindices[0] == face->m_nodeindices[1]) ||
			(face->m_nodeindices[0] == face->m_nodeindices[2]) ||
			(face->m_nodeindices[1] == face->m_nodeindices[2]))
			return false;
	}

	for (nIdx = 0; nIdx < nBeamCount; nIdx++) {
		MESHBEAM * beam = getBeam(nIdx);
		for (j = 0; j < 2; j++)
			if ((beam->m_nodeindices[j] < 0) || (((nfUint32)beam->m_nodeindices[j]) >= nNodeCount))
				return false;

		if (beam->m_nodeindices[0] == beam->m_nodeindices[1])
			return false;
	}

	for (nIdx = 0; nIdx < nBallCount; nIdx++) {
		MESHBALL* ball = getBall(nIdx);
		if ((ball->m_nodeindex < 0) || (((nfUint32)ball->m_nodeindex) >= nNodeCount))
			return false;
	}
	return true;
}""",
            "executable",
        ),
    },
    "Source/Model/Classes/NMR_ModelMeshObject.cpp": {
        "CModelMeshObject::isManifoldAndOriented": (
            126,
            210,
            """nfBool CModelMeshObject::isManifoldAndOriented()
{
	if (!m_pMesh->checkSanity())
		return false;

	nfUint32 nNodeCount = m_pMesh->getNodeCount();
	nfUint32 nFaceCount = m_pMesh->getFaceCount();

	if (nNodeCount < 3)
		return false;
	if (nFaceCount < 3)
		return false;

	CPairMatchingTree PairMatchingTree;
	nfUint32 nFaceIndex;
	nfInt32 nEdgeIndex;
	nfInt32 nEdgeCounter = 0;
	nfUint32 j;

	// Build Edge Tree
	for (nFaceIndex = 0; nFaceIndex < nFaceCount; nFaceIndex++) {
		MESHFACE * pFace = m_pMesh->getFace(nFaceIndex);

		for (j = 0; j < 3; j++) {
			nfInt32 nNodeIndex1 = pFace->m_nodeindices[j];
			nfInt32 nNodeIndex2 = pFace->m_nodeindices[(j + 1) % 3];

			if (!PairMatchingTree.checkMatch(nNodeIndex1, nNodeIndex2, nEdgeIndex)) {
				PairMatchingTree.addMatch(nNodeIndex1, nNodeIndex2, nEdgeCounter);
				nEdgeCounter++;

				if (nEdgeCounter > NMR_MESH_MAXEDGECOUNT)
					throw CNMRException(NMR_ERROR_INVALIDEDGEINDEX);
			}

		}

	}

	// Count positively and negatively oriented edges...
	std::vector<nfInt32> PositiveOrientations;
	std::vector<nfInt32> NegativeOrientations;

	// Create Edge arrays
	PositiveOrientations.resize(nEdgeCounter);
	NegativeOrientations.resize(nEdgeCounter);
	for (nEdgeIndex = 0; nEdgeIndex < nEdgeCounter; nEdgeIndex++) {
		PositiveOrientations[nEdgeIndex] = 0;
		NegativeOrientations[nEdgeIndex] = 0;
	}

	for (nFaceIndex = 0; nFaceIndex < nFaceCount; nFaceIndex++) {
		MESHFACE * pFace = m_pMesh->getFace(nFaceIndex);

		for (j = 0; j < 3; j++) {
			nfInt32 nNodeIndex1 = pFace->m_nodeindices[j];
			nfInt32 nNodeIndex2 = pFace->m_nodeindices[(j + 1) % 3];

			if (PairMatchingTree.checkMatch(nNodeIndex1, nNodeIndex2, nEdgeIndex)) {
				if ((nEdgeIndex < 0) || (nEdgeIndex >= nEdgeCounter))
					throw CNMRException(NMR_ERROR_INVALIDEDGEINDEX);

				if (nNodeIndex1 <= nNodeIndex2) {
					PositiveOrientations[nEdgeIndex]++;
				}
				else {
					NegativeOrientations[nEdgeIndex]++;
				}
			}
			else {
				throw CNMRException(NMR_ERROR_INVALIDMESHTOPOLOGY);
			}
		}
	}

	// Check Edge Orientations and determine manifoldness
	for (nEdgeIndex = 0; nEdgeIndex < nEdgeCounter; nEdgeIndex++) {
		if (PositiveOrientations[nEdgeIndex] != 1)
			return false;
		if (NegativeOrientations[nEdgeIndex] != 1)
			return false;
	}

	// Mesh is non-empty, oriented and manifold
	return true;
}""",
            "executable",
        ),
    },
    "Source/Model/Reader/BeamLattice1702/NMR_ModelReaderNode_BeamLattice1702_Beam.cpp": {
        "stringToCapMode": (
            46,
            54,
            """eModelBeamLatticeCapMode stringToCapMode(const nfChar * capModeStr) {
\t\tif (strcmp(capModeStr, XML_3MF_BEAMLATTICE_CAPMODE_BUTT) == 0)
\t\t\treturn eModelBeamLatticeCapMode::MODELBEAMLATTICECAPMODE_BUTT;
\t\tif ((strcmp(capModeStr, XML_3MF_BEAMLATTICE_CAPMODE_HEMISPHERE) == 0) || (strcmp(capModeStr, XML_3MF_BEAMLATTICE_CAPMODE_ROUND) == 0))
\t\t\treturn eModelBeamLatticeCapMode::MODELBEAMLATTICECAPMODE_HEMISPHERE;
\t\t// if (strcmp(capModeStr, XML_3MF_BEAMLATTICE_CAPMODE_SPHERE) == 0)
\t\t//\treturn eModelBeamLatticeCapMode::MODELBEAMLATTICECAPMODE_SPHERE;
\t\treturn eModelBeamLatticeCapMode::MODELBEAMLATTICECAPMODE_SPHERE;
}""",
            "executable",
        ),
        "Beam::OnAttribute r1/r2/cap1/cap2": (
            142,
            166,
            """\t\telse if (strcmp(pAttributeName, XML_3MF_ATTRIBUTE_BEAMLATTICE_R1) == 0) {
\t\t\tnfFloat fValue = fnStringToFloat(pAttributeValue);
\t\t\tif ((fValue >= 0) && (fValue < XML_3MF_MAXIMUMBEAMRADIUSVALUE)) {
\t\t\t\tm_dRadius1 = fValue;
\t\t\t\tm_bHasRadius1 = true;
\t\t\t}
\t\t}
\t\telse if (strcmp(pAttributeName, XML_3MF_ATTRIBUTE_BEAMLATTICE_R2) == 0) {
\t\t\tnfFloat fValue = fnStringToFloat(pAttributeValue);
\t\t\tif ((fValue >= 0) && (fValue < XML_3MF_MAXIMUMBEAMRADIUSVALUE)) {
\t\t\t\tm_dRadius2 = fValue;
\t\t\t\tm_bHasRadius2 = true;
\t\t\t}
\t\t}
\t\telse if (strcmp(pAttributeName, XML_3MF_ATTRIBUTE_BEAMLATTICE_CAP1) == 0) {
\t\t\tm_bHasCap1 = true;
\t\t\tm_eCapMode1 = stringToCapMode(pAttributeValue);
\t\t}
\t\telse if (strcmp(pAttributeName, XML_3MF_ATTRIBUTE_BEAMLATTICE_CAP2) == 0) {
\t\t\tm_bHasCap2 = true;
\t\t\tm_eCapMode2 = stringToCapMode(pAttributeValue);""",
            "rule",
        ),
    },
    "Source/Model/Reader/BeamLattice1702/NMR_ModelReaderNode_BeamLattice1702_Beams.cpp": {
        "Beams::OnNSChildElement": (
            93,
            115,
            """\t\t\t\t// Retrieve node indices
\t\t\t\tnfInt32 nIndex1, nIndex2;
\t\t\t\tpXMLNode->retrieveIndices(nIndex1, nIndex2, m_pMesh->getNodeCount());

\t\t\t\tMESHNODE* pNode1 = m_pMesh->getNode(nIndex1);
\t\t\t\tMESHNODE* pNode2 = m_pMesh->getNode(nIndex2);
\t\t\t\t
\t\t\t\tif (fnVEC3_length(fnVEC3_sub(pNode1->m_position, pNode2->m_position)) < m_pMesh->getBeamLatticeMinLength())
\t\t\t\t\tm_pWarnings->addException(CNMRException(NMR_ERROR_BEAMLATTICENODESTOOCLOSE), mrwInvalidMandatoryValue);

\t\t\t\tnfInt32 nTag;
\t\t\t\tnfBool bHasTag, bHasRadius1, bHasRadius2;
\t\t\t\tnfDouble dRadius1, dRadius2;
\t\t\t\tpXMLNode->retrieveTag(bHasTag, nTag);
\t\t\t\tpXMLNode->retrieveRadii(bHasRadius1, dRadius1, bHasRadius2, dRadius2);
\t\t\t\tnfDouble dDefaultValueForR2 = m_dDefaultRadius;
\t\t\t\tif (bHasRadius1) {
\t\t\t\t\tdDefaultValueForR2 = dRadius1;
\t\t\t\t}
\t\t\t\telse {
\t\t\t\t\tdRadius1 = m_dDefaultRadius;
\t\t\t\t}

\t\t\t\tif (!bHasRadius2) {
\t\t\t\t\tdRadius2 = dDefaultValueForR2;
\t\t\t\t}""",
            "executable",
        ),
    },
    "Source/Model/Reader/BeamLattice1702/NMR_ModelReaderNode_BeamLattice1702_BeamLattice.cpp": {
        "BeamLattice::OnAttribute": (
            107,
            159,
            """\t\tif (strcmp(pAttributeName, XML_3MF_ATTRIBUTE_BEAMLATTICE_RADIUS) == 0) {
\t\t\tnfDouble dValue = fnStringToDouble(pAttributeValue);
\t\t\tif ( std::isnan(dValue) || (dValue <= 0) || (dValue > XML_3MF_MAXIMUMCOORDINATEVALUE) )
\t\t\t\tthrow CNMRException(NMR_ERROR_BEAMLATTICEINVALIDATTRIBUTE);
\t\t\tm_dDefaultRadius = dValue;
\t\t}
\t\telse if ( (strcmp(pAttributeName, XML_3MF_ATTRIBUTE_BEAMLATTICE_MINLENGTH) == 0) ||
\t\t\t(strcmp(pAttributeName, XML_3MF_ATTRIBUTE_BEAMLATTICE_PRECISION) == 0) )\t// legacy
\t\t{
\t\t\tnfDouble dValue = fnStringToDouble(pAttributeValue);
\t\t\tif (std::isnan(dValue) || (dValue <= 0) || (dValue > XML_3MF_MAXIMUMCOORDINATEVALUE))
\t\t\t\tthrow CNMRException(NMR_ERROR_BEAMLATTICEINVALIDATTRIBUTE);
\t\t\tm_pMesh->setBeamLatticeMinLength(dValue);
\t\t}
\t\telse if (strcmp(pAttributeName, XML_3MF_ATTRIBUTE_BEAMLATTICE_BALLMODE) == 0) {
\t\t\tm_pMesh->setBeamLatticeBallMode(stringToBallMode(pAttributeValue));
\t\t}
\t\telse if (strcmp(pAttributeName, XML_3MF_ATTRIBUTE_BEAMLATTICE_BALLRADIUS) == 0) {
\t\t\tnfDouble dValue = fnStringToDouble(pAttributeValue);
\t\t\tif (std::isnan(dValue) || (dValue <= 0) || (dValue > XML_3MF_MAXIMUMCOORDINATEVALUE))
\t\t\t\tthrow CNMRException(NMR_ERROR_BEAMLATTICEINVALIDATTRIBUTE);
\t\t\tm_dDefaultBallRadius = dValue;
\t\t\tm_pMesh->setDefaultBallRadius(dValue);
\t\t}
\t\telse
\t\t\tm_pWarnings->addException(CNMRException(NMR_ERROR_BEAMLATTICEINVALIDATTRIBUTE), mrwInvalidOptionalValue);""",
            "rule",
        ),
    },
    "Source/Model/Reader/BeamLattice1702/NMR_ModelReaderNode_BeamLattice1702_Ball.cpp": {
        "stringToBallMode": (
            44,
            53,
            """\teModelBeamLatticeBallMode stringToBallMode(const nfChar* ballModeStr) 
\t{
\t\tif (strcmp(ballModeStr, XML_3MF_BEAMLATTICE_BALLMODE_MIXED) == 0)
\t\t\treturn eModelBeamLatticeBallMode::MODELBEAMLATTICEBALLMODE_MIXED;
\t\tif (strcmp(ballModeStr, XML_3MF_BEAMLATTICE_BALLMODE_ALL) == 0)
\t\t\treturn eModelBeamLatticeBallMode::MODELBEAMLATTICEBALLMODE_ALL;
\t\tif (strcmp(ballModeStr, XML_3MF_BEAMLATTICE_BALLMODE_NONE) == 0)
\t\t\treturn eModelBeamLatticeBallMode::MODELBEAMLATTICEBALLMODE_NONE;
\t\t// Invalid ballmode value
\t\tthrow CNMRException(NMR_ERROR_BEAMLATTICEINVALIDATTRIBUTE);
\t}""",
            "executable",
        ),
        "Ball::OnAttribute vindex/r": (
            104,
            123,
            """\t\tif (strcmp(pAttributeName, XML_3MF_ATTRIBUTE_BEAMLATTICE_BALL_VINDEX) == 0) {
\t\t\tnfInt32 nValue = fnStringToInt32(pAttributeValue);
\t\t\tif ((nValue >= 0) && (nValue < XML_3MF_MAXRESOURCEINDEX))
\t\t\t\tm_nIndex = nValue;
\t\t}
\t\telse if (strcmp(pAttributeName, XML_3MF_ATTRIBUTE_BEAMLATTICE_BALL_R) == 0) {
\t\t\tnfFloat fValue = fnStringToFloat(pAttributeValue);
\t\t\tif ((fValue >= 0) && (fValue < XML_3MF_MAXIMUMBALLRADIUSVALUE)) {
\t\t\t\tm_dRadius = fValue;
\t\t\t\tm_bHasRadius = true;
\t\t\t}
\t\t}
\t\telse
\t\t\tm_pWarnings->addException(CNMRException(NMR_ERROR_BEAMLATTICEINVALIDATTRIBUTE), mrwInvalidOptionalValue);""",
            "rule",
        ),
    },
    "Source/Model/Reader/BeamLattice1702/NMR_ModelReaderNode_BeamLattice1702_Balls.cpp": {
        "Balls::OnNSChildElement": (
            71,
            102,
            """\t\tif ((strcmp(pNameSpace, XML_3MF_NAMESPACE_BEAMLATTICESPEC) == 0) || (strcmp(pNameSpace, XML_3MF_NAMESPACE_BEAMLATTICEBALLSSPEC) == 0)) {
\t\t\tif (strcmp(pChildName, XML_3MF_ELEMENT_BALL) == 0) {
\t\t\t\t// Parse XML
\t\t\t\tPModelReaderNode_BeamLattice1702_Ball pXMLNode = std::make_shared<CModelReaderNode_BeamLattice1702_Ball>(m_pModel, m_pWarnings);
\t\t\t\tpXMLNode->parseXML(pXMLReader);

\t\t\t\t// Retrieve node index
\t\t\t\tnfInt32 nIndex;
\t\t\t\tpXMLNode->retrieveIndex(nIndex, m_pMesh->getNodeCount());

\t\t\t\tnfInt32 nTag;
\t\t\t\tnfBool bHasTag, bHasRadius;
\t\t\t\tnfDouble dRadius;
\t\t\t\tpXMLNode->retrieveTag(bHasTag, nTag);
\t\t\t\tpXMLNode->retrieveRadius(bHasRadius, dRadius);
\t\t\t\t
\t\t\t\tif (!bHasRadius) {
\t\t\t\t\tdRadius = m_dDefaultBallRadius;
\t\t\t\t}

\t\t\t\t// Create ball
\t\t\t\tMESHNODE * pNode = m_pMesh->getNode(nIndex);
\t\t\t\tm_pMesh->addBall(pNode, dRadius);
\t\t\t}""",
            "rule",
        ),
    },
}

#: Values the excerpts need but do not define inline.
CONSTANTS = {
    # Include/Common/Math/NMR_Matrix.h:41
    "NMR_MATRIX_IDENTITYTHRESHOLD": 1e-7,
    # Include/Common/Math/NMR_Geometry.h:39
    "NMR_VECTOR_MINNORMALIZELENGTH": 1.0e-10,
    # Include/Common/Mesh/NMR_Mesh.h: NMR_MESH_MAXCOORDINATE 1.0e9f
    "NMR_MESH_MAXCOORDINATE": 1e9,
    "NMR_MESH_MAXNODECOUNT": 2**31 - 1,
    "NMR_MESH_MAXFACECOUNT": 2**31 - 1,
    "NMR_MESH_MAXBEAMCOUNT": 2**31 - 1,
    "NMR_MESH_MAXBALLCOUNT": 2**31 - 1,
    # Include/Model/Classes/NMR_ModelConstants.h:520,527,528,522
    "XML_3MF_MAXIMUMCOORDINATEVALUE": 1e9,
    "XML_3MF_MAXIMUMBEAMRADIUSVALUE": 1e9,
    "XML_3MF_MAXIMUMBALLRADIUSVALUE": 1e9,
    "XML_3MF_MAXRESOURCEINDEX": 2**31 - 1,
}

#: Attribute and element names used by the excerpts (NMR_ModelConstants.h).
NAMES = {
    "radius": "XML_3MF_ATTRIBUTE_BEAMLATTICE_RADIUS",
    "minlength": "XML_3MF_ATTRIBUTE_BEAMLATTICE_MINLENGTH",
    "precision": "XML_3MF_ATTRIBUTE_BEAMLATTICE_PRECISION",
    "ballmode": "XML_3MF_ATTRIBUTE_BEAMLATTICE_BALLMODE",
    "ballradius": "XML_3MF_ATTRIBUTE_BEAMLATTICE_BALLRADIUS",
    "cap": "XML_3MF_ATTRIBUTE_BEAMLATTICE_CAPMODE",
    "r1": "XML_3MF_ATTRIBUTE_BEAMLATTICE_R1",
    "r2": "XML_3MF_ATTRIBUTE_BEAMLATTICE_R2",
    "cap1": "XML_3MF_ATTRIBUTE_BEAMLATTICE_CAP1",
    "cap2": "XML_3MF_ATTRIBUTE_BEAMLATTICE_CAP2",
    "vindex": "XML_3MF_ATTRIBUTE_BEAMLATTICE_BALL_VINDEX",
    "r": "XML_3MF_ATTRIBUTE_BEAMLATTICE_BALL_R",
}

#: Namespaces from NMR_ModelConstants.h:83 and friends.
NAMESPACES = {
    "beam": "http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02",
    "balls": "http://schemas.microsoft.com/3dmanufacturing/beamlattice/balls/2020/07",
    "core": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
}


def source_text(path: str, label: str) -> str:
    first, last, code, _role = SOURCE[path][label]
    return f"// {RAW}{path}#L{first}-L{last}\n" + code


def _split_fields(expression: str) -> list[str]:
    """Split a C++ right-hand side on ``+`` outside parentheses."""
    parts, depth, current = [], 0, ""
    for character in expression:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if character == "+" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += character
    parts.append(current)
    return [part.strip() for part in parts if part.strip()]


def _fields_map(code: str, target: str = "mResult") -> dict[str, list[str]]:
    """``m_fields[i][j]`` -> additive terms of its right-hand side."""
    joined = re.sub(r"\n\s+", " ", code)
    joined = re.sub(r"for\s*\([^)]*\)", " ", joined)
    joined = joined.replace("{", " { ").replace("}", " } ")
    mapping: dict[str, list[str]] = {}
    for statement in joined.split(";"):
        match = re.search(r"(?<![\w.])" + re.escape(target) + r"\.m_fields(\[[^\]]+\](?:\[[^\]]+\])?)\s*=\s*(.+)$", statement)
        if match:
            mapping[match.group(1)] = _split_fields(match.group(2).replace("}", " "))
    return mapping


class _CStruct:
    """Read-only stand-in for an upstream struct: ``X.m_fields[i][j]`` reads literally."""

    def __init__(self, fields):
        self.m_fields = [_Row(row) if isinstance(row, list) else row for row in fields]


class _Row:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)

    # C++ reads ``m_fields[j]`` as a scalar for NVEC3; a one-element row is that scalar.
    def __float__(self):
        if len(self._values) != 1:
            raise TypeError("row is not a single scalar")
        return float(self._values[0])

    def __mul__(self, other):
        return float(self) * float(other)

    __rmul__ = __mul__

    def __add__(self, other):
        return float(self) + float(other)

    __radd__ = __add__


def _struct(values) -> _CStruct:
    """Wrap a nested sequence so C++ ``m_fields`` access works unchanged."""
    return _CStruct([[float(value) for value in row] for row in values])


_TERM = re.compile(
    r"\[(?P<index>[A-Za-z_]\w*)\]"                      # [i] / [j] loop indices
    r"|(?P<var>[A-Za-z_]\w*)\.m_fields"                  # X.m_fields
    r"|(?P<number>(?<!\[)\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)[fF]?\b"  # 1.0f, 0.0f
)


def _arithmetic(expression: str, variables: dict[str, object], indices: dict[str, int] | None = None) -> float:
    """Evaluate one transcribed arithmetic term with C++ value semantics.

    Substitution happens on tokens, so ``m_fields[i][j]`` maps to ``[i][j]``
    and the loop index to its value exactly like the C++ would read them.
    """

    def replace(match: re.Match) -> str:
        if match.group("index") is not None:
            name = match.group("index")
            if indices is None or name not in indices:
                raise KeyError(f"unbound loop index {name!r} in {expression!r}")
            return f"[{indices[name]}]"
        if match.group("var") is not None:
            return f"{match.group('var')}.m_fields"
        return f"float({match.group('number')})"

    translated = _TERM.sub(replace, expression)
    return float(eval(translated, {"__builtins__": {"float": float}}, dict(variables)))


def fnMATRIX3_multiply(matrix1, matrix2):
    """``fnMATRIX3_multiply`` in ``nfDouble``, transcribed from the excerpt."""
    field_terms = _fields_map(
        SOURCE["Source/Common/Math/NMR_Matrix.cpp"]["fnMATRIX3_multiply"][2]
    )
    result = [[0.0] * 4 for _ in range(4)]
    first, second = _struct(matrix1), _struct(matrix2)
    for i in range(4):
        for j in range(4):
            terms = field_terms["[i][j]"]
            result[i][j] = sum(
                _arithmetic(term, {"mMatrix1": first, "mMatrix2": second}, {"i": i, "j": j})
                for term in terms
            )
    return result


def fnMATRIX3_apply(matrix, vector):
    """``fnMATRIX3_apply`` in ``nfFloat``, transcribed from the excerpt.

    ``vResult.m_fields[i]`` is ``nfFloat`` and each addend is computed in
    ``nfFloat``, so the reference rounds the same way the upstream C++ does.
    """
    import numpy as np

    field_terms = _fields_map(
        SOURCE["Source/Common/Math/NMR_Matrix.cpp"]["fnMATRIX3_apply"][2],
        target="vResult",
    )
    matrix = _struct(np.asarray(matrix, dtype=np.float32).tolist())
    vector = _CStruct([np.float32(value) for value in np.asarray(vector, dtype=np.float32)])
    result = np.empty(3, dtype=np.float32)
    for i in range(3):
        total = np.float32(0.0)
        for term in field_terms["[i]"]:
            total = np.float32(total + np.float32(_arithmetic(term, {
                "mMatrix": matrix,
                "vVector": vector,
            }, {"i": i})))
        result[i] = total
    return result


def fnMATRIX3_isIdentity(matrix, threshold: float | None = None) -> bool:
    """``fnMATRIX3_isIdentity`` transcribed; ``dDelta`` is ``nfDouble``."""
    limit = CONSTANTS["NMR_MATRIX_IDENTITYTHRESHOLD"] if threshold is None else threshold
    delta = 0.0
    for i in range(4):
        for j in range(4):
            identity = 1.0 if i == j else 0.0
            difference = float(matrix[i][j]) - identity
            delta += difference * difference
    return delta < limit


def fnMATRIX3_isplanar(matrix, epsilon: float = 1e-7) -> bool:
    """``fnMATRIX3_isplanar`` transcribed verbatim (row/column access as upstream)."""
    return (
        abs(float(matrix[2][0])) < epsilon
        and abs(float(matrix[2][1])) < epsilon
        and abs(float(matrix[2][2]) - 1.0) < epsilon
        and abs(float(matrix[0][2])) < epsilon
        and abs(float(matrix[1][2])) < epsilon
    )


def fnVEC3_calcTriangleNormal(vector1, vector2, vector3):
    """``fnVEC3_calcTriangleNormal`` transcribed, all arithmetic in ``nfFloat``."""
    import numpy as np

    limit = CONSTANTS["NMR_VECTOR_MINNORMALIZELENGTH"]
    u = np.asarray(vector2, dtype=np.float32) - np.asarray(vector1, dtype=np.float32)
    v = np.asarray(vector3, dtype=np.float32) - np.asarray(vector1, dtype=np.float32)
    cross = np.array(
        [
            np.float32(u[1] * v[2]) - np.float32(u[2] * v[1]),
            np.float32(u[2] * v[0]) - np.float32(u[0] * v[2]),
            np.float32(u[0] * v[1]) - np.float32(u[1] * v[0]),
        ],
        dtype=np.float32,
    )
    length = np.float32(np.sqrt(np.float32(cross @ cross)))
    if float(length) > limit:
        return (cross * np.float32(np.float32(1.0) / length)).astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def CHECKSANITY(node_count, face_count, beam_count, ball_count, nodes, faces, beams, balls) -> bool:
    """``CMesh::checkSanity`` transcribed.

    ``nodes`` are ``(x, y, z)`` tuples in storage order, ``faces`` and ``beams``
    are index tuples, ``balls`` are single indices. The per-entry index identity
    check (``node->m_index == nIdx``) always holds for a package-read mesh, and
    the count limits are compile-time constants, so both are reproduced as-is.
    """
    if node_count > CONSTANTS["NMR_MESH_MAXNODECOUNT"]:
        return False
    if face_count > CONSTANTS["NMR_MESH_MAXFACECOUNT"]:
        return False
    if beam_count > CONSTANTS["NMR_MESH_MAXBEAMCOUNT"]:
        return False
    if ball_count > CONSTANTS["NMR_MESH_MAXBALLCOUNT"]:
        return False
    for nIdx in range(node_count):
        for j in range(3):
            if abs(float(nodes[nIdx][j])) > CONSTANTS["NMR_MESH_MAXCOORDINATE"]:
                return False
    for nIdx in range(face_count):
        face = faces[nIdx]
        for j in range(3):
            index = int(face[j])
            if index < 0 or index >= node_count:
                return False
        if face[0] == face[1] or face[0] == face[2] or face[1] == face[2]:
            return False
    for nIdx in range(beam_count):
        beam = beams[nIdx]
        for j in range(2):
            index = int(beam[j])
            if index < 0 or index >= node_count:
                return False
        if beam[0] == beam[1]:
            return False
    for nIdx in range(ball_count):
        index = int(balls[nIdx])
        if index < 0 or index >= node_count:
            return False
    return True


def ISMANIFOLDANDORIENTED(node_count, face_count, nodes, faces, beams=(), balls=()) -> bool:
    """``CModelMeshObject::isManifoldAndOriented`` transcribed.

    ``CPairMatchingTree::checkMatch`` is a lookup keyed on the *ordered* edge
    pair (its hash/equality ignore orientation), so an edge present in either
    direction matches. Reproduced here with an explicit key.
    """
    if not CHECKSANITY(node_count, face_count, len(beams), len(balls), nodes, faces, beams, balls):
        return False
    if node_count < 3:
        return False
    if face_count < 3:
        return False

    edge_slots: dict[tuple[int, int], int] = {}
    edge_counter = 0
    for nFaceIndex in range(face_count):
        face = faces[nFaceIndex]
        for j in range(3):
            nNodeIndex1 = int(face[j])
            nNodeIndex2 = int(face[(j + 1) % 3])
            key = (min(nNodeIndex1, nNodeIndex2), max(nNodeIndex1, nNodeIndex2))
            if key not in edge_slots:
                edge_slots[key] = edge_counter
                edge_counter += 1

    positive = [0] * edge_counter
    negative = [0] * edge_counter
    for nFaceIndex in range(face_count):
        face = faces[nFaceIndex]
        for j in range(3):
            nNodeIndex1 = int(face[j])
            nNodeIndex2 = int(face[(j + 1) % 3])
            key = (min(nNodeIndex1, nNodeIndex2), max(nNodeIndex1, nNodeIndex2))
            nEdgeIndex = edge_slots[key]
            if nNodeIndex1 <= nNodeIndex2:
                positive[nEdgeIndex] += 1
            else:
                negative[nEdgeIndex] += 1

    for nEdgeIndex in range(edge_counter):
        if positive[nEdgeIndex] != 1:
            return False
        if negative[nEdgeIndex] != 1:
            return False
    return True


def stringToCapMode(cap_mode: str) -> str:
    """``stringToCapMode``: butt/hemisphere/round map; everything else is sphere."""
    if cap_mode == "butt":
        return "butt"
    if cap_mode in ("hemisphere", "round"):
        return "hemisphere"
    return "sphere"


def stringToBallMode(ball_mode: str) -> str:
    """``stringToBallMode``: the three names, otherwise a hard failure."""
    if ball_mode in ("mixed", "all", "none"):
        return ball_mode
    raise ValueError("NMR_ERROR_BEAMLATTICEINVALIDATTRIBUTE")


def beam_radius_attribute(value: float) -> float | None:
    """``Beam::OnAttribute`` r1/r2 acceptance: ``0 <= fValue < 1e9``, else ignored."""
    value = float(value)
    if not (math.isnan(value) or value < 0) and value < CONSTANTS["XML_3MF_MAXIMUMBEAMRADIUSVALUE"]:
        return value
    return None


def ball_radius_attribute(value: float) -> float | None:
    """``Ball::OnAttribute`` r acceptance: ``0 <= fValue < 1e9``, else ignored."""
    value = float(value)
    if not (math.isnan(value) or value < 0) and value < CONSTANTS["XML_3MF_MAXIMUMBALLRADIUSVALUE"]:
        return value
    return None


def beam_radii(default_radius, has_radius1, radius1, has_radius2, radius2):
    """The radius fallback block of ``Beams::OnNSChildElement``.

    r1 absent  -> default; r2 absent -> r1 when given, else the default.
    """
    default_for_two = default_radius
    if has_radius1:
        default_for_two = radius1
    else:
        radius1 = default_radius
    if not has_radius2:
        radius2 = default_for_two
    return radius1, radius2


def lattice_attribute_accepted(name: str, value: float) -> bool:
    """``BeamLattice1702_BeamLattice::OnAttribute`` acceptance rule."""
    if name in ("radius", "minlength", "precision", "ballradius"):
        return not (math.isnan(value) or value <= 0 or value > CONSTANTS["XML_3MF_MAXIMUMCOORDINATEVALUE"])
    raise ValueError(name)


def source_lines(path: str, label: str) -> tuple[int, int]:
    first, last, _code, _role = SOURCE[path][label]
    return first, last
