"""
Convert Tokgan JSON to Silhouette FXS format.

This converter transforms vector shape data from the Tokgan JSON format
to Silhouette FXS format, handling the coordinate system differences.

Usage:
    python tokgan_json_to_fxs.py input.json [output.fxs] [--log] [--layers]
                                 [--frame-offset N] [--no-hierarchy]

Arguments:
    input.json    Path to input JSON file (required)
    output.fxs    Path to output FXS file (optional, defaults to input.fxs)

Layouts:
    v3 JSON (camera + persons blocks, Rotobot Next 0.10.0+) -> the camera /
        person / body-part HIERARCHY by default:
            camera        root Layer; its corner pin is the plate camera (the
                          exact ECC homography), keyed every frame
              p<N>        Layer per person; position keyed at the pelvis
                <part>    Layer per body part; position + rotation from the
                          bone; its shape's path is in bone-local units
        so an artist can stabilise (disable the camera transform), move a
        whole person, or adjust one limb. --no-hierarchy for the old layouts.
    --layers   person -> region -> side grouping, transforms unused
    default    flat list of shapes (v2, or --no-hierarchy without --layers)

Coordinate systems (checked in Silhouette, 2026-10-08):
- Tokgan JSON: pixels, (0,0) at the TOP-left, Y down.
- Silhouette FXS shapes and layer positions: origin at the image centre,
  Y down, 1.0 = image height:
      normalized_x = ((pixel_x - w/2) / h) * pixel_aspect
      normalized_y = (pixel_y - h/2) / h
- Layer corner pin: fractions of the image, (0,0) top-left, (1,1)
  bottom-right. Layer rotation: degrees, positive clockwise on screen.
  A parent layer's transform (pin included) carries down to its children.
"""

import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict


# Global transform parameters (set at runtime)
WIDTH = 2160
HEIGHT = 4096
PIXEL_ASPECT = 1.0
# Silhouette frame = JSON frame + FRAME_OFFSET. JSON frames are 1-indexed. Default -1 maps
# them to a 0-indexed Silhouette timeline (historical behaviour); pass --frame-offset 0 when
# the footage in Silhouette is 1-indexed (EXR/mp4 sequences numbered from 1) so the shapes
# line up with the plate instead of landing one frame early.
FRAME_OFFSET = -1


def pixels_to_silhouette_normalized(x, y):
    """
    Convert pixel coordinates (Nuke-style: 0,0 at bottom-left) to
    Silhouette's normalized image coordinates.

    The coordinate systems differ:
    - Tokgan JSON: (0,0) at bottom-left, Y increases upward
    - Silhouette: (0,0) at center, Y increases upward, normalized to [-0.5, 0.5]

    Also note: The JSON frames start at 1, but Silhouette starts at 0,
    so we subtract 1 from frame numbers.

    Args:
        x: X coordinate in pixels
        y: Y coordinate in pixels

    Returns:
        Tuple of (x, y) in Silhouette's normalized coordinate system
    """
    tx = ((x - (WIDTH / 2)) / HEIGHT) * PIXEL_ASPECT
    ty = ((y - (HEIGHT / 2)) / HEIGHT)  # Normalized Y coordinate
    return tx, ty


def create_point_xml(x, y, left_x=None, left_y=None, right_x=None, right_y=None):
    """
    Create a Silhouette Point XML string.

    Args:
        x: X coordinate (normalized)
        y: Y coordinate (normalized)
        left_x, left_y: Left handle coordinates (for Bezier curves)
        right_x, right_y: Right handle coordinates (for Bezier curves)

    Returns:
        XML string for the Point element
    """
    point_str = f"({x:.6f},{y:.6f})"

    # Build attributes for handles
    attrs = []
    if left_x is not None and left_y is not None:
        attrs.append(f'left="({left_x:.6f},{left_y:.6f})"')
    if right_x is not None and right_y is not None:
        attrs.append(f'right="({right_x:.6f},{right_y:.6f})"')

    if attrs:
        return f'<Point {" ".join(attrs)}>{point_str}</Point>'
    return f"<Point>{point_str}</Point>"


def create_path_xml(points, closed=True):
    """
    Create a Path XML element with points.

    Args:
        points: List of point dicts with x, y, and optional handles
        closed: Whether the path is closed

    Returns:
        XML string for the Path element
    """
    path_lines = []
    path_lines.append('\t\t\t\t<Path closed="True" type="Bspline">')

    for pt in points:
        x, y = pixels_to_silhouette_normalized(pt["x"], pt["y"])

        # Get handle coordinates if present
        left_x = left_y = right_x = right_y = None
        if "left_x" in pt:
            left_x, left_y = pixels_to_silhouette_normalized(pt["left_x"], pt["left_y"])
        if "right_x" in pt:
            right_x, right_y = pixels_to_silhouette_normalized(pt["right_x"], pt["right_y"])

        point_xml = create_point_xml(x, y, left_x, left_y, right_x, right_y)
        path_lines.append("\t\t\t\t\t" + point_xml)

    path_lines.append("\t\t\t\t</Path>")
    return "\n".join(path_lines)


def create_key_xml(path_xml, frame):
    """
    Create a Key XML element containing a Path.

    Args:
        path_xml: Path XML string
        frame: Frame number

    Returns:
        XML string for the Key element
    """
    return f'\t\t\t\t<Key frame="{frame}" interp="linear">\n{path_xml}\n\t\t\t\t</Key>'


def create_property_xml(keys_xml):
    """
    Create the Path property element for a shape.

    Args:
        keys_xml: List of Key XML strings

    Returns:
        XML string for the Property element
    """
    # Keys XML already contains proper indentation with 8 tabs
    return '<Property id="path">\n' + "\n".join(keys_xml) + "\n\t\t\t\t</Property>"


def create_shape_xml_with_opacity(label, shape_id, keys_xml, opacity_xml, closed=True):
    """
    Create a Silhouette Shape element with animation keys and opacity property.

    Args:
        label: Shape label
        shape_id: Unique shape ID
        keys_xml: List of Key XML strings for path
        opacity_xml: XML string for opacity property
        closed: Whether the path is closed

    Returns:
        XML string for the Shape element
    """
    # Use tab indentation like original FXS files
    lines = [
        f'\t<Shape type="Shape" id="{shape_id}" label="{label}" selected="True" expanded="True" uuid="{generate_uuid(shape_id)}" shape_type="Bspline">',
        "\t\t<Properties>",
        "\t\t\t<Property id=\"note\" constant=\"True\"></Property>",
        "\t\t\t<Property id=\"path\">",
    ]

    # Each key needs to be properly indented with 8 tabs at the start of <Key>
    for key_xml in keys_xml:
        indented = key_xml.replace("\n", "\n\t\t\t\t")
        lines.append("\t\t\t\t" + indented)

    lines.append("\t\t\t</Property>")
    lines.append(opacity_xml)
    lines.append("\t\t</Properties>")
    lines.append("\t</Shape>")

    return "\n".join(lines)


def create_shape_xml(label, shape_id, keys_xml, closed=True):
    """
    Backward compatible wrapper for create_shape_xml_with_opacity.
    Creates a shape with default full opacity.
    """
    opacity_xml = '<Property id="opacity"><Value>100</Value></Property>'
    return create_shape_xml_with_opacity(label, shape_id, keys_xml, opacity_xml, closed)


def generate_uuid(seed_id):
    """Generate a pseudo-random UUID based on seed_id."""
    import hashlib
    hash_val = hashlib.md5(str(seed_id).encode()).hexdigest()
    return f"{hash_val[:8]}-{hash_val[8:12]}-{hash_val[12:16]}-{hash_val[16:20]}-{hash_val[20:32]}"


def create_opacity_xml(visibility_data):
    """
    Create opacity property XML with minimal keyframes based on visibility data.

    For hold interpolation, we need adjacent keyframes to define the visibility state:
    - Invisible (0) to Visible (100): frame X = 0, frame X+1 = 100
    - Visible (100) to Invisible (0): frame X = 100, frame X+1 = 0

    Args:
        visibility_data: Dict mapping JSON frame numbers to 0/1 visibility

    Returns:
        XML string for opacity property
    """
    if not visibility_data:
        return '<Property id="opacity"><Value>100</Value></Property>'

    # Get sorted JSON frame numbers (only frames where visibility is 1)
    sorted_json_frames = sorted(int(f) for f in visibility_data.keys() if visibility_data[f])

    if not sorted_json_frames:
        return '<Property id="opacity"><Value>0</Value></Property>'

    # Get sorted Silhouette frame numbers (JSON - 1)
    sorted_frames = [f + FRAME_OFFSET for f in sorted_json_frames]

    opacity_lines = ['<Property id="opacity">']
    keyframes = []

    # Build list of (frame, value) keyframes for hold interpolation
    # Track visible segments and generate transitions at boundaries

    # Find visible segments (consecutive frames)
    segments = []
    segment_start = sorted_frames[0]
    segment_end = sorted_frames[0]

    for i in range(1, len(sorted_frames)):
        if sorted_frames[i] == segment_end + 1:
            segment_end = sorted_frames[i]
        else:
            segments.append((segment_start, segment_end))
            segment_start = sorted_frames[i]
            segment_end = sorted_frames[i]
    segments.append((segment_start, segment_end))

    # Generate keyframes for each segment
    for seg_start, seg_end in segments:
        # Invisible→Visible transition at segment start. The "off" key must sit exactly
        # one frame BEFORE the "on" key so hold interpolation reads invisible up to the
        # boundary. Use a fixed -1 gap (NOT FRAME_OFFSET): seg_start is already
        # offset-adjusted, so reusing FRAME_OFFSET here double-applied it and, at
        # --frame-offset 0, collapsed both keys onto the same frame (dropping the "off"
        # and leaving shapes visible from frame 1). -1 == the old seg_start+FRAME_OFFSET
        # when FRAME_OFFSET==-1, so the default path is unchanged.
        keyframes.append((seg_start - 1, 0))
        keyframes.append((seg_start, 100))

        # Visible→Invisible transition after segment end
        keyframes.append((seg_end, 100))
        keyframes.append((seg_end + 1, 0))

    # Sort by frame and merge duplicates (keep last value per frame)
    keyframes.sort(key=lambda x: x[0])

    # Keep last keyframe per frame (hold interpolation uses the value at frame and after)
    seen_frames = {}
    for frame, value in keyframes:
        seen_frames[frame] = value

    unique_keyframes = sorted(seen_frames.items())

    # Add keyframes to XML
    for frame, value in unique_keyframes:
        opacity_lines.append(f'\t\t\t\t\t<Key frame="{frame}" interp="hold">{value}</Key>')

    opacity_lines.append('\t\t\t\t</Property>')

    return '\n'.join(opacity_lines)


def build_layer_hierarchy(data):
    """
    Build a hierarchical layer structure from object names.

    Object names follow format: person:region:side:part
    This creates nested layers: person -> region -> side -> part

    Args:
        data: Parsed JSON data with objects

    Returns:
        Nested dict representing the layer hierarchy with shapes at leaves
        Structure: {person: {region: {side: [(obj_name, obj, part), ...]}}}
    """
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for obj_name, obj in data.get("objects", {}).items():
        parts = obj_name.split(":")
        while len(parts) < 4:
            parts.append("unknown")
        person, region, side, part = parts[:4]

        hierarchy[person][region][side].append((obj_name, obj, part))

    return hierarchy


def get_layer_label(person, region=None, side=None, part=None):
    """
    Generate layer label based on hierarchy level.

    Args:
        person: Person name
        region: Region name (optional)
        side: Side name (optional)
        part: Part name (optional)

    Returns:
        Layer label string
    """
    if part:
        # Part level - for hand fingers, use finger name
        return f"{person}_{part}"
    elif side:
        return f"{person}_{region}_{side}"
    elif region:
        return f"{region}"
    else:
        return person


def create_object_xml(obj_type, label, obj_id, uuid, content_xml, indent="\t"):
    """
    Create an Object XML element (for Layer or Shape) inside <Property id="objects">.

    Args:
        obj_type: "Layer" or "Shape"
        label: Object label
        obj_id: Unique object ID
        uuid: Object UUID
        content_xml: Properties content for this object
        indent: Indentation prefix

    Returns:
        XML string for the Object element
    """
    lines = [
        f'{indent}<Object type="{obj_type}" id="{obj_id}" label="{label}" uuid="{uuid}">',
    ]
    # Add default properties
    default_props = [
        '<Property id="note" constant="True"></Property>',
        '<Property id="color" constant="True"><Value>(1.000000,1.000000,1.000000)</Value></Property>',
        '<Property id="transform" constant="True"><Value></Value></Property>',
        '<Property id="transform.anchor" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
        '<Property id="transform.position" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
        '<Property id="transform.scale" constant="True" gang="True"><Value>(1.000000000,1.000000000,1.000000000)</Value></Property>',
        '<Property id="transform.rotate" constant="True"><Value>0</Value></Property>',
        '<Property id="transform.pin" constant="True"><Value></Value></Property>',
        '<Property id="transform.pin_ul" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
        '<Property id="transform.pin_ur" constant="True"><Value>(1.000000000,0.000000000)</Value></Property>',
        '<Property id="transform.pin_lr" constant="True"><Value>(1.000000000,1.000000000)</Value></Property>',
        '<Property id="transform.pin_ll" constant="True"><Value>(0.000000000,1.000000000)</Value></Property>',
        '<Property id="transform.surface" constant="True"><Value></Value></Property>',
        '<Property id="transform.surface_ul" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
        '<Property id="transform.surface_ur" constant="True"><Value>(1.000000000,0.000000000)</Value></Property>',
        '<Property id="transform.surface_lr" constant="True"><Value>(1.000000000,1.000000000)</Value></Property>',
        '<Property id="transform.surface_ll" constant="True"><Value>(0.000000000,1.000000000)</Value></Property>',
        '<Property id="transform.matrix" constant="True"><Value></Value></Property>',
        '<Property id="stereoOffset"><Value>(0.000000000,0.000000000,0.000000000)</Value></Property>',
        '<Property id="trackSource"><Value>0</Value></Property>',
    ]
    lines.extend([f'{indent}\t{p}' for p in default_props])
    lines.append(f'{indent}\t<Property id="objects" constant="True">')
    lines.append(content_xml)
    lines.append(f'{indent}\t</Property>')
    lines.append(f'{indent}</Object>')
    return "\n".join(lines)


def create_layer_xml(label, contents, indent="\t", obj_id=0):
    """
    Create a Silhouette Layer XML element containing shapes or nested layers.

    The model uses <Object type="Layer"> inside <Property id="objects"> constant="True".

    Args:
        label: Layer label
        contents: List of shape or layer XML strings for the objects property
        indent: Indentation prefix
        obj_id: Unique object ID

    Returns:
        XML string for the Layer element (empty string if contents is empty)
    """
    if not contents:
        return ""

    # Content for objects property
    objects_content = "\n".join(contents)

    uuid = generate_uuid(f"layer_{label}_{obj_id}")
    return create_object_xml("Layer", label, obj_id, uuid, objects_content, indent)


def create_shape_object_xml(label, shape_xml, obj_id):
    """
    Create a Shape Object XML element wrapped in <Object type="Shape">.

    Args:
        label: Shape label
        shape_xml: The full shape XML content
        obj_id: Unique object ID

    Returns:
        XML string for the Shape Object element
    """
    uuid = generate_uuid(f"shape_{label}_{obj_id}")

    lines = [
        f'\t\t\t\t\t<Object type="Shape" id="{obj_id}" label="{label}" uuid="{uuid}" shape_type="Bspline">',
    ]
    # Extract the properties from the shape_xml
    # Shape XML has <Properties>...</Properties> section
    import re
    props_match = re.search(r'<Properties>(.*?)</Properties>', shape_xml, re.DOTALL)
    if props_match:
        properties_content = props_match.group(1)
        # Add properties
        lines.append(f'\t\t\t\t\t\t<Properties>')
        # Indent the properties
        for line in properties_content.split('\n'):
            lines.append(f'\t\t\t\t\t\t\t{line}')
        lines.append(f'\t\t\t\t\t\t</Properties>')
    lines.append(f'\t\t\t\t\t</Object>')
    return "\n".join(lines)


def create_layer_xml_element(label, properties_content, obj_id, obj_type="Layer", expanded=True, uuid=None):
    """
    Create a Silhouette Layer or Shape XML element.

    Args:
        label: Layer/Shape label
        properties_content: Content of the <Properties> element
        obj_id: Unique object ID
        obj_type: "Layer" or "Shape"
        expanded: Whether expanded="True" attribute is added
        uuid: Object UUID (auto-generated if None)

    Returns:
        XML string for the Layer/Shape element
    """
    if uuid is None:
        uuid = generate_uuid(f"{obj_type}_{label}_{obj_id}")

    attrs = [f'type="{obj_type}"', f'id="{obj_id}"', f'label="{label}"', f'uuid="{uuid}"']
    if expanded:
        attrs.append('expanded="True"')

    lines = [
        f'\t<{obj_type} {" ".join(attrs)}>',
        '\t\t<Properties>',
    ]
    lines.append(properties_content)
    lines.append('\t\t</Properties>')
    lines.append(f'\t</{obj_type}>')

    return "\n".join(lines)


def create_layer_object_xml(label, content_objects, obj_id):
    """
    Create a Layer Object XML element wrapped in <Object type="Layer">.

    Args:
        label: Layer label
        content_objects: List of object XML strings (shapes or nested layers)
        obj_id: Unique object ID

    Returns:
        XML string for the Layer Object element
    """
    uuid = generate_uuid(f"layer_{label}_{obj_id}")

    lines = [
        f'\t\t\t\t\t<Object type="Layer" id="{obj_id}" label="{label}" expanded="True" uuid="{uuid}">',
    ]
    # Add default properties
    default_props = [
        '<Property id="note" constant="True"></Property>',
        '<Property id="color" constant="True"><Value>(1.000000,1.000000,1.000000)</Value></Property>',
        '<Property id="transform" constant="True"><Value></Value></Property>',
        '<Property id="transform.anchor" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
        '<Property id="transform.position" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
        '<Property id="transform.scale" constant="True" gang="True"><Value>(1.000000000,1.000000000,1.000000000)</Value></Property>',
        '<Property id="transform.rotate" constant="True"><Value>0</Value></Property>',
        '<Property id="transform.pin" constant="True"><Value></Value></Property>',
        '<Property id="transform.pin_ul" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
        '<Property id="transform.pin_ur" constant="True"><Value>(1.000000000,0.000000000)</Value></Property>',
        '<Property id="transform.pin_lr" constant="True"><Value>(1.000000000,1.000000000)</Value></Property>',
        '<Property id="transform.pin_ll" constant="True"><Value>(0.000000000,1.000000000)</Value></Property>',
        '<Property id="transform.surface" constant="True"><Value></Value></Property>',
        '<Property id="transform.surface_ul" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
        '<Property id="transform.surface_ur" constant="True"><Value>(1.000000000,0.000000000)</Value></Property>',
        '<Property id="transform.surface_lr" constant="True"><Value>(1.000000000,1.000000000)</Value></Property>',
        '<Property id="transform.surface_ll" constant="True"><Value>(0.000000000,1.000000000)</Value></Property>',
        '<Property id="transform.matrix" constant="True"><Value></Value></Property>',
        '<Property id="stereoOffset"><Value>(0.000000000,0.000000000,0.000000000)</Value></Property>',
        '<Property id="trackSource"><Value>0</Value></Property>',
    ]
    lines.extend([f'\t\t\t\t\t\t{p}' for p in default_props])

    # Add objects property with content
    lines.append(f'\t\t\t\t\t\t<Property id="objects" expanded="True" constant="True">')
    for obj_xml in content_objects:
        lines.append(obj_xml)
    lines.append(f'\t\t\t\t\t\t</Property>')

    lines.append(f'\t\t\t\t\t</Object>')
    return "\n".join(lines)


def create_silhouette_xml(data, log=False, use_layers=False):
    """
    Create the complete Silhouette XML structure.

    Args:
        data: Parsed JSON data
        log: If True, log execution details for each shape
        use_layers: If True, organize shapes into hierarchical layers

    Returns:
        Tuple of (formatted XML string, shape count, frame count)
    """
    global WIDTH, HEIGHT, PIXEL_ASPECT
    WIDTH, HEIGHT = data.get("resolution", [2160, 4096])
    PIXEL_ASPECT = data.get("pixelAspect", 1.0)

    # Determine work range from all objects and frames
    all_frames = set()
    for obj_name, obj in data.get("objects", {}).items():
        frames = obj.get("frames", {})
        all_frames.update(int(f) for f in frames.keys())

    start_frame = min(all_frames) if all_frames else 1
    end_frame = max(all_frames) if all_frames else 48

    # Build shape XML
    shape_elements = []
    shape_id = 0
    shape_count = 0

    # If using layers, build hierarchy first
    if use_layers:
        hierarchy = build_layer_hierarchy(data)

    # Build shape XML (used both with and without layers)
    shapes_by_name = {}
    for obj_name, obj in data.get("objects", {}).items():
        points_list = obj.get("frames", {})
        closed = obj.get("closed", True)
        visibility_data = obj.get("visibility", {})

        # Create label from object name (convert person:region:side:part format)
        parts = obj_name.split(":")
        label = "_".join(parts) + "Shape"

        # Build keys for each frame
        keys_xml = []
        sorted_frames = sorted(points_list.keys(), key=int)

        for frame_str in sorted_frames:
            # Silhouette frames start at 0, JSON frames start at 1
            frame_num = int(frame_str) + FRAME_OFFSET
            frame_data = points_list[frame_str]
            pts = frame_data.get("points", [])

            if pts:
                path_xml = create_path_xml(pts, closed)
                key_xml = create_key_xml(path_xml, frame_num)
                keys_xml.append(key_xml)

        if keys_xml:
            opacity_xml = create_opacity_xml(visibility_data)

            shape_xml = create_shape_xml_with_opacity(label, shape_id, keys_xml, opacity_xml, closed)
            shape_elements.append(shape_xml)
            shapes_by_name[obj_name] = shape_xml
            shape_id += 1
            shape_count += 1

            if log:
                print(f"[Shape {shape_count}] {label}: {len(sorted_frames)} frames, {len(pts)} points per frame     ", end="\r", flush=True)



    # Calculate total frames across all shapes
    total_frames = len(set(int(f) for obj in data.get("objects", {}).values() for f in obj.get("frames", {}).keys()))

    # Build layer XML if requested
    if use_layers:
        # Object ID counter for assigning unique IDs
        obj_id_counter = [0]

        def get_next_obj_id():
            obj_id = obj_id_counter[0]
            obj_id_counter[0] += 1
            return obj_id

        # Build a flat list of all shapes with their hierarchy info
        shape_hierarchy = []  # [(obj_name, obj, side_label, region_label), ...]
        for person, regions in sorted(hierarchy.items()):
            for region, sides in sorted(regions.items()):
                for side, items in sorted(sides.items()):
                    for obj_name, obj, part in items:
                        if obj_name in shapes_by_name:
                            side_label = f"{person}_{region}_{side}"
                            region_label = f"{person}_{region}"
                            shape_hierarchy.append((obj_name, obj, part, side_label, region_label))

        # Build layer structure: person -> region -> side -> objects
        def build_shape_object(label, shape_xml, obj_id):
            """Create Object wrapper for a shape."""
            return create_shape_object_xml(label, shape_xml, obj_id)

        def build_layer_object(label, content_objects, obj_id):
            """Build a layer Object with objects property containing nested objects."""
            uuid = generate_uuid(f"layer_{label}_{obj_id}")
            lines = [
                f'\t\t\t\t\t<Object type="Layer" id="{obj_id}" label="{label}" expanded="True" uuid="{uuid}">',
                '\t\t\t\t\t\t<Properties>',
            ]
            default_props = [
                '<Property id="note" constant="True"></Property>',
                '<Property id="color" constant="True"><Value>(1.000000,1.000000,1.000000)</Value></Property>',
                '<Property id="transform" constant="True"><Value></Value></Property>',
                '<Property id="transform.anchor" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
                '<Property id="transform.position" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
                '<Property id="transform.scale" constant="True" gang="True"><Value>(1.000000000,1.000000000,1.000000000)</Value></Property>',
                '<Property id="transform.rotate" constant="True"><Value>0</Value></Property>',
                '<Property id="transform.pin" constant="True"><Value></Value></Property>',
                '<Property id="transform.pin_ul" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
                '<Property id="transform.pin_ur" constant="True"><Value>(1.000000000,0.000000000)</Value></Property>',
                '<Property id="transform.pin_lr" constant="True"><Value>(1.000000000,1.000000000)</Value></Property>',
                '<Property id="transform.pin_ll" constant="True"><Value>(0.000000000,1.000000000)</Value></Property>',
                '<Property id="transform.surface" constant="True"><Value></Value></Property>',
                '<Property id="transform.surface_ul" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
                '<Property id="transform.surface_ur" constant="True"><Value>(1.000000000,0.000000000)</Value></Property>',
                '<Property id="transform.surface_lr" constant="True"><Value>(1.000000000,1.000000000)</Value></Property>',
                '<Property id="transform.surface_ll" constant="True"><Value>(0.000000000,1.000000000)</Value></Property>',
                '<Property id="transform.matrix" constant="True"><Value></Value></Property>',
                '<Property id="stereoOffset"><Value>(0.000000000,0.000000000,0.000000000)</Value></Property>',
                '<Property id="trackSource"><Value>0</Value></Property>',
            ]
            lines.extend([f'\t\t\t\t\t\t\t{p}' for p in default_props])
            lines.append(f'\t\t\t\t\t\t\t<Property id="objects" expanded="True" constant="True">')
            for obj_xml in content_objects:
                lines.append(obj_xml)
            lines.append(f'\t\t\t\t\t\t\t</Property>')
            lines.append('\t\t\t\t\t\t</Properties>')
            lines.append(f'\t\t\t\t\t</Object>')
            return "\n".join(lines)

        def build_nested_layer_xml(label, content_objects, obj_id):
            """Build a layer Object with objects property containing nested objects."""
            uuid = generate_uuid(f"layer_{label}_{obj_id}")
            lines = [
                f'\t\t\t\t\t<Object type="Layer" id="{obj_id}" label="{label}" expanded="True" uuid="{uuid}">',
                '\t\t\t\t\t\t<Properties>',
            ]
            default_props = [
                '<Property id="note" constant="True"></Property>',
                '<Property id="color" constant="True"><Value>(1.000000,1.000000,1.000000)</Value></Property>',
                '<Property id="transform" constant="True"><Value></Value></Property>',
                '<Property id="transform.anchor" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
                '<Property id="transform.position" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
                '<Property id="transform.scale" constant="True" gang="True"><Value>(1.000000000,1.000000000,1.000000000)</Value></Property>',
                '<Property id="transform.rotate" constant="True"><Value>0</Value></Property>',
                '<Property id="transform.pin" constant="True"><Value></Value></Property>',
                '<Property id="transform.pin_ul" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
                '<Property id="transform.pin_ur" constant="True"><Value>(1.000000000,0.000000000)</Value></Property>',
                '<Property id="transform.pin_lr" constant="True"><Value>(1.000000000,1.000000000)</Value></Property>',
                '<Property id="transform.pin_ll" constant="True"><Value>(0.000000000,1.000000000)</Value></Property>',
                '<Property id="transform.surface" constant="True"><Value></Value></Property>',
                '<Property id="transform.surface_ul" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
                '<Property id="transform.surface_ur" constant="True"><Value>(1.000000000,0.000000000)</Value></Property>',
                '<Property id="transform.surface_lr" constant="True"><Value>(1.000000000,1.000000000)</Value></Property>',
                '<Property id="transform.surface_ll" constant="True"><Value>(0.000000000,1.000000000)</Value></Property>',
                '<Property id="transform.matrix" constant="True"><Value></Value></Property>',
                '<Property id="stereoOffset"><Value>(0.000000000,0.000000000,0.000000000)</Value></Property>',
                '<Property id="trackSource"><Value>0</Value></Property>',
            ]
            lines.extend([f'\t\t\t\t\t\t\t{p}' for p in default_props])
            lines.append(f'\t\t\t\t\t\t\t<Property id="objects" expanded="True" constant="True">')
            for obj_xml in content_objects:
                lines.append(obj_xml)
            lines.append(f'\t\t\t\t\t\t\t</Property>')
            lines.append('\t\t\t\t\t\t</Properties>')
            lines.append(f'\t\t\t\t\t</Object>')
            return "\n".join(lines)

        def build_side_layer_xml(side_label, shapes_for_side):
            """Build side layer containing shapes (as Objects)."""
            shape_objects = []
            for obj_name, obj, part in shapes_for_side:
                shape_xml = shapes_by_name[obj_name]
                label = obj_name.replace(":", "_") + "Shape"
                shape_obj_xml = build_shape_object(label, shape_xml, get_next_obj_id())
                shape_objects.append(shape_obj_xml)
            return build_nested_layer_xml(side_label, shape_objects, get_next_obj_id())

        def build_region_layer_xml(region_label, side_layers):
            """Build region layer containing side layers (as Objects)."""
            return build_nested_layer_xml(region_label, side_layers, get_next_obj_id())

        def build_person_layer_xml(person_label, region_layers):
            """Build person layer containing region layers (as Objects).
            Root person layer uses <Layer> element directly under Silhouette."""
            uuid = generate_uuid(f"layer_{person_label}_root")
            lines = [
                f'\t<Layer type="Layer" id="{get_next_obj_id()}" label="{person_label}" expanded="True" uuid="{uuid}">',
                '\t\t<Properties>',
                '\t\t\t<Property id="note" constant="True"></Property>',
                '\t\t\t<Property id="color" constant="True"><Value>(1.000000,1.000000,1.000000)</Value></Property>',
                '\t\t\t<Property id="transform" constant="True"><Value></Value></Property>',
                '\t\t\t<Property id="transform.anchor" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
                '\t\t\t<Property id="transform.position" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
                '\t\t\t<Property id="transform.scale" constant="True" gang="True"><Value>(1.000000000,1.000000000,1.000000000)</Value></Property>',
                '\t\t\t<Property id="transform.rotate" constant="True"><Value>0</Value></Property>',
                '\t\t\t<Property id="transform.pin" constant="True"><Value></Value></Property>',
                '\t\t\t<Property id="transform.pin_ul" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
                '\t\t\t<Property id="transform.pin_ur" constant="True"><Value>(1.000000000,0.000000000)</Value></Property>',
                '\t\t\t<Property id="transform.pin_lr" constant="True"><Value>(1.000000000,1.000000000)</Value></Property>',
                '\t\t\t<Property id="transform.pin_ll" constant="True"><Value>(0.000000000,1.000000000)</Value></Property>',
                '\t\t\t<Property id="transform.surface" constant="True"><Value></Value></Property>',
                '\t\t\t<Property id="transform.surface_ul" constant="True"><Value>(0.000000000,0.000000000)</Value></Property>',
                '\t\t\t<Property id="transform.surface_ur" constant="True"><Value>(1.000000000,0.000000000)</Value></Property>',
                '\t\t\t<Property id="transform.surface_lr" constant="True"><Value>(1.000000000,1.000000000)</Value></Property>',
                '\t\t\t<Property id="transform.surface_ll" constant="True"><Value>(0.000000000,1.000000000)</Value></Property>',
                '\t\t\t<Property id="transform.matrix" constant="True"><Value></Value></Property>',
                '\t\t\t<Property id="stereoOffset"><Value>(0.000000000,0.000000000,0.000000000)</Value></Property>',
                '\t\t\t<Property id="trackSource"><Value>0</Value></Property>',
                '\t\t\t<Property id="objects" expanded="True" constant="True">',
            ]
            for layer_xml in region_layers:
                lines.append(layer_xml)
            lines.append('\t\t\t</Property>')
            lines.append('\t\t</Properties>')
            lines.append('\t</Layer>')
            return "\n".join(lines)

        # Group shapes by person, region, side
        person_data = {}
        for person, regions in sorted(hierarchy.items()):
            if person not in person_data:
                person_data[person] = {}
            for region, sides in sorted(regions.items()):
                if region not in person_data[person]:
                    person_data[person][region] = {}
                for side, items in sorted(sides.items()):
                    if side not in person_data[person][region]:
                        person_data[person][region][side] = []
                    for obj_name, obj, part in items:
                        if obj_name in shapes_by_name:
                            person_data[person][region][side].append((obj_name, obj, part))

        # Build the layer structure
        person_layer_objects = []
        for person in sorted(person_data.keys()):
            regions_dict = person_data[person]
            region_layer_objects = []

            for region in sorted(regions_dict.keys()):
                sides_dict = regions_dict[region]
                side_layer_objects = []

                for side in sorted(sides_dict.keys()):
                    items = sides_dict[side]
                    # Build side layer with shapes
                    side_label = f"{person}_{region}_{side}"
                    side_obj = build_side_layer_xml(side_label, items)
                    side_layer_objects.append(side_obj)

                # Build region layer with side layers
                region_label = f"{person}_{region}"
                region_obj = build_region_layer_xml(region_label, side_layer_objects)
                region_layer_objects.append(region_obj)

            # Build person layer with region layers
            person_label = person
            person_obj = build_person_layer_xml(person_label, region_layer_objects)
            person_layer_objects.append(person_obj)

        # Build root XML with layers
        xml_lines = [
            f'<!-- Silhouette Shape File -->',
            f'<Silhouette width="{WIDTH}" height="{HEIGHT}" pixelAspect="1" workRangeStart="{start_frame+FRAME_OFFSET}" workRangeEnd="{end_frame+FRAME_OFFSET}" sessionStartFrame="1">',
        ]

        for layer_xml in person_layer_objects:
            xml_lines.append(layer_xml)

        xml_lines.append("</Silhouette>")
    else:
        # Build root XML without layers (shapes directly under Silhouette)
        xml_lines = [
            f'<!-- Silhouette Shape File -->',
            f'<Silhouette width="{WIDTH}" height="{HEIGHT}" pixelAspect="1" workRangeStart="{start_frame+FRAME_OFFSET}" workRangeEnd="{end_frame+FRAME_OFFSET}" sessionStartFrame="1">',
        ]

        for shape_xml in shape_elements:
            # Shape XML already contains proper indentation
            xml_lines.append(shape_xml)

        xml_lines.append("</Silhouette>")

    return "\n".join(xml_lines), shape_count, total_frames


# ----------------------------------------------------------------- hierarchy
# v3 camera / person / body-part layers. The decomposition is rotobot-nuke's,
# vendored beside this file as _rotobot_hierarchy (see its VENDORED.txt).


def has_reference_frames(data):
    """True for a v3 JSON that carries the camera or persons blocks."""
    return bool(data.get("camera") or data.get("persons"))


def _hierarchy_modules():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from _rotobot_hierarchy import hierarchy, reader
    return reader, hierarchy


def _v2(x, y):
    return f"({x:.9f},{y:.9f})"


def _const(pid, value, extra=""):
    return f'<Property id="{pid}" constant="True"{extra}><Value>{value}</Value></Property>'


def _keyed(pid, frame_values):
    keys = "".join(f'<Key frame="{f + FRAME_OFFSET}" interp="linear">{v}</Key>'
                   for f, v in frame_values)
    return f'<Property id="{pid}">{keys}</Property>'


def _layer_xml(tag, obj_id, label, overrides, children):
    """A Silhouette layer with the full transform property set Silhouette
    writes itself; `overrides` replaces any of them with keyed versions."""
    props = {
        "transform.anchor": _const("transform.anchor", _v2(0, 0)),
        "transform.position": _const("transform.position", _v2(0, 0)),
        "transform.rotate": _const("transform.rotate", "0"),
        "transform.pin_ul": _const("transform.pin_ul", _v2(0, 0)),
        "transform.pin_ur": _const("transform.pin_ur", _v2(1, 0)),
        "transform.pin_lr": _const("transform.pin_lr", _v2(1, 1)),
        "transform.pin_ll": _const("transform.pin_ll", _v2(0, 1)),
    }
    props.update(overrides)
    body = [
        '<Property id="note" constant="True"></Property>',
        _const("color", "(1.000000,1.000000,1.000000)"),
        _const("transform", ""),
        props["transform.anchor"], props["transform.position"],
        _const("transform.scale", "(1.000000000,1.000000000,1.000000000)", ' gang="True"'),
        props["transform.rotate"],
        _const("transform.pin", ""),
        props["transform.pin_ul"], props["transform.pin_ur"],
        props["transform.pin_lr"], props["transform.pin_ll"],
        _const("transform.surface", ""),
        _const("transform.surface_ul", _v2(0, 0)), _const("transform.surface_ur", _v2(1, 0)),
        _const("transform.surface_lr", _v2(1, 1)), _const("transform.surface_ll", _v2(0, 1)),
        _const("transform.matrix", ""),
        '<Property id="stereoOffset"><Value>(0.000000000,0.000000000,0.000000000)</Value></Property>',
        '<Property id="trackSource"><Value>0</Value></Property>',
        '<Property id="objects" expanded="True" constant="True">' + "".join(children) + "</Property>",
    ]
    uuid = generate_uuid(f"hier_layer_{label}_{obj_id}")
    return (f'<{tag} type="Layer" id="{obj_id}" label="{label}" expanded="True" uuid="{uuid}">'
            f'<Properties>{"".join(body)}</Properties></{tag}>')


def create_silhouette_hierarchy_xml(data, log=False):
    """The camera / person / body-part .fxs for a v3 JSON.

    Returns (xml, shape_count, frame_count) like create_silhouette_xml. Raises
    ValueError when the hierarchy would not reproduce the source: every
    vertex is recomposed through camera, person and part before anything is
    written, and the worst error must be within the core's tolerance.
    """
    import io
    reader, hier_mod = _hierarchy_modules()
    global WIDTH, HEIGHT
    res = data.get("resolution") or [WIDTH, HEIGHT]
    WIDTH, HEIGHT = int(res[0]), int(res[1])
    W, H = WIDTH, HEIGHT

    doc = reader.load_json(io.StringIO(json.dumps(data)))
    h = hier_mod.decompose(doc)
    err = hier_mod.round_trip_error(doc, h)
    if err > hier_mod.ROUND_TRIP_TOLERANCE_PX:
        raise ValueError(
            f"camera/person hierarchy does not reproduce the source shapes "
            f"(worst {err:.3f} px > {hier_mod.ROUND_TRIP_TOLERANCE_PX} px)")
    if log:
        print(f"hierarchy: {len(h.pids)} person(s), {len(h.parts)} parts, "
              f"round-trip {err:.2e} px, {len(h.held)} held frame(s)")
        for note in h.held:
            print(f"  held: {note}")

    next_id = iter(range(1, 10 ** 9))
    persons = []
    frame_count = 0
    for pid in h.pids:
        parts = []
        for key in sorted(k for k in h.parts if h.person_of[k] == pid):
            obj = data["objects"][key]
            keys = []
            for f in sorted(h.knots[key]):
                pts = h.knots[key][f]
                if not pts:
                    continue
                # Bone-local pixels -> layer units: scale by 1/height only;
                # the part layer's position is the origin.
                pxml = "".join(
                    f'<Point left="{_v2(q.left_x / H, q.left_y / H)}" '
                    f'right="{_v2(q.right_x / H, q.right_y / H)}">{_v2(q.x / H, q.y / H)}</Point>'
                    for q in pts)
                keys.append(f'<Key frame="{f + FRAME_OFFSET}" interp="linear">'
                            f'<Path closed="True" type="Bspline">{pxml}</Path></Key>')
            frame_count = max(frame_count, len(keys))
            sid = next(next_id)
            label = key.replace(":", "_")
            shape = (f'<Object type="Shape" id="{sid}" label="{label}Shape" expanded="True" '
                     f'uuid="{generate_uuid(sid)}" shape_type="Bspline"><Properties>'
                     f'<Property id="note" constant="True"></Property>'
                     f'<Property id="path">{"".join(keys)}</Property>'
                     f'{create_opacity_xml(obj.get("visibility", {}))}</Properties></Object>')
            xf = sorted(h.parts[key].items())
            parts.append(_layer_xml("Object", next(next_id), label, {
                "transform.position": _keyed("transform.position",
                                             [(f, _v2(x.tx / H, x.ty / H)) for f, x in xf]),
                # Degrees, positive clockwise on screen -- the same sense as
                # the core's Y-down angle, so no sign change.
                "transform.rotate": _keyed("transform.rotate",
                                           [(f, f"{x.angle:.6f}") for f, x in xf]),
            }, [shape]))
        pel = [(f, _v2((x - W / 2) / H, (y - H / 2) / H)) for f, (x, y) in sorted(h.pelvis[pid].items())]
        persons.append(_layer_xml("Object", next(next_id), f"p{pid}",
                                  {"transform.position": _keyed("transform.position", pel)}, parts))

    # The camera: the plate corners through each frame's homography, as
    # fractions of the image. A four-corner pin is a projective map, so this
    # is the homography exactly, and the pin carries down to every child.
    pins = {c: [] for c in ("ul", "ur", "lr", "ll")}
    for f in h.frames:
        m = h.camera[f]
        for c, (x, y) in zip(("ul", "ur", "lr", "ll"), ((0, 0), (W, 0), (W, H), (0, H))):
            px, py = hier_mod.apply_h(m, x, y)
            pins[c].append((f, _v2(px / W, py / H)))
    camera = _layer_xml("Layer", 0, "camera",
                        {f"transform.pin_{c}": _keyed(f"transform.pin_{c}", v) for c, v in pins.items()},
                        persons)

    xml = "\n".join([
        "<!-- Silhouette Shape File -->",
        f'<Silhouette width="{W}" height="{H}" pixelAspect="1" '
        f'workRangeStart="{h.frames[0] + FRAME_OFFSET}" workRangeEnd="{h.frames[-1] + FRAME_OFFSET}" '
        f'sessionStartFrame="1">',
        camera,
        "</Silhouette>",
    ]) + "\n"
    return xml, len(h.parts), frame_count


def main():
    # Parse command line arguments
    args = sys.argv[1:]

    # Check for --log flag
    log_enabled = "--log" in args
    if log_enabled:
        args.remove("--log")

    # Check for --layers flag
    layers_enabled = "--layers" in args
    if layers_enabled:
        args.remove("--layers")

    # v3 JSONs build the camera / person / part hierarchy unless told not to.
    no_hierarchy = "--no-hierarchy" in args
    if no_hierarchy:
        args.remove("--no-hierarchy")

    # --frame-offset N : Silhouette frame = JSON frame + N (default -1). Use 0 for
    # 1-indexed footage so shapes align with the plate instead of landing one frame early.
    global FRAME_OFFSET
    if "--frame-offset" in args:
        i = args.index("--frame-offset")
        try:
            FRAME_OFFSET = int(args[i + 1])
        except (IndexError, ValueError):
            print("Error: --frame-offset requires an integer (e.g. --frame-offset 0)")
            sys.exit(1)
        del args[i:i + 2]

    if len(args) < 1:
        print("Usage: python tokgan_json_to_fxs.py input.json [output.fxs] [--log] [--layers] [--frame-offset N] [--no-hierarchy]")
        print("")
        print("Options:")
        print("  --log            Log execution details for each shape")
        print("  --layers         Group shapes person -> region -> side (no transforms)")
        print("  --no-hierarchy   For a v3 JSON, skip the camera / person / part hierarchy")
        print("  --frame-offset N Silhouette frame = JSON frame + N (default -1; use 0 for 1-indexed footage)")
        sys.exit(1)

    input_path = args[0]

    # Auto-generate output path if not provided
    if len(args) >= 2:
        output_path = args[1]
    else:
        # Replace .json extension with .fxs
        if input_path.endswith('.json'):
            output_path = input_path[:-5] + '.fxs'
        else:
            output_path = input_path + '.fxs'

    try:
        with open(input_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_path}': {e}")
        sys.exit(1)

    start_time = time.time()

    hierarchy = has_reference_frames(data) and not no_hierarchy
    if hierarchy:
        xml_output, shape_count, frame_count = create_silhouette_hierarchy_xml(data, log=log_enabled)
    else:
        xml_output, shape_count, frame_count = create_silhouette_xml(data, log=log_enabled, use_layers=layers_enabled)

    with open(output_path, "w") as f:
        f.write(xml_output)

    elapsed = time.time() - start_time

    print(f"Converted {input_path} to {output_path}")
    print(f"Resolution: {WIDTH}x{HEIGHT}, Pixel Aspect: {PIXEL_ASPECT}")
    print(f"Shapes: {shape_count}, Frames per shape: {frame_count}")
    if hierarchy:
        print("Layer structure: camera (corner pin) -> person -> body part, transforms keyed")
    elif layers_enabled:
        print("Layer structure: person -> region -> side")
    print(f"Time elapsed: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
