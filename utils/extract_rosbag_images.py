#!/usr/bin/env python3
"""Extract image messages from a ROS 2 SQLite bag into image files.

The script reads a ``.db3`` bag and the two supported ROS message layouts
directly, so neither ``metadata.yaml`` nor a sourced ROS 2 installation is
required.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


SUPPORTED_IMAGE_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
}


@dataclass(frozen=True)
class Topic:
    id: int
    name: str
    message_type: str
    serialization_format: str
    message_count: int


@dataclass(frozen=True)
class RawImage:
    height: int
    width: int
    encoding: str
    is_bigendian: bool
    step: int
    data: memoryview


@dataclass(frozen=True)
class CompressedImage:
    format: str
    data: memoryview


class CdrReader:
    """Minimal CDR1 reader for the stable sensor_msgs image definitions."""

    def __init__(self, serialized: bytes) -> None:
        if len(serialized) < 4:
            raise ValueError("CDR message is shorter than its 4-byte header.")

        encapsulation = int.from_bytes(serialized[:2], byteorder="big")
        if encapsulation == 0x0000:
            self.byte_order = ">"
        elif encapsulation == 0x0001:
            self.byte_order = "<"
        else:
            raise ValueError(
                f"Unsupported CDR encapsulation 0x{encapsulation:04x}; "
                "only plain CDR1 is supported."
            )

        self.data = memoryview(serialized)
        self.offset = 4

    def align(self, alignment: int) -> None:
        self.offset += (-self.offset) % alignment

    def unpack(self, code: str, alignment: int) -> int:
        self.align(alignment)
        size = struct.calcsize(code)
        end = self.offset + size
        if end > len(self.data):
            raise ValueError("Unexpected end of serialized CDR message.")
        value = struct.unpack_from(f"{self.byte_order}{code}", self.data, self.offset)[0]
        self.offset = end
        return value

    def uint8(self) -> int:
        return self.unpack("B", 1)

    def int32(self) -> int:
        return self.unpack("i", 4)

    def uint32(self) -> int:
        return self.unpack("I", 4)

    def string(self) -> str:
        length = self.uint32()
        if length == 0:
            return ""
        end = self.offset + length
        if end > len(self.data):
            raise ValueError("Unexpected end of CDR string.")
        encoded = self.data[self.offset:end]
        self.offset = end
        if encoded[-1] != 0:
            raise ValueError("CDR string is missing its null terminator.")
        return bytes(encoded[:-1]).decode("utf-8")

    def byte_sequence(self) -> memoryview:
        length = self.uint32()
        end = self.offset + length
        if end > len(self.data):
            raise ValueError("Unexpected end of CDR byte sequence.")
        value = self.data[self.offset:end]
        self.offset = end
        return value


def open_bag(bag_path: Path) -> sqlite3.Connection:
    """Open a bag read-only, without creating a missing database by accident."""
    uri = f"{bag_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def read_topics(connection: sqlite3.Connection) -> list[Topic]:
    rows = connection.execute(
        """
        SELECT t.id, t.name, t.type, t.serialization_format, COUNT(m.id)
        FROM topics AS t
        LEFT JOIN messages AS m ON m.topic_id = t.id
        GROUP BY t.id, t.name, t.type, t.serialization_format
        ORDER BY t.id
        """
    )
    return [Topic(*row) for row in rows]


def print_topics(topics: list[Topic]) -> None:
    if not topics:
        print("No topics found in the bag.")
        return

    name_width = max(len("TOPIC"), *(len(topic.name) for topic in topics))
    type_width = max(len("TYPE"), *(len(topic.message_type) for topic in topics))
    print(f"{'TOPIC':<{name_width}}  {'TYPE':<{type_width}}  MESSAGES")
    for topic in topics:
        print(
            f"{topic.name:<{name_width}}  "
            f"{topic.message_type:<{type_width}}  {topic.message_count}"
        )


def select_topic(topics: list[Topic], requested_name: str | None) -> Topic:
    if requested_name is not None:
        matching = [topic for topic in topics if topic.name == requested_name]
        if not matching:
            available = "\n  ".join(topic.name for topic in topics)
            raise ValueError(
                f"Topic {requested_name!r} is not in the bag. Available topics:\n  "
                f"{available}"
            )
        selected = matching[0]
        if selected.message_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(
                f"Topic {requested_name!r} has unsupported type "
                f"{selected.message_type!r}; expected sensor_msgs/msg/Image or "
                "sensor_msgs/msg/CompressedImage."
            )
        return selected

    image_topics = [
        topic for topic in topics if topic.message_type in SUPPORTED_IMAGE_TYPES
    ]
    if len(image_topics) == 1:
        return image_topics[0]
    if not image_topics:
        raise ValueError("The bag contains no supported image topics.")

    choices = "\n  ".join(topic.name for topic in image_topics)
    raise ValueError(
        "The bag contains multiple image topics. Select one with --topic:\n  "
        f"{choices}"
    )


def skip_header(reader: CdrReader) -> None:
    """Read and discard std_msgs/Header (stamp and frame_id)."""
    reader.int32()  # stamp.sec
    reader.uint32()  # stamp.nanosec
    reader.string()  # frame_id


def deserialize_image(serialized: bytes, message_type: str) -> RawImage | CompressedImage:
    reader = CdrReader(serialized)
    skip_header(reader)

    if message_type == "sensor_msgs/msg/Image":
        return RawImage(
            height=reader.uint32(),
            width=reader.uint32(),
            encoding=reader.string(),
            is_bigendian=bool(reader.uint8()),
            step=reader.uint32(),
            data=reader.byte_sequence(),
        )
    if message_type == "sensor_msgs/msg/CompressedImage":
        return CompressedImage(format=reader.string(), data=reader.byte_sequence())
    raise ValueError(f"Unsupported image message type: {message_type}")


def encoding_details(encoding: str) -> tuple[np.dtype, int, int | None]:
    """Return native dtype, channel count, and optional OpenCV conversion."""
    normalized = encoding.lower()
    known: dict[str, tuple[object, int, int | None]] = {
        "mono8": (np.uint8, 1, None),
        "mono16": (np.uint16, 1, None),
        "bgr8": (np.uint8, 3, None),
        "rgb8": (np.uint8, 3, cv2.COLOR_RGB2BGR),
        "bgra8": (np.uint8, 4, None),
        "rgba8": (np.uint8, 4, cv2.COLOR_RGBA2BGRA),
        "bgr16": (np.uint16, 3, None),
        "rgb16": (np.uint16, 3, cv2.COLOR_RGB2BGR),
        "bgra16": (np.uint16, 4, None),
        "rgba16": (np.uint16, 4, cv2.COLOR_RGBA2BGRA),
        "bayer_rggb8": (np.uint8, 1, cv2.COLOR_BAYER_RG2BGR),
        "bayer_bggr8": (np.uint8, 1, cv2.COLOR_BAYER_BG2BGR),
        "bayer_gbrg8": (np.uint8, 1, cv2.COLOR_BAYER_GB2BGR),
        "bayer_grbg8": (np.uint8, 1, cv2.COLOR_BAYER_GR2BGR),
        "bayer_rggb16": (np.uint16, 1, cv2.COLOR_BAYER_RG2BGR),
        "bayer_bggr16": (np.uint16, 1, cv2.COLOR_BAYER_BG2BGR),
        "bayer_gbrg16": (np.uint16, 1, cv2.COLOR_BAYER_GB2BGR),
        "bayer_grbg16": (np.uint16, 1, cv2.COLOR_BAYER_GR2BGR),
    }
    if normalized in known:
        dtype, channels, conversion = known[normalized]
        return np.dtype(dtype), channels, conversion

    generic = re.fullmatch(r"(8|16|32|64)(U|S|F)C([1-4])", encoding.upper())
    if generic:
        bits, kind, channels_text = generic.groups()
        dtype = np.dtype(f"{kind.lower()}{int(bits) // 8}")
        return dtype, int(channels_text), None

    raise ValueError(f"Unsupported sensor_msgs/Image encoding: {encoding!r}")


def raw_image_to_array(message) -> np.ndarray:
    dtype, channels, color_conversion = encoding_details(message.encoding)
    bytes_per_row = message.width * channels * dtype.itemsize

    if message.step < bytes_per_row:
        raise ValueError(
            f"Image step ({message.step}) is smaller than its pixel data "
            f"({bytes_per_row} bytes per row)."
        )

    data = np.frombuffer(message.data, dtype=np.uint8)
    required_size = message.height * message.step
    if data.size < required_size:
        raise ValueError(
            f"Image data is truncated: expected at least {required_size} bytes, "
            f"got {data.size}."
        )

    # Copy removes any per-row padding before interpreting multi-byte pixels.
    rows = data[:required_size].reshape(message.height, message.step)
    packed = rows[:, :bytes_per_row].copy()
    byte_order = ">" if message.is_bigendian else "<"
    wire_dtype = dtype.newbyteorder(byte_order) if dtype.itemsize > 1 else dtype
    image = packed.view(wire_dtype)
    image = image.reshape(message.height, message.width, channels)
    image = image.astype(dtype, copy=False)

    if channels == 1:
        image = image[:, :, 0]
    if color_conversion is not None:
        image = cv2.cvtColor(image, color_conversion)
    return image


def compressed_image_to_array(message) -> np.ndarray:
    encoded = np.frombuffer(message.data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not decode compressed format {message.format!r}.")
    return image


def iter_messages(
    connection: sqlite3.Connection, topic_id: int
) -> Iterator[tuple[int, bytes]]:
    rows = connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic_id,),
    )
    yield from rows


def extract(args: argparse.Namespace) -> int:
    bag_path = args.bag.expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file does not exist: {bag_path}")
    if bag_path.suffix.lower() != ".db3":
        raise ValueError(f"Expected a ROS 2 SQLite .db3 bag, got: {bag_path}")

    with open_bag(bag_path) as connection:
        topics = read_topics(connection)
        if args.list_topics:
            print_topics(topics)
            return 0

        topic = select_topic(topics, args.topic)
        if topic.serialization_format != "cdr":
            raise ValueError(
                f"Unsupported serialization format {topic.serialization_format!r}; "
                "expected 'cdr'."
            )

        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else bag_path.with_name(f"{bag_path.stem}_frames")
        )
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}\n"
                "Use --overwrite to replace frames with matching filenames."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        converter = (
            raw_image_to_array
            if topic.message_type == "sensor_msgs/msg/Image"
            else compressed_image_to_array
        )

        sampled_count = (topic.message_count + args.every - 1) // args.every
        expected_count = (
            min(sampled_count, args.max_frames)
            if args.max_frames is not None
            else sampled_count
        )
        extension = "jpg" if args.format == "jpg" else "png"
        write_options = (
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            if extension == "jpg"
            else [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
        )

        written = 0
        messages = iter_messages(connection, topic.id)
        for message_index, (timestamp, serialized) in enumerate(messages):
            if message_index % args.every:
                continue
            if args.max_frames is not None and written >= args.max_frames:
                break

            message = deserialize_image(serialized, topic.message_type)
            image = converter(message)
            if extension == "jpg" and image.dtype != np.uint8:
                raise ValueError(
                    "JPEG output only supports 8-bit images in this script; use "
                    "--format png for this topic."
                )

            filename = f"frame_{written:06d}_{timestamp}.{extension}"
            destination = output_dir / filename
            if not cv2.imwrite(str(destination), image, write_options):
                raise OSError(f"Failed to write image: {destination}")
            written += 1

            if written == 1:
                print(
                    f"Extracting {expected_count} frame(s) from {topic.name} "
                    f"to {output_dir}"
                )
            if written % 250 == 0:
                print(f"Extracted {written}/{expected_count} frames")

    print(f"Done: wrote {written} frame(s) to {output_dir}")
    return 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def bounded_int(minimum: int, maximum: int):
    def parse(value: str) -> int:
        parsed = int(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"value must be between {minimum} and {maximum}"
            )
        return parsed

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract sensor_msgs/Image or sensor_msgs/CompressedImage messages "
            "from a ROS 2 SQLite bag."
        )
    )
    parser.add_argument("bag", type=Path, help="path to the ROS 2 .db3 bag")
    parser.add_argument(
        "--topic",
        help="image topic to extract (auto-selected when the bag has only one)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output folder (default: <bag_stem>_frames beside the bag)",
    )
    parser.add_argument(
        "--format", choices=("png", "jpg"), default="png", help="output format"
    )
    parser.add_argument(
        "--every",
        type=positive_int,
        default=1,
        help="extract every Nth message (default: 1)",
    )
    parser.add_argument(
        "--max-frames", type=positive_int, help="stop after this many output frames"
    )
    parser.add_argument(
        "--jpeg-quality",
        type=bounded_int(0, 100),
        default=95,
        help="JPEG quality from 0 to 100 (default: 95)",
    )
    parser.add_argument(
        "--png-compression",
        type=bounded_int(0, 9),
        default=3,
        help="PNG compression from 0 to 9 (default: 3)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of frames with matching filenames",
    )
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="list bag topics and message counts without extracting",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return extract(args)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
