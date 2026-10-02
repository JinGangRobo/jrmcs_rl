#!/usr/bin/env python3
"""Generate test/data/dual_history.onnx (dual-input contract fixture).

The fixture is a tiny ONNX graph with two inputs and one output:

    inputs : obs[1, 25] , obs_history[1, 125]   (frame_size=25, history=5)
    output : actions[1, 150]                    (Concat(obs, obs_history) axis=1)

It exercises the dual-input OnnxRuntime binding: the history input receives the
whole observation while the frame input receives its trailing frame. It is built
with a hand-rolled protobuf writer so no `onnx` package is required.

Usage:
    python3 test/gen_dual_history_fixture.py [--frame 25] [--history 5]
"""
import argparse
from pathlib import Path


def varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def f_int(field: int, value: int) -> bytes:
    return varint((field << 3) | 0) + varint(value)


def f_bytes(field: int, payload: bytes) -> bytes:
    return varint((field << 3) | 2) + varint(len(payload)) + payload


def dim(value: int) -> bytes:
    # TensorShapeProto.Dimension { dim_value = value }
    return f_bytes(1, f_int(1, value))


def value_info(name: str, dims) -> bytes:
    shape = b"".join(dim(d) for d in dims)
    tensor = f_int(1, 1) + f_bytes(2, shape)  # Tensor { elem_type = FLOAT, shape }
    type_proto = f_bytes(1, tensor)  # TypeProto { tensor_type }
    return f_bytes(1, name.encode()) + f_bytes(2, type_proto)


def build(frame: int, history: int) -> bytes:
    total = frame * history
    axis = f_bytes(1, b"axis") + f_int(20, 2) + f_int(3, 1)  # name/type=INT/i=1
    node = (
        f_bytes(1, b"obs")
        + f_bytes(1, b"obs_history")
        + f_bytes(2, b"actions")
        + f_bytes(3, b"concat")
        + f_bytes(4, b"Concat")
        + f_bytes(5, axis)
    )
    graph = (
        f_bytes(1, node)
        + f_bytes(2, b"dual_history")
        + f_bytes(11, value_info("obs", [1, frame]))
        + f_bytes(11, value_info("obs_history", [1, total]))
        + f_bytes(12, value_info("actions", [1, frame + total]))
    )
    return (
        f_int(1, 8)  # ir_version
        + f_bytes(2, b"rmcs_rl/gen_dual_history_fixture")
        + f_bytes(7, graph)
        + f_bytes(8, f_int(2, 13))  # opset_import version=13
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=25)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument(
        "-o", "--output",
        default=str(Path(__file__).resolve().parent / "data" / "dual_history.onnx"),
    )
    args = parser.parse_args()
    if args.frame <= 0 or args.history <= 1:
        parser.error("--frame must be > 0 and --history must be > 1")
    payload = build(args.frame, args.history)
    Path(args.output).write_bytes(payload)
    print(f"wrote {args.output} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
