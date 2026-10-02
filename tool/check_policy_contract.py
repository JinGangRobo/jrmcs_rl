#!/usr/bin/env python3
"""Check an ONNX policy against the rmcs_rl contract (v2 layout).

Two modes:

  * self-check (no --config; used by CI, deployment YAML lives in the RMCS repo):
      model loads; one output "actions"; either one input (default "obs") or two
      inputs (--input + --history-input, e.g. obs + obs_history); float32; rank 2;
      batch 1 or dynamic; concrete feature dimensions. Layout metadata is OPTIONAL:
      - missing / v1 -> SKIP (stamped layout metadata is optional for model-only self-check)
      - present v2   -> signatures must be internally consistent with the tensor
                        sizes, and policy_layout_hash must match those signatures
                        (catches a bad stamp without any YAML)

  * full contract check (--config deploy.yaml):
      everything above, plus metadata is REQUIRED and must be byte-identical to the
      signatures derived from the YAML and to policy_layout_hash
      (v1 metadata is reported as "re-stamp required", not as a confusing mismatch).
      Normalization metadata is validated when present (mean/std finite, std > 0, clips float > 0).
      One dummy inference runs and yields finite actions (skipped without onnxruntime).

Prints the layout_hash and the model_id (FNV1a64 of the model file bytes).
Exit code: 0 = PASS, 1 = FAIL.

Usage:
  python3 check_policy_contract.py policy.onnx
  python3 check_policy_contract.py policy.onnx --config deploy.yaml --node rl_bridge
  python3 check_policy_contract.py policy.onnx --history-input obs_history   # 双输入
  python3 check_policy_contract.py --config deploy.yaml --print-layout
  python3 check_policy_contract.py policy.onnx --obs 20 --act 6      # 显式期望尺寸（无 YAML）
"""
import argparse
import math
import sys

import rl_layout as layout


def _shape_of(value_info):
    """[1, 20] / ["batch", 20] / [None, 20] —— 符号维原样保留，便于报错。"""
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def _first_diff(expected: str, actual: str) -> str:
    """第一个不同的 entry 的位置与内容（报错时一眼看出差在哪）。"""
    exp = layout.parse_signature(expected)["entries"]
    got = layout.parse_signature(actual)["entries"]
    for index in range(max(len(exp), len(got))):
        left = exp[index] if index < len(exp) else "<缺失>"
        right = got[index] if index < len(got) else "<多余>"
        if left != right:
            return f"第 {index} 条 entry：YAML={left!r} 模型={right!r}"
    return ""


class Report:
    """PASS/FAIL 报告器：任何一项 FAIL → 退出码 1。"""

    def __init__(self):
        self.ok = True

    def check(self, name, passed, detail=""):
        self.ok = self.ok and bool(passed)
        line = f"{'PASS' if passed else 'FAIL'} {name}"
        print(f"{line}: {detail}" if detail else line)
        return bool(passed)

    def skip(self, name, detail=""):
        print(f"SKIP {name}: {detail}" if detail else f"SKIP {name}")

    def info(self, name, detail=""):
        print(f"INFO {name}: {detail}" if detail else f"INFO {name}")


def _print_layout(config, node):
    obs_terms, act_terms, obs_size, act_size = layout.load_config(config, node)
    obs_sig = layout.obs_signature(obs_terms, act_size, layout.history_length(config, node))
    act_sig = layout.action_signature(act_terms)
    digest = layout.layout_hash(obs_sig, act_sig, obs_size, act_size)
    print(f"layout_hash      : {layout.hex16(digest)}")
    print(f"config           : {config} (node {node})")
    print(f"obs_size         : {obs_size}")
    print(f"action_size      : {act_size}")
    print(f"obs_signature    : {obs_sig}")
    print(f"action_signature : {act_sig}")
    print(f"layout_payload   : {layout.layout_payload(obs_sig, act_sig, obs_size, act_size)}")


def _check_normalization(report, meta, obs_size):
    """归一化 metadata：长度/可解析性；mean/std 必须成对。"""
    mean, std = meta.get("rmcs_obs_mean"), meta.get("rmcs_obs_std")
    if mean is None and std is None:
        report.skip("normalization metadata", "未写 rmcs_obs_mean/std（可选）")
    elif (mean is None) != (std is None):
        report.check("normalization metadata", False,
                     "rmcs_obs_mean 与 rmcs_obs_std 必须成对出现")
    else:
        detail, ok = [], True
        for key, text in (("rmcs_obs_mean", mean), ("rmcs_obs_std", std)):
            values = [item for item in text.replace(" ", "").split(",") if item != ""]
            try:
                numbers = [float(item) for item in values]
            except ValueError:
                ok = False
                detail.append(f"{key} 含非浮点项")
                continue
            if len(numbers) != obs_size:
                ok = False
                detail.append(f"{key} 长度 {len(numbers)} != obs_size {obs_size}")
            if any(not math.isfinite(value) for value in numbers):
                ok = False
                detail.append(f"{key} 含非有限值（NaN/Inf）")
            if key == "rmcs_obs_std" and any(value <= 0.0 for value in numbers):
                ok = False
                detail.append("rmcs_obs_std 含 <= 0（除零/符号反转）")
        report.check("normalization metadata", ok, "; ".join(detail) if detail else "mean/std 长度与 obs_size 一致")
    for key in ("rmcs_obs_clip", "rmcs_action_clip"):
        text = meta.get(key)
        if text is None:
            continue
        try:
            value = float(text)
        except ValueError:
            report.check(key, False, f"{key}={text!r} 不是浮点数")
            continue
        report.check(key, math.isfinite(value) and value > 0.0, f"{key}={text}")


def _dummy_inference(report, model_path, input_widths, act_size):
    """跑一次零输入推理，确认输出有限、形状正确（无 onnxruntime 则跳过）。"""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        report.skip("dummy inference", "环境无 onnxruntime")
        return
    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        feed = {
            name: np.zeros((1, width), dtype=np.float32)
            for name, width in input_widths.items()
        }
        actions = session.run(["actions"], feed)[0]
    except Exception as exc:
        report.check("dummy inference", False, f"{type(exc).__name__}: {exc}")
        return
    finite = bool(np.isfinite(actions).all())
    shape_ok = list(actions.shape) == [1, act_size]
    report.check(
        "dummy inference", finite and shape_ok,
        f"actions{list(actions.shape)} finite={finite} "
        f"range=[{actions.min():.4f},{actions.max():.4f}]",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="校验 ONNX 策略与 rmcs_rl 桥契约（v2 布局）一致"
    )
    parser.add_argument("model", nargs="?", help="path to policy.onnx")
    parser.add_argument("--config", "--from-config", dest="config", default=None,
                        help="部署 YAML（桥或消费侧节点段落）")
    parser.add_argument("--node", default=layout.DEFAULT_NODE,
                        help=f"YAML 节点名（缺省 {layout.DEFAULT_NODE}）")
    parser.add_argument("--input", default="obs",
                        help="单输入名，或双输入模型中的每帧输入名（缺省 obs）")
    parser.add_argument("--history-input", default="", dest="history_input",
                        help="双输入模型的历史输入名（如 obs_history）；留空按单输入校验")
    parser.add_argument("--print-layout", action="store_true",
                        help="只打印规范串 + layout_hash（不需要模型文件）后退出")
    parser.add_argument("--obs", type=int, default=None,
                        help="期望的观测尺寸（缺省取模型形状）")
    parser.add_argument("--act", type=int, default=None,
                        help="期望的动作尺寸（缺省取模型形状）")
    parser.add_argument("--expect-model-id", default=None,
                        help="可选：期望的 model_id（16 位 hex），用于部署对账")
    args = parser.parse_args()

    if args.print_layout:
        if not args.config:
            print("ERROR: --print-layout 需要 --config <deploy.yaml>", file=sys.stderr)
            sys.exit(1)
        try:
            _print_layout(args.config, args.node)
        except layout.LayoutError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    if not args.model:
        parser.error("需要 MODEL.onnx（或用 --print-layout --config <deploy.yaml>）")

    report = Report()
    print("== rmcs_rl policy contract check (v2) ==")
    print(f"model : {args.model}")

    obs_sig = act_sig = None
    if args.config:
        try:
            obs_terms, act_terms, obs_size, act_size = layout.load_config(args.config, args.node)
            obs_sig = layout.obs_signature(
                obs_terms, act_size, layout.history_length(args.config, args.node))
            act_sig = layout.action_signature(act_terms)
        except layout.LayoutError as exc:
            print(f"FAIL config: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"config: {args.config} (node {args.node})")
        report.check("config self-consistent", True,
                     f"obs_size={obs_size} action_size={act_size}")
    else:
        obs_size, act_size = args.obs, args.act
        report.skip("config layout check", "未给 --config（只做模型自检，不比对 YAML）")
    digest = None
    if obs_sig is not None:
        digest = layout.layout_hash(obs_sig, act_sig, obs_size, act_size)

    try:
        import onnx
        from onnx import TensorProto
        model = onnx.load(str(args.model), load_external_data=False)
    except ImportError:
        print("ERROR: 需要 onnx（请用容器内 python3 运行）", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"FAIL model load: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    outputs = list(model.graph.output)
    for value in inputs:
        print(f"input  : {value.name} {_shape_of(value)}")
    for value in outputs:
        print(f"output : {value.name} {_shape_of(value)}")

    def _dtype_name(value_info):
        elem = value_info.type.tensor_type.elem_type
        return "float32" if elem == TensorProto.FLOAT else TensorProto.DataType.Name(elem).lower()

    expected_inputs = [args.input]
    if args.history_input:
        expected_inputs.append(args.history_input)
    input_names = [v.name for v in inputs]
    output_names = [v.name for v in outputs]
    names_ok = (
        len(outputs) == 1 and output_names[0] == "actions"
        and sorted(input_names) == sorted(expected_inputs)
    )
    report.check("tensor count/names", names_ok,
                 f"inputs={input_names} outputs={output_names} expected_inputs={expected_inputs}")
    if not names_ok:
        print("== SUMMARY: FAIL (张量不合法，后续检查跳过) ==")
        sys.exit(1)

    out = outputs[0]
    by_name = {v.name: v for v in inputs}
    # The "full" input carries the whole frame*history observation. Without an explicit
    # history input it is just the single input.
    full_name = args.history_input if args.history_input else args.input
    full_in = by_name[full_name]

    report.check(
        "dtype float32",
        all(v.type.tensor_type.elem_type == TensorProto.FLOAT for v in inputs)
        and out.type.tensor_type.elem_type == TensorProto.FLOAT,
        f"inputs={[_dtype_name(v) for v in inputs]} actions={_dtype_name(out)}")

    in_shapes = {name: _shape_of(value) for name, value in by_name.items()}
    out_shape = _shape_of(out)
    report.check(
        "rank 2",
        all(len(shape) == 2 for shape in in_shapes.values()) and len(out_shape) == 2,
        f"inputs={ {name: shape for name, shape in in_shapes.items()} } actions={out_shape}")
    if any(len(shape) != 2 for shape in in_shapes.values()) or len(out_shape) != 2:
        print("== SUMMARY: FAIL (秩不为 2，后续检查跳过) ==")
        sys.exit(1)

    batch_dims = [shape[0] for shape in in_shapes.values()] + [out_shape[0]]
    batch_ok = all(dim == 1 or dim is None or isinstance(dim, str) for dim in batch_dims)
    report.check(
        "batch dimension", batch_ok,
        f"inputs={ {name: shape[0] for name, shape in in_shapes.items()} } "
        f"actions={out_shape[0]!r}（允许 1 或动态维）")

    if args.history_input:
        frame_w = in_shapes[args.input][1]
        history_w = in_shapes[args.history_input][1]
        report.check(
            "history = frame * integer",
            isinstance(frame_w, int) and isinstance(history_w, int)
            and frame_w > 0 and history_w > 0 and history_w % frame_w == 0,
            f"history={history_w} frame={frame_w}")
        if isinstance(frame_w, int) and isinstance(history_w, int) and frame_w > 0:
            report.info("history length", f"{history_w} / {frame_w} = {history_w // frame_w}")

    model_obs, model_act = in_shapes[full_name][1], out_shape[1]
    if not isinstance(model_obs, int) or not isinstance(model_act, int) \
            or model_obs <= 0 or model_act <= 0:
        report.check("concrete obs/action feature size", False,
                     f"obs={in_shapes[full_name]} actions={out_shape}（特征维必须是正整数）")
        print(f"== SUMMARY: {'PASS' if report.ok else 'FAIL'} ==")
        sys.exit(0 if report.ok else 1)

    if obs_size is None:
        obs_size = model_obs
    if act_size is None:
        act_size = model_act
    if args.obs is None and args.act is None:
        report.info("model sizes", f"obs={model_obs} actions={model_act}（取自模型）")

    report.check("obs feature shape", model_obs == obs_size,
                 f"{in_shapes[full_name]} feature={model_obs} expected={obs_size}")
    report.check("actions feature shape", model_act == act_size,
                 f"{out_shape} feature={model_act} expected={act_size}")

    meta = {prop.key: prop.value for prop in model.metadata_props}
    obs_meta, act_meta = meta.get("rmcs_obs_layout"), meta.get("rmcs_actions_layout")
    version_line = meta.get("policy_version")
    if version_line:
        report.info("policy_version", version_line)

    require_metadata = args.config is not None
    v1_hint_printed = False
    for key, value in (("rmcs_obs_layout", obs_meta), ("rmcs_actions_layout", act_meta)):
        if value is None:
            if require_metadata:
                report.check(key, False,
                             f"metadata 缺少 {key}（v2 布局串）→ 用 stamp_layout_metadata.py 盖章")
            else:
                report.skip(key, "未盖章（模型自检模式不要求；完整比对需 --config）")
        elif layout.is_v1_signature(value):
            if not require_metadata:
                report.skip(key, "v1 旧盖章（模型自检模式忽略）")
                continue
            report.check(key, False,
                         f"v1 detected → re-stamp with stamp_layout_metadata.py（{key}={value!r}）")
            if v1_hint_printed:
                continue
            v1_hint_printed = True
            print("     v1 的 id 带类型前缀（" + "vec3:<path> / "
                  + " / ".join(f"{marker}<path>:x" for marker in layout.V1_ID_MARKERS)
                  + " / gravity:<path>，动作条目 <joint>:<mode>），"
                  "同一控制量在 Vector3d 与 DirectionVector 下会生成不同串 → 跨实现漂移；"
                  "v2 去类型化后只需重盖章，不需要重训。")

    if obs_meta and not layout.is_v1_signature(obs_meta):
        try:
            dim_sum = layout.obs_signature_dim(obs_meta)
            report.check("metadata obs dims", dim_sum == obs_size,
                         f"Σdim={dim_sum} == obs_size={obs_size}" if dim_sum == obs_size
                         else f"Σdim={dim_sum} != obs_size={obs_size}")
        except layout.LayoutError as exc:
            report.check("metadata obs dims", False, str(exc))
    if act_meta and not layout.is_v1_signature(act_meta):
        entries = layout.parse_signature(act_meta)["entries"]
        report.check("metadata action entries", len(entries) == act_size,
                     f"{len(entries)} 条 == action_size={act_size}" if len(entries) == act_size
                     else f"{len(entries)} 条 != action_size={act_size}")

    if obs_sig is not None:
        for key, value, expected in (("rmcs_obs_layout", obs_meta, obs_sig),
                                     ("rmcs_actions_layout", act_meta, act_sig)):
            if value is None or layout.is_v1_signature(value):
                continue
            same = value == expected
            diff = "" if same else f"（{_first_diff(expected, value)}）"
            report.check(f"{key} == YAML", same,
                         "byte-identical" if same else f"模型={value!r} YAML={expected!r}{diff}")

    declared_hash = meta.get("policy_layout_hash")
    if declared_hash is None:
        report.skip("policy_layout_hash", "未写（可选，建议盖章）")
    elif not args.config:
        if obs_meta and act_meta and not layout.is_v1_signature(obs_meta) \
                and not layout.is_v1_signature(act_meta):
            expected = layout.hex16(layout.layout_hash(obs_meta, act_meta, obs_size, act_size))
            text = declared_hash.strip().lower()
            if text.startswith("0x"):
                text = text[2:]
            report.check("policy_layout_hash (metadata 自洽)", text == expected[2:],
                         expected if text == expected[2:]
                         else f"模型={declared_hash!r} 期望={expected}")
        else:
            report.skip("policy_layout_hash", "无 --config 且 metadata 不完整，无从核对")
    else:
        text = declared_hash.strip().lower()
        if text.startswith("0x"):
            text = text[2:]
        expected_hash = layout.hex16(digest)[2:]
        report.check("policy_layout_hash", text == expected_hash,
                     layout.hex16(digest) if text == expected_hash
                     else f"模型={declared_hash!r} 期望={layout.hex16(digest)}")

    _check_normalization(report, meta, obs_size)
    _dummy_inference(
        report, args.model, {name: in_shapes[name][1] for name in in_shapes}, act_size)

    try:
        file_id = layout.model_id(args.model)
    except OSError as exc:
        print(f"FAIL model_id: {exc}", file=sys.stderr)
        sys.exit(1)
    if args.expect_model_id:
        expected_id = args.expect_model_id.strip().lower()
        if expected_id.startswith("0x"):
            expected_id = expected_id[2:]
        report.check("expected model_id", expected_id == layout.hex16(file_id)[2:],
                     f"{layout.hex16(file_id)}")

    print("---- identity ----")
    if digest is not None:
        print(f"layout_hash      : {layout.hex16(digest)}")
        print(f"layout_payload   : {layout.layout_payload(obs_sig, act_sig, obs_size, act_size)}")
        print(f"obs_signature    : {obs_sig}")
        print(f"action_signature : {act_sig}")
    print(f"model_id         : {layout.hex16(file_id)}")
    print(f"== SUMMARY: {'PASS' if report.ok else 'FAIL'} ==")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
