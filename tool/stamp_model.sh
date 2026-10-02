#!/usr/bin/env bash
# 给新 ONNX 盖上 RMCS 布局元数据并复验，然后打印要手工改的 YAML 两行。
# 只盖 metadata，不改计算图/权重；不改任何 YAML（第 3 步由你手工改，最安全）。
#
# 用法:
#   stamp_model.sh <model名或路径> [--version X] [--config <yaml>] [--node rl_bridge]
#                  [--input obs] [--history-input obs_history]
#                  [--obs-mean a,b,... --obs-std a,b,...] [--obs-clip V] [--action-clip V]
#
# 例:
#   stamp_model.sh wheel_leg_v3.onnx --version 3.0
set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$TOOL_DIR/.." && pwd)"
WS_DIR="$(cd "$PKG_DIR/../.." && pwd)"
DEFAULT_MODELS_DIR="$PKG_DIR/models"
DEFAULT_CONFIG="$WS_DIR/src/rmcs_bringup/config/wheel-leg-infantry-rl.yaml"
VENV_PY="${RMCS_TOOLS_VENV:-/opt/rmcs-tools/venv}/bin/python"

log() { printf '[stamp_model] %s\n' "$*"; }
die() { printf '[stamp_model] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,11p' "${BASH_SOURCE[0]}"
}

find_python() {
    local cand
    for cand in "${RMCS_ONNX_PYTHON:-}" "$VENV_PY" "$(command -v python3 || true)"; do
        [[ -n "$cand" && -x "$cand" ]] || continue
        if "$cand" -c "import onnx, yaml" >/dev/null 2>&1; then
            printf '%s' "$cand"
            return 0
        fi
    done
    return 1
}

MODEL=""
VERSION=""
CONFIG="$DEFAULT_CONFIG"
NODE="rl_bridge"
INPUT="obs"
HISTORY_INPUT=""
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
    -h | --help)
        usage
        exit 0
        ;;
    --version)
        VERSION="${2:?--version 需要值}"
        shift 2
        ;;
    --config)
        CONFIG="${2:?--config 需要值}"
        shift 2
        ;;
    --node)
        NODE="${2:?--node 需要值}"
        shift 2
        ;;
    --input)
        INPUT="${2:?--input 需要值}"
        shift 2
        ;;
    --history-input)
        HISTORY_INPUT="${2:?--history-input 需要值}"
        shift 2
        ;;
    --obs-mean | --obs-std | --obs-clip | --action-clip)
        EXTRA+=("$1" "${2:?$1 需要值}")
        shift 2
        ;;
    -*)
        die "未知选项 $1"
        ;;
    *)
        if [[ -z "$MODEL" ]]; then MODEL="$1"; else die "多余参数 $1"; fi
        shift
        ;;
    esac
done

[[ -n "$MODEL" ]] || { usage; die "缺少模型名"; }
[[ -f "$CONFIG" ]] || die "配置不存在: $CONFIG"

PY="$(find_python)" \
    || die "找不到带 onnx 的 python；先运行 install_model_tools.sh，或 export RMCS_ONNX_PYTHON=<venv>/bin/python"

case "$MODEL" in
*/*) MODEL_PATH="$MODEL" ;;
*) MODEL_PATH="$DEFAULT_MODELS_DIR/$MODEL" ;;
esac
[[ -f "$MODEL_PATH" ]] || die "模型不存在: $MODEL_PATH"

# 只读预检：模型已盖章时，提示其布局与当前 YAML 是否一致（不阻断，按约定继续盖 + 警告）。
layout_state="$(
    PYTHONPATH="$TOOL_DIR" "$PY" - "$MODEL_PATH" "$CONFIG" "$NODE" <<'PY'
import sys
import rl_layout as L

model, config, node = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    meta = L.read_metadata(model)
except Exception as exc:  # noqa: BLE001
    print(f"UNREADABLE {type(exc).__name__}: {exc}")
    raise SystemExit(0)

obs_sig, act_sig = meta.get("rmcs_obs_layout"), meta.get("rmcs_actions_layout")
if not obs_sig or not act_sig:
    print("UNSTAMPED")
    raise SystemExit(0)

obs, act, _obs_size, act_size = L.load_config(config, node)
want_obs = L.obs_signature(obs, act_size, L.history_length(config, node))
want_act = L.action_signature(act)
print("MATCH" if (obs_sig == want_obs and act_sig == want_act) else "MISMATCH")
PY
)"
case "$layout_state" in
MATCH) : ;;
UNSTAMPED) log "警告：模型未盖章，将按当前 YAML 布局盖章（请确认训练布局与当前 YAML 逐条一致）" ;;
MISMATCH) log "警告：模型已有盖章布局与当前 YAML 不一致！将用当前 YAML 覆盖其布局签名，请确认这是你要的" ;;
*) log "警告：无法读取模型 metadata（$layout_state），仍继续盖章" ;;
esac

STAMP_ARGS=(--model "$MODEL_PATH" --from-config "$CONFIG" --node "$NODE")
[[ -n "$VERSION" ]] && STAMP_ARGS+=(--policy-version "$VERSION")
[[ ${#EXTRA[@]} -gt 0 ]] && STAMP_ARGS+=("${EXTRA[@]}")

STAMP_OUT="$("$PY" "$TOOL_DIR/stamp_layout_metadata.py" "${STAMP_ARGS[@]}")"
printf '%s\n' "$STAMP_OUT"
NEW_ID="$(printf '%s\n' "$STAMP_OUT" | sed -n 's/^model_id[[:space:]]*:[[:space:]]*//p')"
[[ -n "$NEW_ID" ]] || die "未能从盖章输出解析 model_id"

log "复验 check_policy_contract ..."
CHECK_IO=(--input "$INPUT")
[[ -n "$HISTORY_INPUT" ]] && CHECK_IO+=(--history-input "$HISTORY_INPUT")
"$PY" "$TOOL_DIR/check_policy_contract.py" "$MODEL_PATH" \
    --config "$CONFIG" --node "$NODE" --expect-model-id "$NEW_ID" "${CHECK_IO[@]}" \
    || die "复验失败：模型已盖章，但契约校验未通过（见上面的 FAIL 项）"

cat <<EOF

================ 下一步：手工改 YAML 两处 ================
文件: $CONFIG
  rl_bridge.ros__parameters.expected_model_id : $NEW_ID
  policy_server.ros__parameters.rl_model_path : models/$(basename "$MODEL_PATH")
==========================================================
EOF
