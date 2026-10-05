#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
DEFAULT_REMOTE_HOST="remote"
ONNXRUNTIME_VERSION="1.20.0"

declare -A ONNXRUNTIME_SHA256=(
    [linux-x64]=aa70d48b22e264b82e83f63245b51ddc9a47ae4a3a66903efaff1ba68b7b5930
    [linux-aarch64]=b4d7c6e2c45f8edabe5d28e9bc59ec8d5a4a4af36660cda16e94b2ad85f2a52a
)

usage() {
    cat <<EOF
Usage:
  $(basename "$0") local
  $(basename "$0") remote [host]
EOF
}

require_command() {
    local command_name="$1"
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'Missing command: %s\n' "${command_name}" >&2
        exit 1
    fi
}

run_privileged() {
    if command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        "$@"
    fi
}

platform_of() {
    case "$(uname -m)" in
    x86_64 | amd64) echo "linux-x64" ;;
    aarch64 | arm64) echo "linux-aarch64" ;;
    *)
        echo "Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
    esac
}

installed_matches() {
    [[ -f "/usr/local/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION}" ]]
}

install_onnxruntime() {
    local platform="$1"
    local archive="/tmp/onnxruntime-${platform}-${ONNXRUNTIME_VERSION}.tgz"
    local extract_dir="/tmp/onnxruntime-${platform}-${ONNXRUNTIME_VERSION}"

    if installed_matches; then
        printf 'ONNX Runtime %s (%s) 已安装，跳过\n' "${ONNXRUNTIME_VERSION}" "${platform}"
        return
    fi

    if [[ ! -f "${archive}" ]]; then
        printf '下载 ONNX Runtime %s (%s) ...\n' "${ONNXRUNTIME_VERSION}" "${platform}"
        curl -L --fail \
            "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${platform}-${ONNXRUNTIME_VERSION}.tgz" \
            -o "${archive}"
    fi

    local expected="${ONNXRUNTIME_SHA256[${platform}]:-}"
    if [[ -n "${expected}" ]]; then
        printf '校验 SHA256 ...\n'
        printf '%s  %s\n' "${expected}" "${archive}" | sha256sum --check --status || {
            printf 'SHA256 校验失败: %s\n' "${archive}" >&2
            exit 1
        }
    fi

    rm -rf "${extract_dir}"
    mkdir -p "${extract_dir}"
    tar -xzf "${archive}" -C "${extract_dir}" --strip-components=1

    run_privileged cp "${extract_dir}/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION}" /usr/local/lib/
    run_privileged ln -sf "/usr/local/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION}" /usr/local/lib/libonnxruntime.so.1
    run_privileged ln -sf /usr/local/lib/libonnxruntime.so.1 /usr/local/lib/libonnxruntime.so
    run_privileged ldconfig
}

install_local() {
    local platform
    platform="$(platform_of)"
    install_onnxruntime "${platform}"

    cat <<EOF

安装完成：$(ldconfig -p | grep -o 'libonnxruntime\.so\.[0-9.]*' | sort -u | tr '\n' ' ')

验证：
  ldconfig -p | grep onnxruntime    # 应显示 libonnxruntime.so.1 => /usr/local/lib/...
EOF
}

install_remote() {
    local remote_host="$1"
    local remote_dir="/tmp/rmcs-rl-install"

    require_command ssh
    require_command scp

    printf '打包并推送到 %s ...\n' "${remote_host}"
    ssh "${remote_host}" "rm -rf '${remote_dir}' && mkdir -p '${remote_dir}'"
    scp "$0" "${remote_host}:${remote_dir}/install_rl_deps.sh"
    ssh "${remote_host}" "bash '${remote_dir}/install_rl_deps.sh' local"
}

main() {
    case "${1:-}" in
    local)
        install_local
        ;;
    remote)
        install_remote "${2:-${DEFAULT_REMOTE_HOST}}"
        ;;
    *)
        usage >&2
        exit 1
        ;;
    esac
}

main "$@"
