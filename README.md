# rmcs_rl

RMCS 的 RL 部署包。控制进程通过组件接口组装观测，独立策略进程执行 ONNX 推理；
车辆侧消费器负责动作语义、模式切换、限位和电机控制。

## 职责与源码

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 控制桥 | `src/rl_bridge.cpp` | 组件接口、观测发布、动作时效与契约门控 |
| 桥配置 | `src/rl_bridge/config.cpp` | 参数、维数、词条布局的启动检查 |
| 观测与历史 | `src/rl_bridge/observation.cpp`、`observation_history.cpp`、`observation_timeline.hpp` | 单帧观测、历史堆叠、序号时效与复位 |
| 动作通道 | `src/rl_bridge/action_channel.cpp` | ROS 回调与控制循环之间传递完整快照，控制循环不等待锁 |
| ROS 策略服务 | `src/policy_server.cpp` | 订阅观测、发布动作和可选状态 |
| 策略模型 | `src/policy/policy_model.cpp` | 元数据、模型 ID、归一化、裁剪、有限值检查 |
| 推理后端 | `src/policy/onnx_runtime.cpp` | ONNX Runtime 会话、张量形状及执行 |
| 子进程管理 | `src/policy_server_launcher.cpp` | 启停、退出回收、延时重启 |
| 参数解析 | `src/parameters.cpp` | 上述组件共用的类型及数值检查 |

组件保持 RMCS 的 `.cpp` 类布局，构造函数直接注册输出。观测输入类型依赖其他组件的
输出表，在 `before_pairing()` 中解析。非模板模型实现位于 `.cpp`，私有头文件放在
`src/policy/`；控制桥不链接 ONNX Runtime。

## 车辆控制链

| 车辆 | 配置（位于 rmcs_bringup/config） | 消费及控制 |
| --- | --- | --- |
| wheel-leg | `wheel-leg-infantry-rl.yaml` | `WheelLegRlImu` → 桥 → `WheelLegRlConsumer` → 配对角度环 → 速度 PID → DM MIT 力矩 |
| deformable | `deformable-infantry-omni-rl.yaml` | 桥 → `DeformableRlSuspension` → `DeformableSuspensionArbiter` → 关节控制器 |

wheel-leg 的 IMU 观测使用 Body xyz；关节 offset、配对选弧和机械限位由硬件及车辆控制层管理。
deformable 使用 RF、LF、LB、RB 的策略动作顺序，消费器根据标定把 RL 坐标转换为物理角；
RL 未就绪时仲裁器使用传统姿态目标。双下复位经组件接口清除历史、动作及控制状态。

## 构建与启动

在 ROS 工作区内：

```sh
colcon build --merge-install --symlink-install --packages-select rmcs_rl rmcs_core rmcs_bringup
source install/setup.bash
```

使用对应车辆的 bringup 配置启动 RMCS；配置中的 `PolicyServerLauncher` 自动启动策略进程。
需要独立调试策略进程时，将 launcher 的 `autostart` 设为 false，再运行：

```sh
ros2 run rmcs_rl policy_server --ros-args --params-file <车辆配置.yaml>
```

车辆配置统一放在 `rmcs_bringup/config/`；旧的、不含完整消费链的 `config/executor.yaml` 已移除。

## 模型与契约

模型输入为 float32 `obs[batch,N]`，输出为 float32 `actions[batch,M]`，batch 为 1 或动态。
也支持**双输入**模型（如训练导出常见的 `obs` + `obs_history`）：在 `policy_server` 上设置
`history_input_name`（例如 `obs_history`），此时模型须有两个输入——每帧输入（较窄）与整段历史
输入（较宽，为帧宽的整数倍）；桥发来的整段观测全部喂给历史输入，其**尾部一帧**喂给每帧输入。
`input_name` 缺省 `obs`。单输入模型保持原行为（`history_input_name` 留空）。
部署模型须含 `rmcs_obs_layout`、`rmcs_actions_layout`；建议带 `policy_layout_hash` 和版本。
`history_length × 单帧维数 = rl_obs_size`。归一化均值和标准差必须成对提供，标准差为正数。

新模型先核对训练的观测顺序、坐标、基准角、缩放和频率，再盖章：

```sh
bash src/rmcs_rl/tool/stamp_model.sh <模型.onnx> --config <车辆配置.yaml> --version <版本>
```

按工具输出同步 `policy_server.rl_model_path`、`rl_bridge.expected_model_id`，然后验证：

```sh
python3 src/rmcs_rl/tool/check_policy_contract.py <模型.onnx> --config <车辆配置.yaml> --node rl_bridge --expect-model-id <ID>
```

盖章只写元数据；布局匹配不能证明训练侧的坐标或动作含义正确。
更新模型或配置后重启 RMCS。模型或布局不匹配会锁存拒绝动作，需要重启恢复。

当前 `models/deformable_sps_V2.onnx` 缺少 `rmcs_obs_layout` 和 `rmcs_actions_layout`，
完整契约检查与策略服务会拒绝它。这是已有模型的部署缺口，需核对训练契约后补元数据；
不能只按张量维数推断其含义。

## 运行接口

- `<rl_base>/obs`、`<rl_base>/action`：两个进程间的 ROS 消息。
- `<rl_base>/policy_status`：可选模型身份和推理耗时信息。
- 控制进程内部使用组件接口 `<rl_base>/valid`、`healthy`、`action_age` 和配置的动作输出。
- 动作必须对应最近两次有效观测之一且未超时；复位前的动作不能重新获得控制权。
- `invalid_value` 可选 `nan`、`zero`、`hold`；实际车辆使用 `"nan"` 让消费器处理失效。

## 工具和测试

`tool/` 仅保留模型校验、盖章和依赖安装工具。离线模型生成及回归放在 `test/`，
不安装到部署目录；合成模型不用于实车启动。历史临时脚本与生成报告可从 Git 历史获取。

```sh
colcon build --merge-install --symlink-install --packages-select rmcs_rl rmcs_core --cmake-args -DBUILD_TESTING=ON
colcon test --merge-install --packages-select rmcs_rl rmcs_core
PYTHON=<带 onnx、onnxruntime、PyYAML 的 Python> bash src/rmcs_rl/test/test_layout_contract.sh
```

C++ 回归覆盖推理归一化、历史帧与跨线程动作快照；Python 回归覆盖布局解析、盖章及错误契约拒绝。
车辆包保留 DM 使能、帧解码、闭链几何、Body IMU 以及 MuJoCo 查看器回归。
