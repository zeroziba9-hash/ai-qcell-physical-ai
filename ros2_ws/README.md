# AI-QCell ROS2 workspace

This workspace provides the production-style runtime behind the Streamlit mock demo.

## Graph

```text
camera_node
  └─ /qcell/camera/product [ProductFrame]
       └─ inspection_node (Deep PatchCore)
            └─ /qcell/inspection/result [InspectionResult]
                 └─ decision_node
                      ├─ PASS: /qcell/sort/pass
                      └─ REJECT: /qcell/reject_product [RejectProduct Action]
```

## Verified environment

- Ubuntu 24.04.4 WSL2
- ROS2 Jazzy `ros-base`
- Python 3.12 system-compatible virtual environment
- PyTorch 2.7.0+cu128
- NVIDIA GeForce RTX 4080 SUPER
- `ai_qcell_interfaces` and `ai_qcell_ros`: colcon build passed
- Deep PatchCore → InspectionResult → RejectProduct Action runtime passed

## One-time WSL setup

From PowerShell:

```powershell
cd C:\Users\user\ai-qcell
wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/Users/user/ai-qcell/scripts/install_ros2_wsl.sh
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/ai-qcell/scripts/install_ros2_deep_env.sh
```

## Run

```powershell
# Deterministic ROS2 graph without loading the deep model
powershell -ExecutionPolicy Bypass -File scripts/run_ros2_wsl.ps1 -Mode mock

# CUDA Deep PatchCore inspection graph
powershell -ExecutionPolicy Bypass -File scripts/run_ros2_wsl.ps1 -Mode deep
```

## Reproduce verification

```powershell
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/ai-qcell/scripts/verify_ros2_wsl.sh /mnt/c/Users/user/ai-qcell
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/ai-qcell/scripts/verify_ros2_deep_wsl.sh /mnt/c/Users/user/ai-qcell
```

The first script validates generated interfaces and Topic/Action flow with deterministic
mock inference. The second requires the CUDA environment and verifies the real Deep
PatchCore score, PASS/REJECT decision, action feedback, and final sorting result.
