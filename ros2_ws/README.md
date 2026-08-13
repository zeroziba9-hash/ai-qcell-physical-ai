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

## Build and run

Run inside a terminal where ROS2 is installed and sourced. The project Python package
must also be importable so that `inspection_node` can load `qcell.deep_patchcore`.

```powershell
cd C:\Users\user\ai-qcell
pip install -e .
cd ros2_ws
colcon build --symlink-install
call install\setup.bat
ros2 launch ai_qcell_ros qcell_pipeline.launch.py model_path:=C:/Users/user/ai-qcell/models/deep_patchcore_bottle.pt dataset_root:=C:/Users/user/ai-qcell/data/mvtec-ad/bottle
```

The current machine does not have ROS2 installed, so the Streamlit `ROS2 Sorting
Pipeline` page uses `qcell.ros2_pipeline` as a contract-compatible mock runtime.
