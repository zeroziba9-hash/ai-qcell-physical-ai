# AI-QCell

Physical AI 기반 스마트팩토리 품질검사 셀 포트폴리오 프로젝트입니다.

AI-QCell은 가상 생산라인, 기준 이미지 비교, 정상 제품만 학습하는 경량 모델과 MVTec AD 실제 산업 이미지용 Deep PatchCore를 제공합니다. 결함 히트맵, REJECT 명령, 생산 KPI와 모델 평가 결과를 한 대시보드에서 확인할 수 있습니다.

## 시각화 결과

![AI-QCell 비전 불량검사 결과](docs/images/qcell_vision_demo.png)

- 정상 기준 / 검사 대상 / AI 결함 히트맵 비교
- 불량 바운딩 박스와 REJECT 명령
- 이상 점수, 결함 면적, 추론시간 KPI
- 생산량, 불량률, 결함 유형 분포 대시보드

Streamlit 왼쪽 메뉴의 `vision inspection` 페이지에서 직접 조작할 수 있습니다.

## 학습 모델 평가

![Patch Memory 모델 평가](docs/images/model_evaluation.png)

`PatchMemoryDetector`는 정상 제품 이미지 40장만 학습하고, 위치별 패치 특징을 최근접 정상 메모리와 비교합니다.

| 지표 | 합성 평가셋 결과 |
|---|---:|
| Accuracy | 98.21% |
| Precision | 97.30% |
| Recall | 100.00% |
| F1 | 98.63% |
| AUROC | 1.000 |

모델 재학습 및 평가:

```bash
python -m scripts.train_patch_memory
python -m scripts.evaluate_patch_memory
```

> 위 수치는 정렬된 합성 제품 56개에 대한 베이스라인 결과이며 MVTec AD 벤치마크 점수가 아닙니다.

## Deep PatchCore · 실제 MVTec AD

![Deep PatchCore MVTec bottle 평가](docs/images/deep_patchcore_bottle_evaluation.png)

- ImageNet 사전학습 ResNet18 `layer2`/`layer3` 특징
- 정상 이미지 177장으로 메모리 뱅크 학습
- 정상 이미지 32장으로 판정 임계값 보정
- 실제 MVTec bottle 테스트 이미지 83장 평가
- 결함 이미지 없이 대형 파손, 소형 파손, 오염 탐지

| 지표 | MVTec bottle 결과 |
|---|---:|
| Image AUROC | 1.000 |
| Pixel AUROC | 0.9782 |
| Accuracy / F1 | 100.00% / 100.00% |
| 정상 / 결함 테스트 | 20 / 63 |

> MVTec AD 데이터는 CC BY-NC-SA 4.0이며 연구·교육용 포트폴리오 범위로 사용합니다.

## ROS2 자동 선별 파이프라인

![AI-QCell ROS2 자동 선별 구조](docs/images/ros2_pipeline_architecture.png)

Deep PatchCore의 판정을 실제 생산 셀 동작으로 연결하는 ROS2 패키지를 추가했습니다.

| 구성 요소 | 인터페이스 | 역할 |
|---|---|---|
| `camera_node` | `/qcell/camera/product` Topic | 제품 ID와 카메라 이미지 발행 |
| `inspection_node` | `/qcell/inspection/result` Topic | GPU Deep PatchCore 검사 결과 발행 |
| `decision_node` | Topic + Action Client | PASS 통과 또는 REJECT 목표 전송 |
| `reject_action_server` | `/qcell/reject_product` Action | 선별기 진행률 25/50/75/100% 피드백 |
| `dashboard_bridge` | Topic Subscriber | 검사·선별 이벤트와 KPI 연결 |

현재 PC에는 ROS2가 설치되어 있지 않으므로 Streamlit의 `ROS2 Sorting Pipeline` 페이지는 같은
메시지·토픽·액션 계약을 재현하는 Mock Runtime으로 즉시 시연됩니다. 실제 ROS2 패키지는
[`ros2_ws`](ros2_ws/README.md)에 있으며 ROS2 환경에서 `colcon build` 후 실행할 수 있습니다.

## 실행

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-ml.txt --index-url https://download.pytorch.org/whl/cu128
python -m scripts.download_mvtec_bottle
python -m scripts.train_deep_patchcore
python -m scripts.evaluate_deep_patchcore
streamlit run app.py
```

## 테스트

```bash
pytest
```

## 로드맵

- Phase 1: 가상 생산라인과 품질 대시보드
- Phase 2: MVTec AD bottle + Deep PatchCore 완료
- Phase 3: ROS2 Topic/Action 자동 선별 파이프라인 완료
- Phase 4: 카메라·센서 입력 및 디지털 트윈

