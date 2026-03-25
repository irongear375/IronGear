# Iron Gear — Sprint 1 Results Summary
Generated: 2026-03-24 22:08

## Selected Model
**YOLO11x** — best.pt saved at:
`/home/sagemaker-user/user-default-efs/IronGear/Sprint1-POC/models/sprint1_yolo11x/weights/best.pt`

## Training Config
| Parameter | Value |
|---|---|
| Epochs trained | 50 |
| Best epoch | 49 |
| Best val mAP@0.5 | 0.6694 |
| Image size | 640 |
| Batch size | 2 |
| Optimizer | AdamW |
| LR | 0.001 cosine |
| Augmentation | mosaic, mixup=0.15, copy_paste=0.3 |

## Test Set Metrics
| Metric | Score |
|---|---|
| mAP@0.5 | 0.6850 |
| mAP@0.5:0.95 | 0.3913 |
| Precision | 0.6932 |
| Recall | 0.6971 |

## Per-Class AP@0.5 (Test)
| Class | AP@0.5 | AP@0.5:0.95 |
|---|---|---|
| fracture | 0.9385 | 0.5495 |
| metal_implant | 0.8646 | 0.6641 |
| periosteal_reaction | 0.6494 | 0.2875 |
| pronator_sign | 0.6654 | 0.3237 |
| soft_tissue | 0.3068 | 0.1317 |

## Sprint 2 Plan
- Upgrade to 8-class detection (add bone_anomaly, bone_lesion, foreign_body)
- Add plaster cast binary classifier
- ClearML experiment tracking (MLOps Level 1)
- Automated retraining pipeline
- TTA (Test Time Augmentation) during evaluation
