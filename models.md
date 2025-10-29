---
layout: default
title: "Models"
---

# 🤖 Models Used

## YOLOv8
Used for object and agent detection in frames.

- **Classes:** 50+ Valorant entities
- **Trained weights:** `runs/detect/train14/weights/best.pt`
- **Framework:** Ultralytics YOLOv8

## LLaVA-7B
Used for generating contextual and tactical commentary from detected frames.

- **Type:** Vision-Language Model
- **Size:** 7B parameters
- **Quantization:** 4-bit NF4
- **Prompt:** Valorant-specific tactical commentary generation.
