# Inference Guide

This document shows how to load and use trained Being-VL models for inference.

## Prerequisites

- Completed 3-stage training (see [Train.md](Train.md))
- Trained model available at `/path/to/your/workspace/models/beingvl/stage-3/`
- vBPE tokenizer available at `/path/to/your/workspace/data/vbpe/vbpe.pkl`

## Load Model

Load your trained Being-VL model:

```python
import torch
from PIL import Image
from beingvl import BeingVL

# Load the trained model
model = BeingVL(
    model_path="/path/to/your/workspace/models/beingvl/stage-3",
    vbpe_path="/path/to/your/workspace/data/vbpe/vbpe.pkl"
)
model.load_model(
    torch_dtype=torch.bfloat16,
)
model.model.eval()
```

## Inference with Multimodal Inputs

Generate responses for images with text prompts:

```python
# Prepare images and prompts
images_list = [
    "/path/to/your/workspace/data/images/image1.jpg",
    "/path/to/your/workspace/data/images/image2.jpg",
    "/path/to/your/workspace/data/images/image3.jpg",
]
images = [Image.open(image_path).convert('RGB') for image_path in images_list]
prompts = [
    'Render a clear and concise summary of the photo.',
    'Create a compact narrative representing the image presented.',
    'Give a brief description of the image.',
]

# Generate responses
for prompt, image in zip(prompts, images):
    resp = model.generate_with_vbpe(
        [prompt], 
        [image], 
        max_new_tokens=200, 
        do_sample=False
    )[0]
    print("\n##### Prompt #####\n", prompt)
    display(image.resize((224, 224)))  # For Jupyter notebooks
    print("\n##### Our output #####\n", resp)
```
