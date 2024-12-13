<div align="center" markdown>
<img src="https://github.com/user-attachments/assets/42f08187-a610-49a9-9377-5f49853049e0" />

# Apply Florence-2 to Images Project

Integration of the Microsoft Florence-2 model for prompt-based object detection.

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-to-Run">How to Run</a> •
  <a href="#Examples">Examples</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/apply-florence-2-to-images-project)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/apply-florence-2-to-images-project)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/apply-florence-2-to-images-project.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/apply-florence-2-to-images-project.png)](https://supervisely.com)

</div>

## Overview

Application allows you to label project images using the Florence-2 detection model with bounding boxes and utilizes the Segment Anything 2.1 model to generate object masks based on the bounding box annotations.

Application key points:

- Select project or dataset to label
- Serve models by [Serve Florence-2](https://ecosystem.supervisely.com/apps/serve-florence-2) and [Serve Segment Anything 2.1](https://ecosystem.supervisely.com/apps/serve-segment-anything-2) apps. The `Florence-2-large` and `SAM 2.1 Hiera small` models will be deployed automatically. You can change it if needed.

  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/serve-florence-2" src="https://user-images.githubusercontent.com/placeholder" height="70px" margin-bottom="20px"/>

  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/serve-segment-anything-2" src="https://user-images.githubusercontent.com/placeholder" height="70px" margin-bottom="20px"/>

- If models were deployed before the application was launched, they will be connected automatically.
- Set up model input data as text-prompt. If you left it empty - model will apply `Detailed Caption` task for every image to predict all known objects.
- Preview detection results
- Apply model to project images and save new annotations to new project or add to existed. If you only need annotations like `mask`, you can skip saving `bbox`.

## How to Run

1. Start the application from Ecosystem or context menu of an Images Project.

2. Choose your input project / dataset.

<img src="https://github.com/user-attachments/assets/7245ce39-5935-4cba-a1fb-a1cf1640ffbf" />

3. Select your served models and click `Select model` button if they have not been automatically selected.

<img src="https://github.com/user-attachments/assets/ed22476d-831b-415e-b7f8-6aa5cb83d713" />

4. Write down the **Text Prompt** that will help Florence-2 detect objects you need.

<img src="https://github.com/user-attachments/assets/f61070cb-5538-44d6-b00d-cbf56c01e416" />

5. View predictions preview by clicking according buttons.

<img src="https://github.com/user-attachments/assets/68f08021-7ef8-45bd-b4fe-de8e996cb1a2" />

6. Select the way you want to save the project and click `Apply to Project`.

<img src="https://github.com/user-attachments/assets/79c67c97-bf50-4707-a7d1-ce1602a10302" />

## Examples

`Text Prompt:` The image shows a room with tables and chairs.

<img src="https://github.com/user-attachments/assets/6d5d9100-df6f-4908-b524-207bc993c842" />
