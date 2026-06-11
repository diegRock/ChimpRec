# ChimpRec

This repository contains the source code for a Master thesis project realised by Théodore Cousin and Julien Demeure at UCLouvain (Belgium).

The project develops a modular computer vision pipeline to detect, track, and recognise individual chimpanzees in video footage. It combines body detection, face detection, facial recognition, and tracking, with a manual correction tool for refining outputs where model performance is uncertain.

The central codebase is located in `Code/`, while the trained models are stored in `Models/`.

## Repository structure

```
ChimpRec/
├── ChimpPic/
├── ChimpVideos/
├── Code/
│   ├── Body_detection/
│   ├── Face_detection/
│   ├── FinalPipeline/
│   ├── Metric/
│   ├── PostProcessing/
│   ├── PostProcessingDualHeuristic/
│   ├── Recognition/
│   ├── Tracking/
│   ├── chimplib/
│   ├── requirements.txt
│   └── requirements_linux.txt
├── Models/
│   ├── Body Detection/
│   ├── Face Detection/
│   ├── Face Recognition/
|   └── Re-ID/
```

### Key components

- `Code/chimplib/` - core library shared across the repository.
- `Code/Body_detection/` - body detector training, evaluation, and inference.
- `Code/Face_detection/` - face detector training, evaluation, and inference.
- `Code/Recognition/` - face recognition and identity assignment.
- `Code/Tracking/` - tracking pipeline and manual correction utilities.
- `Code/FinalPipeline/` - end-to-end pipeline integration and final orchestration.
- `Code/Metric/` - evaluation metrics and performance analysis.
- `Code/PostProcessing/` and `Code/PostProcessingDualHeuristic/` - result refinement and heuristic-based post-processing.

### Models

- `Models/Body Detection/` - body detection model weights.
  - Includes `Body_detection_model`, the best YOLO_v8s body detector for chimpanzee body detection.
- `Models/Face Detection/` - face detection model weights.
  - Includes `yolox_best_only_model`, the face detector downloaded from the [ChimpUFE](https://www.robots.ox.ac.uk/~vgg/research/ChimpUFE/) GitHub repository used to detect chimpanzee faces.
- `Models/Face Recognition/` - facial recognition model weights.
  - Includes `facenet_16_layers_fc.pth`, the face recognition model from last year.
  - Includes `ChimpUFE_finetuned`, our fine-tuned ChimpUFE model trained specifically for the 20 chimpanzee identities in this project (perform the best).
  - Includes `25-08-29T11-49-28_340k`, the DINOv2-based ChimpUFE model for universal chimpanzee face embedding backbone downloaded from the [ChimpUFE](https://www.robots.ox.ac.uk/~vgg/research/ChimpUFE/) GitHub repository.
- `Models/Re-ID/` - OSnet appearance re-identification model weights tracker that use reidentification (DeepSORT & StrongSORT)

## Data and model downloads

To run the project correctly, download the `Models` folder (1.54 Go), `ChimpPic` (4.66 Go), and `ChimpVideos`(78.46 Go) directories as shown in the repository tree from the project material:

[Project Material](https://uclouvain-my.sharepoint.com/:f:/g/personal/thomas_rixen_student_uclouvain_be/IgAM7kZZgnqbToKoIta_q-2PAfRTU7FuuAJeQOcZYRXypFc?e=exYcsi)

All models were trained on the Lyra supercomputer. They are large, heavy, and optimized for CUDA and high-end GPUs, so please consider this when reproducing the results.

## Setup

Create and activate a Python virtual environment, then install dependencies. We highly recommand to use Python3.10. To avoid depency conflict within library versions.

On Windows
```bash
python3.10 -m venv .venv
.\.venv\Scripts\activate
pip install -r Code/requirements.txt
```

On Linux
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r Code/requirements.txt
```

## Acknowledgements

- Computational resources have been provided by the Consortium des Équipements de Calcul Intensif (CÉCI), funded by the Fonds de la Recherche Scientifique de Belgique (F.R.S.-FNRS) under Grant No. 2.5020.11 and by the Walloon Region.
- This project draws on the work of several open-source repositories and research projects, including:
  - [ChimpUFE](https://www.robots.ox.ac.uk/~vgg/research/ChimpUFE/) for universal chimpanzee face embedding research.
  - [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking.
  - [DeepSORT](https://github.com/nwojke/deep_sort) for appearance-aware multi-object tracking.
  - [StrongSORT](https://github.com/ultralytics/ultralytics) for robust multi-object tracking enhancements.
  - [OSNet](https://github.com/KaiyangZhou/deep-person-reid) for lightweight person re-identification backbone architecture.
  - [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for object detection.
  - [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection and model development.

## License

This project is licensed under the [MIT License](LICENSE).
