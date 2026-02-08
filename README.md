## ChimpRec

This repository is the source code related to the Master thesis project realised by Théodore Cousin and Julien Demeure studying at UCLouvain (Belgium).

This thesis presents the development of a computer vision architecture to automatically detect, track, and recognise individual chimpanzees in video footage. The project was designed to support the doctoral research of Calogero Montedoro, who studies how environmental conditions influence the social development of chimpanzees. By replacing manual identification with an automated system, the goal is to make behavioural studies more scalable and less time-consuming. The architecture is modular and combines body and face detection, identity recognition, and object tracking. We used deep learning models such as YOLOv8 and InceptionResNetV1, which were adapted to the specific visual challenges posed by chimpanzee imagery. A correction tool was also developed to let users manually fix tracking errors and assign identities when the models are uncertain. Although some limitations remain, especially in the recognition step, the system performs well overall and can already be used in a semi-automated way. Corrected videos can also be reused to train better models in the future.

The following is the structure of the project. The central library, `chimplib`, serves as the core module used across the entire codebase. All other components in the project rely on and import functions from this library.

```plaintext
Code/
├── Body_detection/
│   └── Metric/
├── chimplib/
├── Face_detection/
├── Final pipeline/
├── recognition/
├── Tracking/
│   └── Manual Correction/
Models/
├── Body Detection/
├── Face Detection/
└── Facial Recognition/
```

`Code/Tracking/Manual Correction/` contains a framework directly usable to correct the output of the body detection and tracking stages in the architectures. It allows users to take profit from the work already performed eventhough the whole architecture isn't functional yet.

`Models` contains all the trained models that we came up with. For the body and face detection tasks, `YOLOv8s` is the model offering the best performance. In the case of facial recognition, the fully connected approach demonstrated better results.

All the dependencies necessary are listed in the file `./requirements.txt`. To execute the code. Create a python virtual environment, activate it, and install all the dependencies:

**For Windows**

```plaintext
$python -m venv <venv name="">
$./<venv name="">/Scripts/activate 
$pip install -r Code/requirements.txt
```

**For Linux/MacOS**

```plaintext
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r Code/requirements_linux.txt

```

![Alt text](./cover.webp)

## License

This project is licensed under the [MIT License](LICENSE).