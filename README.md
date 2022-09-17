# anpr-luxonis-python

ANPR using Luxonis camera

## Requirements

+ Python 3.9
+ opencv-python==4.6.0.66
+ depthai==2.17.3.1
+ blobconverter==1.3.0
+ depthai-sdk==1.2.3

### developer tools

+ black
+pylint

## Setup

clone the repo

```bash
git clone git@github.com:slashdevops/anpr-luxonis-python.git
```

create a virtual environment and activate it

```bash
python3.9 -m venv venv

source venv/bin/activate
pip install --upgrade pip
```

install the dependencies

```bash
pip install -r requirements.txt
```

## Pretrained models

Luxonis zoo

+ [luxonis / blobconverter -> 2022_1](https://github.com/luxonis/blobconverter/tree/master/models/2022_1)
+ [Overview of OpenVINO™ Toolkit Intel’s Pre-Trained Models](https://docs.openvino.ai/latest/omz_models_group_intel.html)
