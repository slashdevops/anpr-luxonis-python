# anpr-luxonis-python

ANPR using Luxonis camera

## Requirements

Required

+ Python 3.9
+ opencv-python==4.6.0.66
+ depthai==2.17.3.1
+ blobconverter==1.3.0
+ depthai-sdk==1.2.3

Optionals

+ openvino-dev==2021.4.582

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

## Run

```bash
python3 main.py -nn ~/Downloads/frozen_inference_graph_openvino_2021.4_6shave.blob -vid ~/Videos/1.mp4
```

## Pretrained models

Luxonis zoo

+ [luxonis / blobconverter -> 2022_1](https://github.com/luxonis/blobconverter/tree/master/models/2022_1)
+ [Overview of OpenVINO™ Toolkit Intel’s Pre-Trained Models](https://docs.openvino.ai/latest/omz_models_group_intel.html)
+ [doxid-omz-demos-text-detection-demo-cpp](https://docs.openvino.ai/latest/omz_demos_text_detection_demo_cpp.html#doxid-omz-demos-text-detection-demo-cpp)

## Utils

+ [blobconverter](http://blobconverter.luxonis.com/)
+ [Converting model to MyriadX blob](https://docs.luxonis.com/en/latest/pages/model_conversion/#converting-model-to-myriadx-blob)

## Converters

### OpenVINO to luxonis

Using `converter.py` script

```bash
python3 converter.py \
  -bin /home/christian/ai/openvino/intel/text-recognition-0014/FP16/text-recognition-0014.bin \
  -xml /home/christian/ai/openvino/intel/text-recognition-0014/FP16/text-recognition-0014.xml \
  -s 6
```

Using `blobconverter` tool

```bash
blobconverter \
  --openvino-bin "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/text-recognition-0014/FP16/text-recognition-0014.bin" \
  --openvino-xml "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/text-recognition-0014/FP16/text-recognition-0014.xml" \
  --shaves 6 \
  --no-cache
```

```bash
blobconverter \
  --zoo-name text-recognition-0014 \
  --shaves 6
```
