# NNCF scaling proposal based on the PyTorch Timm project
This is just an experimental project that shows the results of applying NNCF quantization to a subset of models from Timm.

## Installation
- Clone this repo
- Install dependencies
  - `pip install -r requirements.txt`
  - Clone OpenVINO [Model Analyzer](https://github.com/openvinotoolkit/model_analyzer) in the working directory.

## Running the script
- Just execute the script with the parameter that specifies a directory where the output models will be bumped:
```
nncf_timm.py ./models
```
- See the reults in the `log.txt` file or in console log