# Distributed Computing Performance Study for Semantic Segmentation

## Members
- Karthik Jayarama (pennkey: jkarthik)
- Shantanu Sampath (pennkey: shantz)
- Naomi Maranga (pennkey: kmaranga)
- Riju Datta (pennkey: dattarij)

## Description
Data parallelisation replicates the model on every computational resource to generate gradients independently which are then communicated across models iteratively. In our project, we compare different models and training algorithms to optimise training. Data parallelism allows distributed training by communicating gradients before the optimiser step ensuring consistency across model replicas. Semantic segmentation on the other hand assigns labels to image pixels and is useful in vision-based applications. In our project, we specifically work with the ADE20K(2016) dataset and perform semantic segmentation on images therein using three models on either single or multi GPUs, namely: PSPNet, UPerNet, and DeepLabV3 and compare their performance.

## Installation
1. Download the ADEK2016 dataset from https://groups.csail.mit.edu/vision/datasets/ADE20K/ and copy the contents to folder name 'ADEChallengeData2016' in the parent directory of this repository. Make sure the folder name is exactly as given.
2. Install the following packages. You can also refer to 'requirements.txt' for the full list of dependencies.
   ```bash
   pip install boto3==1.33.11
   pip install matplotlib==3.8.2
   pip install numpy==1.26.2
   pip install opencv-python==4.8.1.78
   pip install Pillow==10.1.0
   pip install scikit-learn==1.3.2
   pip install scipy==1.11.4
   pip install torch==2.1.1
   pip install torchvision==0.16.1
   pip install tqdm==4.66.1
   ```

## Usage
Navigate to the main directory. The entry point script `train.py` allows you to train a PyTorch model for your project. You can configure various aspects of the training process using command-line arguments.

### Command-Line Arguments
- **-c, --config**
  - **Description:** Path to the config file.
  - **Default Value:** `config.json`
  - **Example Usage:** 
    ```bash
    python train.py -c my_config.json
    ```

- **-d, --device**
  - **Description:** Indices of GPUs to enable for training. By default, all available GPUs are used.
  - **Default Value:** `None` (i.e., all GPUs)
  - **Example Usage:** 
    ```bash
    python train.py -d 0 1 2
    ```

- **-p, --parallel**
  - **Description:** Specifies the parallelization algorithm to use for training (e.g., dp for DataParallel, ddp for DistributedDataParallel).
  - **Default Value:** `None` (i.e., Single GPU)
  - **Example Usage:** 
    ```bash
    python train.py -p dp
    ```

- **-m, --model**
  - **Description:** Choose the model architecture for training. Options include Deeplab, PSP, and UperNet.
  - **Default Value:** `None` (Specify the model in your config file)
  - **Example Usage:** 
    ```bash
    python train.py -m PSPnet
    ```

### Running the Script
To run the script, use the following command as an example:
```bash
python train.py -c config.json -d 0 1 -p ddp -m PSPnet
```
The above will train the PSPnet model with distributed data parallel mode. 

## License
This project is licensed under the MIT License

### MIT License

Copyright (c) 2023 ESE 546 group 99

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Contact
Please reach out to any of us if you have any questions or clarifications:
- shantz@seas.upenn.edu
- jkarthik@seas.upenn.edu
- kmaranga@seas.upenn.edu
- dattarij@seas.upenn.edu
