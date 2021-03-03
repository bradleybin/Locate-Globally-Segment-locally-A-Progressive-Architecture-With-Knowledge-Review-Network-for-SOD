Code will be released soon.

# Locate-Globally-Segment-locally-A-Progressive-Architecture-With-Knowledge-Review-Network-for-SOD
This repository is the official implementation of PA-KRN and SGL-KRN, which is proposed in "Locate Globally, Segment locally: A Progressive Architecture With Knowledge Review Network for Salient Object Detection."

![image](https://user-images.githubusercontent.com/42328490/109591578-ba656100-7b48-11eb-8419-d258e20ed9d0.png)

## Prerequisites
- Python 3.6
- PyTorch 0.4.1+
- torchvision
- numpy
- visdom

## Usage
### 1. Download datasets
Download the ` DUTS`  and other datasets and unzip them into `demo/data` folder.
The directory shows as follow:
```bash
├─DUTS
│        └── DUTS-TR
│                  ├── DUTS-TR-Image
│                  ├── DUTS-TR-Mask
│                  └── DUTS-TR-Edge
├─DUTS-TE
│        ├── Imgs
│        └── test.lst
├─PASCALS
│        ├── Imgs
│        └── test.lst
├─DUTOMRON
│        ├── Imgs
│        └── test.lst
├─HKU-IS
│        ├── Imgs
│        └── test.lst
└─ECSSD
          ├── Imgs
          └── test.lst
```
### 2. Download Pretrained ResNet-50 Model for backbone
Download ResNet-50 pretrained models and save it into `demo/dataset/pretrained` folder

### 3. Install body-atttention sampler related tools (MobulaOP)
```bash
# Enter the directory
cd MobulaOP

# Install MobulaOP
pip install -v -e .
```

The directory shows as follow:
```bash
├─demo
│   ├── attention_sampler
│   ├── data
│   ├── dataset
│   ├── networks
│   ├── results
│   ├── KRN.py
│   ├── KRN_edge.py
│   ├── main_clm.py
│   ├── main_fsm.py
│   ├── main_joint.py
│   ├── Solver_clm.py
│   ├── Solver_fsm.py
│   └── Solver_joint.py
├── MobulaOP
```

### 4. Train
The whole system can be trained in an end-to-end manner. To get finer results, we first train CLM and FSM sequentially and then combine them to fine-tune. 
```
cd demo
```
1. Train CLM. We use ResNet-50 as network backbone and train CLM with an initial lr of 5e-5 for 24 epochs, which is divided by 10 after 15 epochs.
```
python main_clm.py
```
After training, the resulting model will be stored under results/clm/run-* folder.

2. Train FSM. Based on pretrained CLM with fixed parameters, we use ResNet-50 as network backbone and train FSM with an initial lr of 5e-5 for 24 epochs, which is divided by 10 after 15 epochs.
```
python main_fsm.py  --clm_model path/to/pretrained/clm-model/folder/
```
After training, the resulting model will be stored under results/fsm/run-* folder.

3. Train PA-KRN. Based on pretrained CLM and FSM, we fine-tune the whole system with an initial lr of 5e-6 for 15 epochs, which is divided by 10 after 9 epochs.
```
python main_joint.py  --clm_model path/to/pretrained/clm-model/folder/  --fsm_model path/to/pretrained/fsm-model/folder/
```
After training, the resulting model will be stored under results/joint/run-* folder. 'net_\*.pth' is the parameter of CLM model and '.pth' is the parameter of FSM model.


### 5. Test
For DUTS-TE dataset testing.
```
python main.py --mode test --clm_model path/to/pretrained/clm-model/folder/  --fsm_model path/to/pretrained/fsm-model/folder/ --test_fold path/to/test/folder/ --sal_mode e
```
All results saliency maps will be stored under results/run-*-sal-* folders in .png formats. For testing other datasets, download them and unzip them into `data` folder, and test them by the same steps. 'sal_mode' of ECSSD, PASCALS, DUT-OMRON, and HKU-IS are 'e', 'p', 'd', and 'h', respectively.


### 6. Saliency maps
[Baidu Disk](https://pan.baidu.com/s/1pKE4K8bckxgvttO4rgjEBw) (9wxg) or [Google Drive](https://drive.google.com/drive/folders/1crvlMRp5oBNHs3zJ9kEYJREfw4ZjxnQm?usp=sharing)
