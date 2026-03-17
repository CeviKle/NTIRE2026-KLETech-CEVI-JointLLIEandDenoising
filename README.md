# MST++ for Joint Denoising and Low-Light Image Enhancement

---

# ENVIRONMENT SETUP

## 1. Create Conda Environment

```bash
conda create -n mst python=3.8
conda activate mst
```

---

## 2. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

## 3. Install PyTorch

```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 \
-f https://download.pytorch.org/whl/torch_stable.html
```

---

## 4. Install Required Dependencies

```bash
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

---

## 5. Install BasicSR

```bash
python setup.py develop --no_cuda_ext
```

---

# DATASET PREPARATION

Organize the NTIRE dataset as follows:

```
/NTIRE2026/C8_JointDenoisingLLIE/
    ├── Training
    │   ├── inputPatchNLL
    │   └── gtPatchNLL
    ├── Validation
    │   ├── val_in
    │   └── val_gt
```

---

# TRAINING

Train the MST++ model using:

```bash
python basicsr/train.py --opt options/MST_Plus_Plus_NTIRE_8x1150.yml
```

---

# TESTING

To run testing on the NTIRE dataset:

```bash
python Enhancement/test_from_dataset.py \
--opt options/MST_Plus_Plus_NTIRE_8x1150.yml \
--weights path_to_trained_model.pth \
--dataset NTIRE
```

