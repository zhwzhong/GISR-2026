# GuidedSR-2026

## Environment

The project is implemented using **PyTorch (>= 2.10)**.

We recommend creating the environment using the provided `environment.yaml` file.

### Install with Conda

```bash
conda env create -f environment.yaml
conda activate guidedsr
```

Alternatively, you can manually install the main dependency:

```bash
pip install torch>=2.10
```

---

## Training and Testing

### Training

Example training command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
--nnodes 1 \
--nproc_per_node=4 \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:11342 \
main.py \
--scale 8/16 \
--model_name Base3 \
--num_gpus 4 \
--embed_dim 64 \
--opt Adam \
--file_name File \
--dataset NIR \
--batch_size 8 \
--patch_size 256 \
--loss '1*L1'
```

---

### Testing

Example testing command:

```bash
python main.py \
--test_only \
--load_name your_path/model_x8_or_x16.pth \
--scale 8/16
```

---

## Pre-trained Models and Dataset

The pre-trained models and test dataset can be downloaded from:

- **8× Super-Resolution Model**  
  https://pan.baidu.com/s/1ZtJgF8a2yxetaYMufrXREw?pwd=GISR

- **16× Super-Resolution Model**  
  https://pan.baidu.com/s/1y-cbYSCb-RvH-ZFB5YljxA?pwd=myvz

- **Test Dataset**  
  https://pan.baidu.com/s/1-oj21cz309ngfFD4C2parw?pwd=wmxc
