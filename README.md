# Chuẩn bị môi trường: 


**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Tạo và kích hoạt môi trường Conda.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Cài PyTorch: 

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# Cài đặt các thư viện của mmlab:

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install MMDetection.

```shell
git clone https://github.com/hungvm2/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

## Test mô hình Swin Transformer

**Step 1.** Tải file weights và config:

```shell
mim download mmdet --config mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco --dest .
```

**Step 2.** Test mô hình:

```shell
python demo/image_demo.py demo/demo.jpg mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth --device cpu --out-file result.jpg
```

Xem kết quả trong file `result.jpg`.
