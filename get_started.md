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
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

<<<<<<< HEAD
## Test mô hình Swin Transformer
=======
Case b: If you use mmdet as a dependency or third-party package, install it with pip:

```shell
pip install mmdet
```

## Verify the installation

To verify whether MMDetection is installed correctly, we provide some sample codes to run an inference demo.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `yolov3_mobilenetv2_320_300e_coco.py` and `yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth` in your current folder.

**Step 2.** Verify the inference demo.

Option (a). If you install mmdetection from source, just run the following command.

```shell
python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file result.jpg
```

You will see a new image `result.jpg` on your current folder, where bounding boxes are plotted on cars, benches, etc.

Option (b). If you install mmdetection with pip, open you python interpreter and copy&paste the following codes.

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'yolov3_mobilenetv2_320_300e_coco.py'
checkpoint_file = 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```

You will see a list of arrays printed, indicating the detected bounding boxes.

## Customize Installation

### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

```{note}
Installing CUDA runtime libraries is enough if you follow our best practices, because no CUDA code will be compiled locally. However if you hope to compile MMCV from source or develop other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in `conda install` command.
```

### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv-full built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### Install on CPU-only platforms

MMDetection can be built for CPU only environment. In CPU mode you can train (requires MMCV version >= 1.4.4), test or inference a model.

However some functionalities are gone in this mode:

- Deformable Convolution
- Modulated Deformable Convolution
- ROI pooling
- Deformable ROI pooling
- CARAFE
- SyncBatchNorm
- CrissCrossAttention
- MaskedConv2d
- Temporal Interlace Shift
- nms_cuda
- sigmoid_focal_loss_cuda
- bbox_overlaps

If you try to train/test/inference a model containing above ops, an error will be raised.
The following table lists affected algorithms.

|                        Operator                         |                                          Model                                           |
| :-----------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| Deformable Convolution/Modulated Deformable Convolution | DCN、Guided Anchoring、RepPoints、CentripetalNet、VFNet、CascadeRPN、NAS-FCOS、DetectoRS |
|                      MaskedConv2d                       |                                     Guided Anchoring                                     |
|                         CARAFE                          |                                          CARAFE                                          |
|                      SyncBatchNorm                      |                                         ResNeSt                                          |

### Install on Google Colab

[Google Colab](https://research.google.com/) usually has PyTorch installed,
thus we only need to install MMCV and MMDetection with the following commands.

**Step 1.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
!pip3 install openmim
!mim install mmcv-full
```

**Step 2.** Install MMDetection from the source.

```shell
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection
!pip install -e .
```

**Step 3.** Verification.

```python
import mmdet
print(mmdet.__version__)
# Example output: 2.23.0
```

```{note}
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
```

### Using MMDetection with Docker
>>>>>>> e71b499608e9c3ccd4211e7c815fa20eeedf18a2

**Step 1.** Tải file weights và config:

```shell
mim download mmdet --config mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco --dest .
```

**Step 2.** Test mô hình:

```shell
python demo/image_demo.py demo/demo.jpg mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth --device cpu --out-file result.jpg
```

Xem kết quả trong file `result.jpg`.
