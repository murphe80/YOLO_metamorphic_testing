# YOLO_metamorphic_testing
Repository to run various metamorphic tests on YOLOv5, YOLOv8 and YOLOv12 for the purpose of analysing performance changes in the context of Formula Trinity.

## Troubleshooting

### NumPy Compatibility Issues

If you encounter the error:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash.
```

**Solution:**
1. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Downgrade NumPy to a compatible version:
   ```bash
   pip install "numpy<2.0" --force-reinstall
   ```

3. Reinstall OpenCV to ensure compatibility:
   ```bash
   pip uninstall opencv-python opencv-python-headless opencv-contrib-python-headless -y
   pip install opencv-python==4.8.1.78
   ```

### PyTorch Model Loading Issues

If you encounter the error:
```
WeightsUnpickler error: Unsupported global: GLOBAL models.yolo.DetectionModel was not an allowed global by default.
```

**Solution:**
This is due to PyTorch 2.6+ security changes. You need to modify the YOLOv5 experimental.py file:

1. Navigate to the YOLOv5 installation in your virtual environment:
   ```bash
   find venv -name "experimental.py" -path "*/yolov5/models/*"
   ```

2. Edit the file and find the `torch.load` call (around line 82):
   ```python
   # Change this line:
   ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
   
   # To this:
   ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)  # load
   ```

3. Save the file and try running your script again.

```
### Verification

After applying the fixes, test with:
```bash
python models/yolov5/infer_v5.py --weights models/yolov5/yolov5_parameters.pt --image ./data/fsoco_sample/images/amz_00016.jpg --conf_thres 0.25 --iou_thres 0.45
```
