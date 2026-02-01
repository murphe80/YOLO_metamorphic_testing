MODELS="/moredata/emma/meta/YOLO_metamorphic_testing/training/kfold_results/v8"
FOLDS=5
TEST_DATA="/moredata/emma/meta/YOLO_metamorphic_testing/training/dataset/split/dataset.yml"

i=1
while [ $i -le $FOLDS ]; do
  yolo detect val \
    model="$MODELS/fold_$i/weights/best.pt" \
    data="$TEST_DATA" \
    imgsz=640 \
    batch=16
  i=$((i+1))
done
