# put the downloaded dataset in the 'datas/archive' directory.
cd /root/autodl-tmp/AI_Engine/datas/archive && \
mkdir -p data/data_train data/data_test gt/gt_train gt/gt_test && \
cp -av derain_drop_dataset/train/train/data/. data/data_train/ && \
cp -av derain_drop_dataset/train/train/gt/.   gt/gt_train/ && \
cp -av derain_drop_dataset/test_a/test_a/data/. data/data_test/ && \
cp -av derain_drop_dataset/test_a/test_a/gt/.   gt/gt_test/

# use the following command to remove the __MACOSX directory if it exists.
# find /root/autodl-tmp/AFDNet/datas/archive/derain_drop_dataset -type d -name "__MACOSX" -exec rm -rf {} +

# use the following command to remove the original downloaded dataset if you don't need it anymore.
# cd /root/autodl-tmp/AFDNet/datas/archive && \
# echo "train data: $(ls data/data_train | wc -l)" && \
# echo "train gt:   $(ls gt/gt_train | wc -l)" && \
# echo "test data:  $(ls data/data_test | wc -l)" && \
# echo "test gt:    $(ls gt/gt_test | wc -l)"