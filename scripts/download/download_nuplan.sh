# NOTE: Please check the LICENSE file when downloading the nuPlan dataset
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE

# maps
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip

# train
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_boston.zip
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_pittsburgh.zip
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_singapore.zip
for split in {1..6}; do
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_vegas_${split}.zip
done

# val
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip

# test
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_val.zip

# mini
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_mini.zip