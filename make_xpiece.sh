log_file=$1 # where to save the test results. For example: log.txt
ot_dir=$2 # The dir path to save the xpiece tokenization distribution. For example: ot_data/

export ROOT_PATH=.
trial_src=("conll")
trial_tgt=("twitter" "ontonotes" "webpage")

for source in ${trial_src[*]}
do

for target in ${trial_tgt[*]}
do

if [ $source = $target ];
then
  continue
fi

echo "----------------------- Transfering from $source to $target ----------------------------" | tee -a ${log_file}


export SOURCE=$source
export TARGET=$target
export TRAIN_DATA_DIR=${ROOT_PATH}/ner_data/conll/plo/ #/home/mrt/data/ontonotes/v4_${SOURCE}/ #/home/mrt/mlm_ner/dataset/conll/
export TEST_DATA_DIR=${ROOT_PATH}/ner_data/${TARGET}/plo/ #/home/mrt/data/ontonotes/v4_${TARGET}/ #/home/mrt/mlm_ner/dataset/wikigold_distant/
export LABEL_DIR=${ROOT_PATH}/ner_data/conll/plo/ #~/mlm_ner/dataset/conll/ #~/mlm_ner/dataset/ontonotes/
export SUBWORD_DATA_DIR=${ROOT_PATH}/$ot_dir/${SOURCE}2${TARGET}/
mkdir ${ROOT_PATH}/$ot_dir/

cd utils || return
python3 bpe_ot_new.py --label_mode plo --source_domain $SOURCE --target_domain $TARGET \
--source_path $TRAIN_DATA_DIR --target_path $TEST_DATA_DIR \
--subword_data_dir $SUBWORD_DATA_DIR \
--mode func3 \
--reg 1.0 \
--log_file $log_file \

done
done