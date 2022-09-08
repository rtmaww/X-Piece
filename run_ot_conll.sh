log_file=$1  #
ot_dir=$2
gpu=$3
export MAX_LENGTH=200
export BERT_MODEL=bert-base-cased
export ROOT_PATH=.
#python3 scripts/preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt
#python3 scripts/preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > dev.txt
#python3 scripts/preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > test.txt
#cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
# domain_list=("bn" "bc" "mz" "nw" "tc" "wb")
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

export OUTPUT_DIR=model_${source}2${target}/
export BATCH_SIZE=64
export NUM_EPOCHS=3
export SAVE_STEPS=5000
export SEED=42
export SOURCE=$source
export TARGET=$target
export TRAIN_DATA_DIR=${ROOT_PATH}/ner_data/conll/plo/ #/home/mrt/data/ontonotes/v4_${SOURCE}/ #/home/mrt/mlm_ner/dataset/conll/
export TEST_DATA_DIR=${ROOT_PATH}/ner_data/${TARGET}/plo/ #/home/mrt/data/ontonotes/v4_${TARGET}/ #/home/mrt/mlm_ner/dataset/wikigold_distant/
export LABEL_DIR=${ROOT_PATH}/ner_data/conll/plo/ #~/mlm_ner/dataset/conll/ #~/mlm_ner/dataset/ontonotes/
export SUBWORD_DATA_DIR=${ROOT_PATH}/$ot_dir/${SOURCE}2${TARGET}/
mkdir ${ROOT_PATH}/$ot_dir/

CUDA_VISIBLE_DEVICES=$gpu python3 run_ner_ot.py \
--src $SOURCE \
--tgt $TARGET \
--task_type NER \
--train_data_dir $TRAIN_DATA_DIR \
--test_data_dir $TEST_DATA_DIR \
--labels $LABEL_DIR/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--log_file $log_file \
--tokenize ot \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--subword_data_dir $SUBWORD_DATA_DIR \
--save_steps $SAVE_STEPS \
--seed $SEED \
--overwrite_output_dir \
--overwrite_cache \
--do_train \
--do_eval \
--do_predict

done
done