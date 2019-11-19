#!/bin/bash
#SBATCH --job-name "glue infer"
#SBATCH --ntasks 1
#SBATCH --qos long
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --mem=20GB
#SBATCH -o glue.out
#SBATCH -e glue.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fan.jinzhen@gene.com
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1 
export BERT_BASE_DIR=~/virtualenvs/BERT/forked-repo/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=~/virtualenvs/BERT/data/glue
export TRAINED_CLASSIFIER=~/virtualenvs/BERT/forked-repo/bert/uncased_L-12_H-768_A-12

python3 run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=~/virtualenvs/BERT/tmp/mrpc_output/