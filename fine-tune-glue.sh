#!/bin/bash
#SBATCH --job-name "bert glue"
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
ml python3
ml spider CUDA
#installed tf 1.11.0 into virtualenv BERT 
source ../../bin/activate
time mpirun python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=~/virtualenvs/BERT/output/mrpc_output/
deactivate