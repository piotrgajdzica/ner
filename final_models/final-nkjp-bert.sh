#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --time=72:00:00
#SBATCH -A lemkingpu
#SBATCH --partition=plgrid-gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gajdzica@agh.edu.pl

# podać nazwę eksperymentu
# name="test-5g-d-0.05-m-p-x-ugc"
name="final-nkjp-bert"

module load plgrid/libs/openblas
module load plgrid/libs/atlas/3.10.3
module load plgrid/libs/lapack/3.8.0
module load plgrid/libs/hdf5/1.8.17

module load plgrid/apps/cuda/10.1
module load plgrid/tools/python/3.6.5

source ~/ner/venv/bin/activate

export LD_LIBRARY_PATH=/net/scratch/people/plgpgajdzica/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/net/scratch/people/plgpgajdzica/cuda/lib64:$LIBRARY_PATH
export CPATH=/net/scratch/people/plgpgajdzica/cuda/include:$CPATH

echo "Is CUDA available?"
python -c "import torch; print(torch.cuda.is_available()); print(torch.backends.cudnn.enabled)"

#time python train_tagger.py "taggers/${name}" ../../data_simplified/ -m -a  -u
time python \
 ../train_bert.py \
--data_dir /net/people/plgpgajdzica/scratch/ner/data/training_datasets/nkjp/bert \
--model_type bert \
--labels /net/people/plgpgajdzica/scratch/ner/data/embeddings/bert/slavic/labels.txt \
--model_name_or_path bert-base-multilingual-cased \
--output_dir /net/people/plgpgajdzica/scratch/ner/data/taggers/${name} \
--max_seq_length 128 \
--num_train_epochs 25 \
--per_gpu_train_batch_size 8 \
--save_steps 750 \
--seed 44 \
--do_train \
--do_eval \
--do_predict \
--learning_rate 0.000007 \
--evaluate_during_training \
