epochs=3
bs=16
lr=0.00001
pvalue=0.5
seed=11711
option='finetune'
name='multiple-negatives_lr-05'

python3 multitask_classifier.py --modelname $name --option $option --use_gpu --epochs $epochs --lr $lr --batch_size $bs --hidden_dropout_prob $pvalue --seed $seed
