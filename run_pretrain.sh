epochs=3
bs=64
lr=0.00003
pvalue=0.1
seed=11711
option='finetune'
name='simcse_pretraining'
gradclip='False'
gradsurgery='False'
# checkpoint='./mtdnn_new-finetune-4-1e-05-8-0.1-multitask.pt'
# checkpoint='averagepooling-finetune-4-1e-05-16-0.1-multitask.pt'
checkpoint=None
method='simcse'
# python3 multitask_classifier.py --modelname $name --option $option --use_gpu --epochs $epochs --lr $lr --batch_size $bs --hidden_dropout_prob $pvalue --seed $seed 
python3 multitask_classifier.py  --checkpoint $checkpoint --grad_clipping  --method $method --modelname $name --option $option --use_gpu --epochs $epochs --lr $lr --batch_size $bs --hidden_dropout_prob $pvalue --seed $seed 
