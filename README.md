```
************************** miniImagenet, 5way 5shot noskip****************************
python eval.py --device cuda:1 --dataset mini --num_ways 5 --num_shots 5 --transductive True --pool_mode kn --unet_mode addold
python eval.py --device cuda:3 --dataset mini --num_ways 5 --num_shots 5 --transductive False --pool_mode kn --unet_mode addold
python eval.py --device cuda:3 --dataset mini --num_ways 5 --num_shots 1 --transductive False --pool_mode kn --unet_mode addold
```

