

for seeds in 0 1 2 3 4 5 6 7 8 9
do
    python3.12 train.py --which_example ns_high_r --which_model CRNO --seed $seeds
done
