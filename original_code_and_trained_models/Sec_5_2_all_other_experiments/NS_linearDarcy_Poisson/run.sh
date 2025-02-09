

for seeds in 0 1 2 3 4 5 6 7 8 9
do
    CUDA_VISIBLE_DEVICES=2 python3.12 train.py ns_high_r FNO $seeds
done
