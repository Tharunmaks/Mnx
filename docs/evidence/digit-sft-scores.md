# Raw score sheets for the digit-aligned run, 2026-10-26

Every number in docs/sft-digits.md comes from one of these, pasted as the scorer
printed it. Nothing here is rounded or reconstructed.

## The protocol, n=200, seed 777

### Untrained `runs/cpu`, no exclusion (run twice, identical both times)
```
arithmetic     3/63      5%
code          80/80    100%
dates          1/17      6%
factorise      0/1       0%
gcd_lcm        3/11     27%
linear         0/1       0%
percent        0/1       0%
primality      6/6     100%
syllogism     14/14    100%
system2        0/6       0%
total        107/200    54%  (unseen questions, 3000000 in training db)
```

### Untrained `runs/cpu`, under the identical `--exclude` as the trained run
```
arithmetic     3/62      5%
code          82/82    100%
dates          1/16      6%
factorise      0/1       0%
gcd_lcm        3/12     25%
percent        0/2       0%
primality      6/6     100%
syllogism     14/14    100%
system2        0/5       0%
total        109/200    55%  (unseen questions, 3992037 in training db)
```

### After 9,000 steps, `runs/digit`, excluding all 1,176,499 trained questions
```
arithmetic    26/62     42%
code          82/82    100%
dates          0/16      0%
factorise      0/1       0%
gcd_lcm        3/12     25%
percent        0/2       0%
primality      6/6     100%
syllogism     14/14    100%
system2        0/5       0%
total        131/200    66%  (unseen questions, 3992037 in training db)
```

## Dev checks, n=150, seed 4242

### Untrained
```
arithmetic     1/51      2%
code          53/53    100%
dates          0/12      0%
factorise      0/4       0%
gcd_lcm        8/12     67%
linear         0/1       0%
percent        0/2       0%
primality      3/4      75%
syllogism      7/7     100%
system2        0/4       0%
total         72/150    48%  (unseen questions, 3000000 in training db)
```

### Step 3,000
```
arithmetic    12/51     24%
code          53/53    100%
dates          0/12      0%
factorise      1/4      25%
gcd_lcm        7/12     58%
linear         0/1       0%
percent        0/2       0%
primality      3/4      75%
syllogism      7/7     100%
system2        0/4       0%
total         83/150    55%  (unseen questions, 3328023 in training db)
```

### Step 6,000
```
arithmetic    16/51     31%
code          53/53    100%
dates          0/12      0%
factorise      0/4       0%
gcd_lcm        7/12     58%
linear         0/1       0%
percent        0/2       0%
primality      3/4      75%
syllogism      7/7     100%
system2        0/4       0%
total         86/150    57%  (unseen questions, 3661090 in training db)
```

## The training log

```
step,lr,loss,items,answer_tokens,elapsed_s
1,1.000000e-06,7.557110,133,1037,30.4
1000,1.963634e-04,0.962714,129557,1061879,2389.6
2000,1.820680e-04,0.734780,257711,2112230,3484.8
3000,1.586847e-04,0.546531,385247,3158136,4579.0
4000,1.291623e-04,0.442012,122581,992572,1039.1
5000,9.722347e-05,0.402548,255018,2066327,2170.6
6000,6.689579e-05,0.402163,387278,3137924,3286.3
7000,4.200358e-05,0.368695,518772,4208231,4501.6
8000,2.568576e-05,0.425517,650282,5281956,5644.0
9000,2.000001e-05,0.386435,780223,6351438,6744.5
```

The two series in sft_log.csv restart their elapsed clock at step 3,088: the run
was stopped there to pull the dates kill switch and resumed from the same
checkpoint and optimizer state with the dates share reallocated.
