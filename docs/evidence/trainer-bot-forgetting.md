# Unseen Q&A: 200 questions, none of them trained on (seed 777)

## Untrained checkpoint (runs/cpu, before any trainer-bot step)
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
total        107/200    54%  (unseen questions, 3001403 in training db)
```

## After 1,480 trainer-bot steps on level 1 with replay filtered to that level
```
arithmetic     1/63      2%
code           2/80      2%
dates          0/17      0%
factorise      0/1       0%
gcd_lcm        2/11     18%
linear         0/1       0%
percent        0/1       0%
primality      6/6     100%
syllogism      0/14      0%
system2        0/6       0%
total         11/200     6%  (unseen questions, 3001403 in training db)
```

Training one level while replaying only that level cost the model everything
else it knew. This is why replay now ignores the curriculum level; see
[../trainer-bot.md](../trainer-bot.md).

## Third regime: batch 8, one pass per batch, lr 1e-5, 1,500 steps

```
arithmetic     4/63      6%
code          80/80    100%
dates          2/17     12%
factorise      0/1       0%
gcd_lcm        1/11      9%
linear         0/1       0%
percent        0/1       0%
primality      6/6     100%
syllogism     14/14    100%
system2        0/6       0%
total        107/200    54%  (unseen questions, 3008665 in training db)
```

Per kind against the untrained checkpoint on the same questions: arithmetic
3/63 -> 4/63, dates 1/17 -> 2/17, gcd_lcm 3/11 -> 1/11, and code 80/80,
syllogisms 14/14, primality 6/6 all unchanged. Total 107/200 both ways. This is
the regime the defaults use: it does not damage the model.
