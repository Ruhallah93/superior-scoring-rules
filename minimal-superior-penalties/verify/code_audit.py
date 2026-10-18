import numpy as np, math
# Faithful numpy transcription of the released superior_scoring_rules.pbs
def pbs_published(y, q):
    y = y.astype(np.float64); c = y.shape[1]
    ST = q - np.sum(np.where(y == 1, q, y), axis=1)[:, None]
    ST = np.where(ST < 0, 0.0, ST)
    payoff = np.sum(np.ceil(ST), axis=1)
    M = (c - 1) / c
    payoff = np.where(payoff > 0, M, payoff)
    brier = np.mean(np.square(y - q), axis=1)      # MEAN form
    return np.mean(brier + payoff), brier, payoff

def pbs_definition6(y, q):
    c = y.shape[1]
    wrong = np.argmax(q, 1) != np.argmax(y, 1)
    return np.sum(np.square(y - q), 1) + np.where(wrong, (c - 1) / c, 0.0)

for c in [3, 13]:
    y = np.eye(c)[[1, 1]]
    q = np.zeros((2, c)); q[0, 1] = 0.34; q[0, 0] = 0.33; q[0, 2] = 0.33
    q[1, 0] = 0.51; q[1, 1] = 0.49
    if c > 3: q[0] = np.full(c, 1/c); q[1] = 0; q[1,0] = 0.51; q[1,1] = 0.49
    tot, brier, payoff = pbs_published(y, q)
    d6 = pbs_definition6(y, q)
    per = brier + payoff
    print(f"c={c}")
    print(f"  published per-sample : {np.round(per,4)}   (x c -> {np.round(per*c,4)})")
    print(f"  Definition 6         : {np.round(d6,4)}")
    print(f"  implied sum-form penalty from published = c*(c-1)/c = {c-1}")
    print(f"  tie handling: ceil(0)=0 -> exact argmax tie counted CORRECT: "
          f"{np.ceil(0.0)==0}")
