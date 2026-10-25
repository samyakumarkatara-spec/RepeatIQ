"""Synthetic grocery data with a PLANTED, KNOWN causal effect of quality issues (-0.6 logit on being active)
and realistic confounders (big baskets & late deliveries raise both complaint risk and affect retention).
Usage: python generate_data.py [--pg postgresql://user:pw@host/db]"""
import sys, numpy as np, pandas as pd
ASOF = pd.Timestamp("2026-10-26")
TEXT = {"quality": ["Spinach was wilted", "Milk close to expiry", "Bruised fruit", "Eggs were cracked"],
        "late_delivery": ["Order arrived 2 hours late", "Delivery slot missed"],
        "missing_item": ["Paneer missing from bag", "Substituted item not what I wanted"],
        "price": ["Prices higher than local store", "Charged more than app price"]}

def generate(n=6000, seed=7):
    r = np.random.default_rng(seed)
    signup = ASOF - pd.to_timedelta(r.integers(0, 150, n), unit="D")
    basket = r.lognormal(6.3, 0.45, n)                       # first basket value (INR)
    z = (np.log(basket) - 6.3) / 0.45
    late1 = r.random(n) < 0.12 + 0.05 * (z > 0.5)
    p_q = np.clip(0.05 + 0.06 * (z > 0.5) + 0.05 * late1, 0, 1)   # confounded exposure
    q1 = r.random(n) < p_q
    gap = r.lognormal(np.log(12), 0.5, n)                    # latent shopping interval (days)
    logit = -0.4 + 0.35 * z - 0.5 * late1 - 0.6 * q1         # true effect of q1 = -0.6
    active = r.random(n) < 1 / (1 + np.exp(-logit))
    orders, tickets = [], []
    oid = tid = 0
    for i in range(n):
        t, k = signup[i], 0
        while True:
            oid += 1
            late = bool(late1[i]) if k == 0 else r.random() < 0.12
            amt = basket[i] if k == 0 else basket[i] * r.lognormal(0, 0.25)
            orders.append((oid, i + 1, t.normalize(), round(amt, 2), late))
            cats = []
            if k == 0 and q1[i]: cats.append("quality")
            if k > 0 and r.random() < 0.04: cats.append("quality")
            if late and r.random() < 0.5: cats.append("late_delivery")
            if r.random() < 0.03: cats.append("missing_item")
            if r.random() < 0.02: cats.append("price")
            for c in cats:
                tid += 1
                tickets.append((tid, i + 1, oid, (t + pd.Timedelta(days=int(r.integers(0, 2)))).normalize(), c,
                                TEXT[c][r.integers(len(TEXT[c]))]))
            k += 1
            if not active[i]: break
            t = t + pd.Timedelta(days=float(gap[i] * r.lognormal(0, 0.3)))
            if t > ASOF: break
    return (pd.DataFrame({"customer_id": range(1, n + 1), "signup_date": signup.normalize()}),
            pd.DataFrame(orders, columns=["order_id", "customer_id", "order_date", "amount", "late"]),
            pd.DataFrame(tickets, columns=["ticket_id", "customer_id", "order_id", "created_date", "category", "text"]))

if __name__ == "__main__":
    c, o, t = generate()
    if "--pg" in sys.argv:
        from sqlalchemy import create_engine
        e = create_engine(sys.argv[sys.argv.index("--pg") + 1])
        for name, df in [("customers", c), ("orders", o), ("tickets", t)]:
            df.to_sql(name, e, if_exists="replace", index=False)
    else:
        for name, df in [("customers", c), ("orders", o), ("tickets", t)]:
            df.to_csv(f"{name}.csv", index=False)
    print(len(c), "customers", len(o), "orders", len(t), "tickets")
