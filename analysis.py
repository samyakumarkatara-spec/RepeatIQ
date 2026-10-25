import os, math, numpy as np, pandas as pd
import queries as Q
from generate_data import generate, ASOF

ASOF_S = ASOF.strftime("%Y-%m-%d")

def run_sql(name_or_sql: str) -> pd.DataFrame:
    sql = getattr(Q, name_or_sql, name_or_sql).format(asof=ASOF_S)
    url = os.getenv("DATABASE_URL")
    if url:
        from sqlalchemy import create_engine
        return pd.read_sql(sql, create_engine(url))
    import duckdb
    con = duckdb.connect()
    c, o, t = generate()
    for n, d in [("customers", c), ("orders", o), ("tickets", t)]:
        con.register(n, d)
    return con.execute(sql).df()

def quality_effect(df: pd.DataFrame, boot=300, seed=1) -> dict:
    df = df.copy()
    df["stratum"] = pd.qcut(df.first_amount, 5, labels=False, duplicates="drop").astype(str) + "_" + df.first_late.astype(str)
    def att(d):
        g = d.groupby(["stratum", "quality_issue"]).reordered.agg(["mean", "size"]).unstack()
        g = g.dropna()
        w = g[("size", 1)] / g[("size", 1)].sum()
        return float(((g[("mean", 1)] - g[("mean", 0)]) * w).sum())
    naive = df[df.quality_issue == 1].reordered.mean() - df[df.quality_issue == 0].reordered.mean()
    r = np.random.default_rng(seed)
    bs = [att(df.sample(len(df), replace=True, random_state=int(r.integers(1e9)))) for _ in range(boot)]
    out = {"n_exposed": int(df.quality_issue.sum()), "n_control": int((1 - df.quality_issue).sum()),
           "rate_exposed": df[df.quality_issue == 1].reordered.mean(), "rate_unexposed": df[df.quality_issue == 0].reordered.mean(),
           "naive_diff": naive, "adjusted_diff": att(df), "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5))}
    try:
        from sklearn.linear_model import LogisticRegression
        X = pd.DataFrame({"q": df.quality_issue, "lb": np.log(df.first_amount), "late": df.first_late})
        out["adj_logit_coef_quality"] = float(LogisticRegression(C=1e6, max_iter=1000).fit(X, df.reordered).coef_[0][0])
    except ImportError:
        pass
    return out

CAVEAT = ("Observational estimate. We adjust for basket size and late delivery, but not for unmeasured traits "
          "(e.g. pickier customers may both complain more and leave sooner). Treat as an association that survives "
          "adjustment, not proof of causation; confirm with a service-recovery A/B test.")

INTERVENTION = {"quality_issue": ("Service recovery message + refund/credit within 24h", "Sorry about the {item}. Credit added; fresher replacement on your next order."),
                "late_delivery": ("Service recovery message + free-delivery coupon", "Your last order was late - next delivery is free."),
                "overdue_vs_usual_cadence": ("Replenishment reminder", "You usually reorder about now - your staples are one tap away."),
                "new_customer": ("Personalised recommendation (items bought with their first basket)", "Customers who bought {item} often add these.")}

def _cdf(z): return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def simulate_experiment(n_per_arm, baseline, true_lifts: dict, seed=0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    c = r.binomial(n_per_arm, baseline); pc = c / n_per_arm
    rows = [{"arm": "control", "n": n_per_arm, "rate": pc, "lift": 0.0, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}]
    for k, lift in true_lifts.items():
        x = r.binomial(n_per_arm, min(max(baseline + lift, 0), 1)); p = x / n_per_arm
        se = math.sqrt(p * (1 - p) / n_per_arm + pc * (1 - pc) / n_per_arm)
        pool = (x + c) / (2 * n_per_arm); se0 = math.sqrt(2 * pool * (1 - pool) / n_per_arm)
        z = (p - pc) / se0 if se0 else 0
        rows.append({"arm": k, "n": n_per_arm, "rate": p, "lift": p - pc, "ci_low": p - pc - 1.96 * se,
                     "ci_high": p - pc + 1.96 * se, "p_value": 2 * (1 - _cdf(abs(z)))})
    return pd.DataFrame(rows)

def sample_size_per_arm(baseline, mde, alpha_z=1.96, power_z=0.84):
    p1, p2 = baseline, baseline + mde; pb = (p1 + p2) / 2
    return math.ceil(((alpha_z * math.sqrt(2 * pb * (1 - pb)) + power_z * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / mde ** 2)

def build_facts(ret, qe, risk, tickets) -> dict:
    mature = ret.dropna(subset=["d30"])
    return {"asof": ASOF_S, "customers": int(ret.customers.sum()),
            "avg_d7_pct": round(ret.d7.mean(), 1), "avg_d30_pct": round(mature.d30.mean(), 1), "avg_d60_pct": round(ret.d60.mean(), 1),
            "quality_reorder_rate_pct": round(100 * qe["rate_exposed"], 1), "no_quality_reorder_rate_pct": round(100 * qe["rate_unexposed"], 1),
            "quality_gap_naive_pp": round(100 * qe["naive_diff"], 1), "quality_gap_adjusted_pp": round(100 * qe["adjusted_diff"], 1),
            "quality_gap_ci_pp": [round(100 * qe["ci_low"], 1), round(100 * qe["ci_high"], 1)],
            "high_risk_customers": int((risk.risk_score >= 60).sum()),
            "risk_drivers_high_risk": risk[risk.risk_score >= 60].top_driver.value_counts().to_dict(),
            "tickets_by_category": dict(zip(tickets.category, tickets.tickets))}

def ai_summary(facts: dict) -> str:
    prompt = ("You are a retention analyst. Using ONLY these facts (JSON), write a 5-bullet insight summary for a grocery "
              "operations lead. Quote numbers exactly, never invent figures, use 'associated with' not 'causes', note the "
              "quality-gap confidence interval, and end with one A/B test to run.\n" + str(facts))
    fallback = ""
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            m = anthropic.Anthropic().messages.create(model="claude-sonnet-5-5", max_tokens=600,
                                                      messages=[{"role": "user", "content": prompt}])
            return m.content[0].text
        except Exception as e:
            fallback = f"(LLM unavailable: {e})\n\n"
    f = facts
    return fallback + "\n".join([
        f"- Average retention across cohorts: D7 {f['avg_d7_pct']}%, D30 {f['avg_d30_pct']}%, D60 {f['avg_d60_pct']}%.",
        f"- Customers with a first-order quality ticket reorder at {f['quality_reorder_rate_pct']}% vs {f['no_quality_reorder_rate_pct']}% without.",
        f"- After adjusting for basket size and late delivery the gap is {f['quality_gap_adjusted_pp']} pp (95% CI {f['quality_gap_ci_pp'][0]} to {f['quality_gap_ci_pp'][1]}); naive gap was {f['quality_gap_naive_pp']} pp. Associated with, not proven to cause, lower repeat.",
        f"- {f['high_risk_customers']} customers score high risk; main drivers: {f['risk_drivers_high_risk']}.",
        "- Test next: service-recovery message within 24h of a quality ticket vs. no message (randomised control)."])
