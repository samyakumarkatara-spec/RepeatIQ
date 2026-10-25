import streamlit as st, pandas as pd
import analysis as A

st.set_page_config(page_title="RepeatIQ", layout="wide")
st.title("RepeatIQ - grocery retention & churn intelligence")

@st.cache_data
def load():
    return {k: A.run_sql(k) for k in ["RETENTION", "RFM", "RISK", "QUALITY_BASE", "SUPPORT"]}
D = load(); qe = A.quality_effect(D["QUALITY_BASE"])
t1, t2, t3, t4, t5, t6 = st.tabs(["Cohorts", "RFM", "Quality vs repeat", "Churn risk & actions", "Experiment simulator", "AI insight"])

with t1:
    ret = D["RETENTION"].set_index("cohort_week")
    st.caption("Retention = share of a weekly first-order cohort placing another order on a later day within N days. Immature cohorts are excluded from each window.")
    c = st.columns(3)
    for col, k in zip(c, ["d7", "d30", "d60"]): col.metric(k.upper(), f"{ret[k].mean():.1f}%")
    st.line_chart(ret[["d7", "d30", "d60"]]); st.dataframe(ret.round(1))

with t2:
    rfm = D["RFM"]
    s = rfm.groupby("segment").agg(customers=("customer_id", "count"), avg_recency=("recency_days", "mean"),
                                    avg_orders=("frequency", "mean"), avg_spend=("monetary", "mean")).round(1)
    st.bar_chart(s.customers); st.dataframe(s)

with t3:
    st.subheader("Do customers who report a quality issue reorder less than comparable customers?")
    a, b, c = st.columns(3)
    a.metric("Reorder rate, quality issue", f"{100*qe['rate_exposed']:.1f}%"); b.metric("Reorder rate, no issue", f"{100*qe['rate_unexposed']:.1f}%")
    c.metric("Adjusted gap", f"{100*qe['adjusted_diff']:.1f} pp", f"95% CI {100*qe['ci_low']:.1f} to {100*qe['ci_high']:.1f}")
    st.write(f"Naive gap: **{100*qe['naive_diff']:.1f} pp** (exposed n={qe['n_exposed']}, comparison n={qe['n_control']}). "
             "Adjusted gap compares customers within the same basket-size band and late-delivery status.")
    st.warning("Correlation vs causation: " + A.CAVEAT)
    st.dataframe(D["SUPPORT"])

with t4:
    r = D["RISK"]; band = st.slider("Minimum risk score", 0, 100, 60)
    hi = r[r.risk_score >= band].copy()
    hi["intervention"] = hi.top_driver.map(lambda d: A.INTERVENTION[d][0])
    st.write(f"{len(hi)} customers at or above {band}")
    st.bar_chart(hi.top_driver.value_counts())
    st.dataframe(hi.sort_values("risk_score", ascending=False)[["customer_id", "risk_score", "top_driver", "days_since_last", "n_orders", "intervention"]].head(200))
    for k, (name, msg) in A.INTERVENTION.items(): st.markdown(f"**{k}** -> {name}: _{msg}_")

with t5:
    base = st.slider("Baseline D30 reorder rate", 0.05, 0.6, 0.25); n = st.slider("Customers per arm", 100, 5000, 1000, 100)
    lifts = {"replenishment_reminder": st.slider("Assumed lift: reminder (pp)", 0.0, 10.0, 3.0) / 100,
             "recommendation": st.slider("Assumed lift: recommendation (pp)", 0.0, 10.0, 2.0) / 100,
             "service_recovery": st.slider("Assumed lift: service recovery (pp)", 0.0, 10.0, 5.0) / 100}
    seed = st.number_input("Random seed", 0, 9999, 0)
    st.dataframe(A.simulate_experiment(n, base, lifts, int(seed)).round(4))
    st.caption(f"Required per arm to detect a 3pp lift at 80% power: {A.sample_size_per_arm(base, 0.03):,}. Re-run with other seeds to see how often small samples mislead.")

with t6:
    facts = A.build_facts(D["RETENTION"], qe, D["RISK"], D["SUPPORT"])
    with st.expander("Facts passed to the LLM"): st.json(facts)
    st.markdown(A.ai_summary(facts))
