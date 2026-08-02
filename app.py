"""
ICU Agent OS - single Streamlit entry point.
Run:  streamlit run app.py
"""
import streamlit as st

import icu_scores as sc
from database import DB
from guidelines import GuidelineStore
from llm import LLMClient
from agents import Router, Context

st.set_page_config(page_title="ICU Agent OS", page_icon="🏥", layout="wide")

# --- shared, cached singletons -------------------------------------------- #
@st.cache_resource
def get_db():
    return DB("vatsal_icu.db")

@st.cache_resource
def get_store():
    return GuidelineStore()

db, store, router = get_db(), get_store(), Router()

# --- sidebar: LLM config + patient select --------------------------------- #
with st.sidebar:
    st.header("ICU Agent OS")
    st.caption("Decision support. Scores are computed deterministically; the LLM "
               "only phrases text and never sees un-redacted identifiers.")
    with st.expander("LLM settings (optional)", expanded=False):
        base_url = st.text_input("Base URL", "https://api.openai.com/v1",
                                 help="Use http://localhost:11434/v1 for a local "
                                      "model so patient data never leaves the hospital.")
        model = st.text_input("Model", "gpt-4o-mini")
        api_key = st.text_input("API key", type="password")
    llm = LLMClient(base_url=base_url, api_key=api_key, model=model)
    st.write("LLM: " + ("🟢 configured" if llm.configured else "⚪ offline (templates)"))

    st.divider()
    with st.form("add_patient", clear_on_submit=True):
        st.caption("Add patient")
        n = st.text_input("Name")
        c1, c2 = st.columns(2)
        bed = c1.text_input("Bed")
        age = c2.number_input("Age", 0, 120, 60)
        dx = st.text_input("Diagnosis")
        if st.form_submit_button("Add") and n:
            db.add_patient(n, age=int(age), bed=bed, diagnosis=dx)
            st.rerun()

    patients = db.active_patients()
    labels = [f"Bed {p['bed']} - {p['name']}" for p in patients]
    idx = st.selectbox("Active patient", range(len(patients)),
                       format_func=lambda i: labels[i]) if patients else None
    patient = patients[idx] if idx is not None else None

tabs = st.tabs(["Census", "Daily rounds", "Differential", "Guidelines", "Scores", "Ask"])

# --- Census --------------------------------------------------------------- #
with tabs[0]:
    st.subheader("ICU census")
    if not patients:
        st.info("No active patients. Add one from the sidebar.")
    for p in patients:
        with st.container(border=True):
            st.markdown(f"**Bed {p['bed']} · {p['name']}** — {p.get('diagnosis') or '-'}")
            mem = db.get_memory(p["id"])
            if mem:
                st.caption(" · ".join(f"{k}: {v}" for k, v in mem.items()))

# --- Daily rounds --------------------------------------------------------- #
with tabs[1]:
    if not patient:
        st.info("Select a patient in the sidebar.")
    else:
        st.subheader(f"Rounds — {patient['name']} (Bed {patient['bed']})")
        c = st.columns(4)
        v = {}
        v["hr"] = c[0].number_input("HR", 0, 300, 90)
        v["sbp"] = c[1].number_input("SBP", 0, 300, 120)
        v["dbp"] = c[2].number_input("DBP", 0, 200, 70)
        v["resp_rate"] = c[3].number_input("RR", 0, 80, 18)
        c = st.columns(4)
        v["spo2"] = c[0].number_input("SpO2", 0, 100, 97)
        v["temp"] = c[1].number_input("Temp °C", 30.0, 44.0, 37.0, 0.1)
        v["gcs"] = c[2].number_input("GCS", 3, 15, 15)
        v["map"] = sc.mean_arterial_pressure(v["sbp"], v["dbp"])
        c[3].metric("MAP", v["map"])

        col1, col2 = st.columns(2)
        if col1.button("Generate progress note"):
            ctx = Context(message="progress note", patient=patient, values=v,
                          db=db, guidelines=store, llm=llm)
            r = router.route(ctx)
            st.session_state["note"] = r.text
            if r.used_llm:
                st.caption("Drafted with LLM (identifiers redacted before sending).")
        if col2.button("Save vitals"):
            db.add_update(patient["id"], vitals=v)
            st.success("Saved.")

        if "note" in st.session_state:
            st.text_area("Progress note (edit before use)",
                         st.session_state["note"], height=260)

        st.divider()
        st.caption("Standing memory (separate from clinical notes)")
        mk = st.text_input("Key", "weaning")
        mv = st.text_input("Value")
        if st.button("Save to memory") and mv:
            db.set_memory(patient["id"], mk, mv)
            st.rerun()

# --- Differential --------------------------------------------------------- #
with tabs[2]:
    st.subheader("Differential support")
    findings = st.multiselect("Findings", [
        "hypotension", "lactate_high", "warm", "cold", "raised_jvp",
        "clear_chest", "fever", "eschar", "tropical", "bilateral_infiltrates",
        "hypoxia", "oliguria", "creatinine_high"])
    note = st.text_input("Free-text context (optional)")
    if st.button("Suggest differentials"):
        ctx = Context(message=note or "differential", patient=patient,
                      values={"findings": findings}, llm=llm)
        r = router.route(ctx)
        st.markdown(r.text)
        if r.used_llm:
            st.caption("Refined with LLM. Support only — not a diagnosis.")

# --- Guidelines ----------------------------------------------------------- #
with tabs[3]:
    st.subheader("Guideline lookup (local, citable)")
    q = st.text_input("Search", "sepsis")
    if q:
        ctx = Context(message=q, patient=patient, guidelines=store, llm=llm)
        r = router.route(ctx)
        st.markdown(r.text)
        if r.data.get("sources"):
            st.caption("Sources: " + ", ".join(r.data["sources"]))
    st.caption("Stored guidelines: " + ", ".join(store.titles()))

# --- Scores --------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Deterministic severity scores")
    which = st.radio("Score", ["qSOFA", "SOFA", "APACHE II"], horizontal=True)
    try:
        if which == "qSOFA":
            c = st.columns(3)
            rr = c[0].number_input("RR", 0, 80, 22)
            sbp = c[1].number_input("SBP", 0, 300, 100)
            gcs = c[2].number_input("GCS", 3, 15, 15)
            st.code(str(sc.qsofa(rr, sbp, gcs)))
        elif which == "SOFA":
            c = st.columns(3)
            pf = c[0].number_input("PaO2/FiO2", 0, 600, 300)
            supp = c[0].checkbox("On resp support")
            plt = c[1].number_input("Platelets ×10³", 0, 1000, 150)
            bili = c[1].number_input("Bilirubin mg/dL", 0.0, 60.0, 1.0, 0.1)
            mapv = c[2].number_input("MAP", 0, 200, 70)
            gcs = c[2].number_input("GCS ", 3, 15, 15)
            creat = c[2].number_input("Creatinine mg/dL", 0.0, 20.0, 1.0, 0.1)
            st.code(str(sc.sofa(pao2_fio2=pf, on_respiratory_support=supp,
                                platelets=plt, bilirubin=bili, map_mmhg=mapv,
                                gcs=gcs, creatinine=creat)))
        else:
            st.caption("APACHE II uses worst values in first 24h.")
            c = st.columns(4)
            age = c[0].number_input("Age", 0, 120, 60)
            temp = c[1].number_input("Temp °C", 25.0, 45.0, 37.0, 0.1)
            mapv = c[2].number_input("MAP ", 0, 250, 80)
            hr = c[3].number_input("HR", 0, 300, 90)
            c = st.columns(4)
            rr = c[0].number_input("RR ", 0, 80, 18)
            fio2 = c[1].number_input("FiO2", 0.21, 1.0, 0.21, 0.01)
            pao2 = c[2].number_input("PaO2 (FiO2<0.5)", 0, 600, 90)
            aado2 = c[3].number_input("A-aDO2 (FiO2≥0.5)", 0, 800, 100)
            c = st.columns(4)
            ph = c[0].number_input("pH", 6.5, 8.0, 7.4, 0.01)
            na = c[1].number_input("Na", 100, 200, 140)
            k = c[2].number_input("K", 1.0, 9.0, 4.0, 0.1)
            creat = c[3].number_input("Creatinine", 0.0, 20.0, 1.0, 0.1)
            c = st.columns(4)
            hct = c[0].number_input("Hct %", 10, 70, 40)
            wbc = c[1].number_input("WBC ×10³", 0.0, 100.0, 8.0, 0.1)
            gcs = c[2].number_input("GCS  ", 3, 15, 15)
            arf = c[3].checkbox("Acute renal failure")
            st.code(str(sc.apache2(age=age, temperature_c=temp, map_mmhg=mapv,
                                   heart_rate=hr, resp_rate=rr, fio2=fio2, pao2=pao2,
                                   aado2=aado2, arterial_ph=ph, sodium=na, potassium=k,
                                   creatinine=creat, acute_renal_failure=arf,
                                   hematocrit=hct, wbc=wbc, gcs=gcs)))
    except ValueError as e:
        st.error(str(e))

# --- Ask (free routing) --------------------------------------------------- #
with tabs[5]:
    st.subheader("Ask the OS")
    q = st.text_input("Message", "sepsis protocol")
    if st.button("Send"):
        ctx = Context(message=q, patient=patient, db=db, guidelines=store, llm=llm)
        r = router.route(ctx)
        st.caption(f"Routed to: {r.agent}")
        st.markdown(r.text)
