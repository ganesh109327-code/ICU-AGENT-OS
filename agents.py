"""
agents.py - The "OS": a router and specialist agents over shared context.

Design principle: agents that produce FACTS (scores, guideline text) are
deterministic and work with zero LLM. The LLM is an optional phrasing layer on
top of already-correct data - it is never the source of a number or a protocol.
"""

from dataclasses import dataclass, field
from typing import Optional

import icu_scores as sc


@dataclass
class Context:
    message: str = ""
    patient: Optional[dict] = None
    values: dict = field(default_factory=dict)   # parsed clinical values
    db: object = None
    guidelines: object = None
    llm: object = None


@dataclass
class AgentResponse:
    agent: str
    text: str
    data: dict = field(default_factory=dict)
    used_llm: bool = False


# --------------------------------------------------------------------------- #
class ScoringAgent:
    name = "scoring"
    description = "Deterministic severity scores (qSOFA, SOFA, APACHE II)."

    def can_handle(self, ctx):
        m = ctx.message.lower()
        return any(k in m for k in ("qsofa", "sofa", "apache", "score", "severity"))

    def handle(self, ctx):
        m = ctx.message.lower()
        v = ctx.values
        results = []
        try:
            if "qsofa" in m or ("score" in m and not any(x in m for x in ("sofa", "apache"))):
                results.append(sc.qsofa(v.get("resp_rate"), v.get("sbp"), v.get("gcs")))
            if "sofa" in m and "qsofa" not in m:
                results.append(sc.sofa(
                    pao2_fio2=v.get("pao2_fio2"),
                    on_respiratory_support=v.get("on_respiratory_support", False),
                    platelets=v.get("platelets"), bilirubin=v.get("bilirubin"),
                    map_mmhg=v.get("map"), gcs=v.get("gcs"),
                    creatinine=v.get("creatinine"),
                    urine_output_ml_day=v.get("urine_output_ml_day")))
            if "apache" in m:
                results.append(sc.apache2(
                    age=v.get("age"), temperature_c=v.get("temp"), map_mmhg=v.get("map"),
                    heart_rate=v.get("hr"), resp_rate=v.get("resp_rate"),
                    fio2=v.get("fio2", 0.21), pao2=v.get("pao2"), aado2=v.get("aado2"),
                    arterial_ph=v.get("ph"), sodium=v.get("sodium"),
                    potassium=v.get("potassium"), creatinine=v.get("creatinine"),
                    acute_renal_failure=v.get("arf", False),
                    hematocrit=v.get("hct"), wbc=v.get("wbc"), gcs=v.get("gcs")))
        except ValueError as e:
            return AgentResponse(self.name, f"Cannot compute score: {e}")

        if not results:
            return AgentResponse(self.name,
                "Specify which score (qSOFA / SOFA / APACHE II) and provide values.")
        text = "\n\n".join(str(r) for r in results)
        return AgentResponse(self.name, text, data={"scores": [r.__dict__ for r in results]})


# --------------------------------------------------------------------------- #
class GuidelineAgent:
    name = "guideline"
    description = "Retrieves and (optionally) summarises stored local guidelines."

    def can_handle(self, ctx):
        m = ctx.message.lower()
        return any(k in m for k in ("guideline", "protocol", "manage", "treatment", "bundle", "how to treat"))

    def handle(self, ctx):
        hits = ctx.guidelines.search(ctx.message, top=2) if ctx.guidelines else []
        if not hits:
            return AgentResponse(self.name, "No matching guideline found in the local store.")
        base = "\n\n".join(f"## {g['title']} ({g['category']})\n{g['content']}" for g in hits)

        if ctx.llm and ctx.llm.configured:
            resp = ctx.llm.chat(
                system="You are an ICU assistant. Using ONLY the guideline text "
                       "provided, give a concise, actionable summary for the "
                       "clinical question. Do not add facts not in the text.",
                user=f"Question: {ctx.message}\n\nGuideline text:\n{base}",
                patient=ctx.patient)
            if resp.available:
                return AgentResponse(self.name, resp.text,
                                     data={"sources": [g["title"] for g in hits]},
                                     used_llm=True)
        return AgentResponse(self.name, base, data={"sources": [g["title"] for g in hits]})


# --------------------------------------------------------------------------- #
# Rule-based differential fallback so this works with no LLM.
_DDX_RULES = [
    ({"hypotension", "lactate_high", "warm"}, "Septic / distributive shock",
     "Broad-spectrum antibiotics + cultures, 30 mL/kg crystalloid, norepinephrine to MAP>=65, source control."),
    ({"hypotension", "cold", "raised_jvp"}, "Cardiogenic shock",
     "Echo, cautious fluids, inotrope (dobutamine), treat ischaemia/arrhythmia, consider mechanical support."),
    ({"hypotension", "raised_jvp", "clear_chest"}, "Obstructive shock (PE / tamponade / tension PTX)",
     "Bedside echo + lung US, decompress/thrombolyse as indicated."),
    ({"fever", "eschar", "tropical"}, "Scrub typhus / tropical sepsis",
     "Empiric doxycycline, look for MODS, supportive ICU care."),
    ({"bilateral_infiltrates", "hypoxia"}, "ARDS",
     "Lung-protective ventilation, PEEP titration, prone if P/F<150."),
    ({"oliguria", "creatinine_high"}, "AKI",
     "Optimise perfusion, stop nephrotoxins, KDIGO staging, RRT per AEIOU."),
]


class DifferentialAgent:
    name = "differential"
    description = "Ranks differentials from selected findings (rule-based, LLM-optional)."

    def can_handle(self, ctx):
        m = ctx.message.lower()
        return any(k in m for k in ("differential", "ddx", "what could", "diagnos"))

    def handle(self, ctx):
        findings = set(ctx.values.get("findings", []))
        ranked = []
        for req, dx, action in _DDX_RULES:
            overlap = len(req & findings)
            if overlap:
                ranked.append((overlap, dx, action))
        ranked.sort(key=lambda x: x[0], reverse=True)

        if ranked:
            lines = [f"- {dx}\n    Next steps: {action}" for _, dx, action in ranked]
            base = "Differentials by selected findings:\n" + "\n".join(lines)
        else:
            base = "No rule matched the selected findings."

        if ctx.llm and ctx.llm.configured and ctx.message.strip():
            resp = ctx.llm.chat(
                system="ICU decision support. Given the findings and the rule-based "
                       "differential list, refine ranking and note red flags. Keep it "
                       "brief. This is support, not a final diagnosis.",
                user=f"Findings: {sorted(findings)}\nNote: {ctx.message}\n\n{base}",
                patient=ctx.patient)
            if resp.available:
                return AgentResponse(self.name, resp.text, used_llm=True)
        return AgentResponse(self.name, base)


# --------------------------------------------------------------------------- #
class DocumentationAgent:
    name = "documentation"
    description = "Assembles a SOAP progress note from structured data."

    def can_handle(self, ctx):
        m = ctx.message.lower()
        return any(k in m for k in ("note", "soap", "progress", "handover", "document"))

    def _template(self, ctx):
        p = ctx.patient or {}
        v = ctx.values
        vit = ", ".join(f"{k} {val}" for k, val in v.items()
                        if k in ("hr", "sbp", "dbp", "map", "resp_rate", "spo2", "temp", "gcs"))
        mem = ctx.db.get_memory(p["id"]) if (ctx.db and p.get("id")) else {}
        mem_txt = "; ".join(f"{k}: {val}" for k, val in mem.items()) or "none"
        return (
            f"PROGRESS NOTE - {p.get('name','[patient]')} (Bed {p.get('bed','-')})\n"
            f"Dx: {p.get('diagnosis','-')}\n\n"
            f"S/O (vitals): {vit or 'not entered'}\n"
            f"Standing issues: {mem_txt}\n\n"
            f"A: [assessment]\nP: [plan]")

    def handle(self, ctx):
        template = self._template(ctx)
        if ctx.llm and ctx.llm.configured:
            resp = ctx.llm.chat(
                system="You are an ICU physician. Expand the structured data into a "
                       "clean SOAP progress note. Do not invent vitals or findings that "
                       "are not present; leave placeholders where data is missing.",
                user=template, patient=ctx.patient)
            if resp.available:
                return AgentResponse(self.name, resp.text, used_llm=True)
        return AgentResponse(self.name, template)


# --------------------------------------------------------------------------- #
class Router:
    """Deterministic routing to the first agent that claims the message."""

    def __init__(self, agents=None):
        self.agents = agents or [
            ScoringAgent(), DifferentialAgent(),
            DocumentationAgent(), GuidelineAgent()]

    def route(self, ctx):
        for a in self.agents:
            if a.can_handle(ctx):
                return a.handle(ctx)
        # default: try guidelines, else help text
        g = GuidelineAgent()
        if ctx.guidelines and ctx.guidelines.search(ctx.message):
            return g.handle(ctx)
        return AgentResponse("router",
            "I can help with: severity scores, differentials, progress notes, and "
            "guideline lookup. Try 'calculate qSOFA' or 'sepsis protocol'.")
