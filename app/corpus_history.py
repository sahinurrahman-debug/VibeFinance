"""Synthetic *historical* filings: 8 quarters of earnings calls and SEBI disclosures per company,
two earlier annual reports, and quarterly sector notes.

Each company has a tone trajectory (e.g. VIKRAMINFRA deteriorates from +0.5 to -0.4 over two
years, INFOSPHERE improves) so semantic search and the filings agent's "disclosure trend" have real
time structure to work on. Text is template-generated with seeded numeric variation: synthetic by design.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .corpus import Document
from .market import UNIVERSE

# oldest -> newest (Q3 FY25 ... Q1 FY27)
QUARTERS = [("Q3", 25, dt.date(2025, 1, 22)), ("Q4", 25, dt.date(2025, 5, 21)), ("Q1", 26, dt.date(2025, 7, 24)),
            ("Q2", 26, dt.date(2025, 10, 22)), ("Q3", 26, dt.date(2026, 1, 22)), ("Q4", 26, dt.date(2026, 5, 20)),
            ("Q1", 27, dt.date(2026, 7, 24))]
QUARTERS = [("Q2", 25, dt.date(2024, 10, 23))] + QUARTERS      # 8 quarters

TRAJ = {
    "BHARATOIL": [.1, .2, .3, .3, .4, .5, .5, .6], "TECHNOVA": [.2, .2, .3, .3, .3, .4, .4, .4],
    "NAVBANK": [0, .1, .2, .2, .3, .3, .3, .4], "INFOSPHERE": [-.2, -.1, 0, .1, .2, .3, .4, .5],
    "SAMRIDDHI": [.1, 0, .1, 0, .1, .1, 0, .1], "KISANFOODS": [.1, .1, 0, .1, 0, .1, 0, .1],
    "RASHTRIYA": [-.3, -.2, -.1, 0, .2, .3, .4, .5], "SANCHARNET": [-.1, 0, .1, .2, .3, .3, .4, .4],
    "VIKRAMINFRA": [.5, .4, .3, .2, 0, -.2, -.3, -.4], "GATIMOTORS": [.4, .3, .3, .1, 0, -.2, -.3, -.4],
    "AROGYAPHARMA": [.3, .2, .1, 0, -.1, 0, .1, 0], "HINDCONSUMER": [.3, .2, .2, .1, 0, -.1, -.1, -.2],
}
METRIC = {"Energy": "refining margins", "IT": "deal wins", "Banking": "net interest margin", "Financials": "assets under management",
          "FMCG": "volume growth", "Telecom": "ARPU", "Infrastructure": "order execution", "Auto": "vehicle volumes",
          "Pharma": "US generics sales"}

ER_POS = [
    "{name} reported {q} revenue growth of {g}% year on year with operating margin improving to {m}%, and management raised guidance citing strong demand in {metric}.",
    "Order inflow was a record at ₹{o} crore, which gives healthy visibility into the coming quarters. Deleveraging continued as net debt fell to {d}x EBITDA.",
    "The company beat estimates on profit as {metric} improved; management said robust demand and margin expansion should continue.",
    "Capex is fully funded from internal accruals and the dividend payout was maintained. Management expects resilient growth through the year.",
]
ER_NEG = [
    "{name} reported {q} revenue growth of only {g}% and missed estimates as {metric} came under margin pressure; management cut guidance for the year.",
    "Execution was delayed on key projects and elevated leverage of {d}x EBITDA remains a concern, with analysts flagging weak demand and headwinds.",
    "Management acknowledged a slowdown in {metric} and a shortfall against internal targets; provisions increased and the outlook was lowered.",
    "Brokerages downgraded the stock after the call, citing stress in the core business and uncertain visibility on recovery.",
]
ER_MIX = [
    "{name} delivered {q} revenue growth of {g}% with margins broadly flat; management noted headwinds in {metric} offset by healthy demand elsewhere.",
    "Profit was in line with estimates. Management described the environment as mixed, with some concern on costs but steady order flow.",
    "Guidance was maintained. The CFO flagged margin pressure in one segment while pointing to improved collections and stable leverage of {d}x EBITDA.",
]
LODR_POS = [
    "The Board approved a dividend of ₹{div} per share. Promoter holding stood at {ph}% with no pledge, and institutional investors increased their stake during the quarter.",
    "The company won a large multi-year contract, which is accretive to earnings. No regulatory action is pending against the company.",
    "The Board approved a fundraise to refinance debt on better terms; the rating agency reaffirmed the rating with a stable outlook.",
]
LODR_NEG = [
    "Promoter pledge increased to {p}% of promoter holding from {p0}% last quarter, and FII holding fell during the period.",
    "The company received a show-cause notice from the securities regulator regarding related-party transactions; a penalty may be imposed if the allegations are upheld.",
    "A rating agency downgraded the company's debt outlook to negative, citing weak demand and elevated leverage. Litigation provisions were increased.",
]
LODR_MIX = [
    "The Board approved the quarterly results and an interim dividend of ₹{div}. Promoter holding was {ph}% and no material change in shareholding was reported.",
    "The company disclosed routine compliance filings. A minor delayed filing was regularised; no penalty was imposed.",
]
AR_POS = ["The year saw record performance across segments with growth ahead of the industry. Management expects resilient demand and healthy margins, and the Board recommended a higher dividend."]
AR_NEG = ["The year was challenging, with margin pressure, delayed projects and weak demand. The auditor highlighted contingent liabilities and management flagged continued headwinds."]
AR_MIX = ["Performance was steady. Management sees stable demand but cautioned about commodity costs and competitive intensity in {metric}."]
NOTE = {
    "pos": "Sector demand remained healthy in the quarter and {metric} improved across most companies, with robust order flow and resilient margins.",
    "neg": "Sector demand weakened in the quarter and {metric} came under pressure, with analysts flagging headwinds and downgrades across the group.",
    "mix": "Sector conditions were mixed: {metric} was steady, but cost inflation remains a concern for several companies.",
}


def _bank(kind: str, tone: float, rng) -> list[str]:
    banks = {"ER": (ER_POS, ER_MIX, ER_NEG), "LODR": (LODR_POS, LODR_MIX, LODR_NEG), "AR": (AR_POS, AR_MIX, AR_NEG)}[kind]
    if rng.random() < 0.2:                       # occasional off-trend quarter
        tone += rng.choice([-0.3, 0.3])
    return banks[0] if tone > 0.15 else banks[2] if tone < -0.15 else banks[1]


def _fill(t: str, st, q: str, tone: float, rng) -> str:
    g = round(float(np.clip(8 + 14 * tone + rng.normal(0, 1.5), 0.5, 30)), 1)
    return t.format(name=st.name, q=q, g=g, m=round(18 + 6 * tone + rng.normal(0, .8), 1), o=int(rng.integers(8, 60)) * 1000,
                    d=round(float(np.clip(2.0 - 1.5 * tone + rng.normal(0, .2), .3, 4.5)), 1), div=round(float(rng.uniform(4, 30)), 1),
                    ph=round(float(rng.uniform(25, 72)), 1), p=int(np.clip(24 - 30 * tone, 2, 60)), p0=int(np.clip(20 - 30 * tone, 1, 55)),
                    metric=METRIC[st.sector])


def build_history(seed: int = 5) -> list[Document]:
    rng = np.random.default_rng(seed)
    docs: list[Document] = []
    for i, st in enumerate(UNIVERSE):
        traj = TRAJ[st.symbol]
        for k, (q, yy, date) in enumerate(QUARTERS):
            tone = traj[k]
            label = f"{q} FY{yy}"
            er = [("Opening remarks", " ".join(_fill(t, st, label, tone, rng) for t in rng.choice(_bank("ER", tone, rng), 2, replace=False))),
                  ("Outlook and Q&A", _fill(str(rng.choice(_bank("ER", tone, rng))), st, label, tone, rng))]
            lodr = [("Board outcome", _fill(str(rng.choice(_bank("LODR", tone, rng))), st, label, tone, rng)),
                    ("Shareholding and compliance", _fill(str(rng.choice(_bank("LODR", tone, rng))), st, label, tone, rng))]
            d = date + dt.timedelta(days=int(i % 5))
            docs.append(Document(f"{st.symbol}-ER-{q}FY{yy}", st.symbol, [st.sector], f"{label} Earnings Call Transcript - {st.name}",
                                 "Earnings Transcript", d, er, period=label))
            docs.append(Document(f"{st.symbol}-LODR-{q}FY{yy}", st.symbol, [st.sector], f"SEBI LODR Reg. 30 Disclosure ({label}) - {st.name}",
                                 "SEBI Filing", d + dt.timedelta(days=2), lodr, period=label))
        for fy, date, tone in ((24, dt.date(2024, 6, 18), traj[0] + .1), (25, dt.date(2025, 6, 17), traj[2])):
            docs.append(Document(f"{st.symbol}-AR-FY{fy}", st.symbol, [st.sector], f"Annual Report FY{fy} (MD&A) - {st.name}", "Annual Report",
                                 date, [("Management discussion", _fill(str(rng.choice(_bank("AR", tone, rng))), st, f"FY{fy}", tone, rng))],
                                 period=f"FY{fy}"))
    sectors = sorted({s.sector for s in UNIVERSE})
    mean_tone = {sec: np.mean([TRAJ[s.symbol][-1] for s in UNIVERSE if s.sector == sec]) for sec in sectors}
    for sec in sectors:
        for k, (q, yy, date) in enumerate(QUARTERS[-4:]):
            tone = mean_tone[sec] * (0.5 + 0.15 * k) + rng.normal(0, .1)
            kind = "pos" if tone > .15 else "neg" if tone < -.15 else "mix"
            docs.append(Document(f"SECTOR-{sec.upper()}-{q}FY{yy}", None, [sec], f"{sec} Sector Note - {q} FY{yy}", "Sector Note",
                                 date + dt.timedelta(days=7), [("Sector overview", NOTE[kind].format(metric=METRIC[sec]))], period=f"{q} FY{yy}"))
    return docs
