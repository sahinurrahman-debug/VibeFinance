"""Synthetic regulatory-filing / earnings-transcript corpus.

All companies are fictional and every document is synthetic (modelled on the *shape* of
SEBI LODR Reg. 30 disclosures, earnings-call transcripts and annual-report MD&A / risk
sections). They are authored to give each stock a distinct fundamental story that either
agrees or disagrees with its price action. Nothing here describes a real company.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .market import UNIVERSE, BY_SYMBOL


@dataclass
class Document:
    doc_id: str
    symbol: str | None
    sectors: list[str]
    title: str
    doc_type: str
    date: dt.date
    sections: list[tuple[str, str]]
    source_ref: str = "Synthetic document (modelled on real filing formats)"
    period: str = ""
    synthetic: bool = True
    url: str = ""

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"{h}\n{t}" for h, t in self.sections)

    def meta(self) -> dict:
        return {"doc_id": self.doc_id, "symbol": self.symbol, "title": self.title, "doc_type": self.doc_type,
                "date": self.date.isoformat(), "source_ref": self.source_ref, "synthetic": self.synthetic, "period": self.period,
                "sections": [{"heading": h, "text": t} for h, t in self.sections]}


# (earnings, disclosure, annual) -> each a list of (heading, text)
FACTS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "BHARATOIL": {
        "ER": [("Opening remarks", "Bharat Petrochem delivered record consolidated EBITDA of ₹41,200 crore in Q2 FY27, which beat estimates, as refining margins improved to $11.4 per barrel. The CFO raised guidance for FY27 EBITDA by 6%, citing strong demand across retail and energy."),
               ("Balance sheet and capital allocation", "Deleveraging continued as net debt fell to 0.9x EBITDA. Management reiterated its dividend policy and said the new-energy capex is fully funded from internal accruals. The green hydrogen order book stands at ₹6,800 crore, giving healthy visibility into FY28.")],
        "LODR": [("Board outcome", "The Board approved a buyback of up to ₹9,000 crore and a dividend of ₹11 per share. The retail subsidiary also won an exclusive supply agreement that is accretive to FY27 earnings."),
                 ("Shareholding pattern", "Promoter holding stood at 50.3% with no shares pledged. Mutual funds and FIIs both raised holdings during the quarter, and no regulatory orders are pending against the company.")],
        "AR": [("Management discussion", "Retail revenue grew 21% year on year and the company continued to gain market share. Management expects resilient demand through FY27."),
               ("Risk factors", "Crude price swings and changes in fuel-pricing regulation remain the principal risks; a decline in refining spreads could weigh on margins. Contingent liabilities were ₹2,100 crore, unchanged from the prior year.")],
    },
    "TECHNOVA": {
        "ER": [("Opening remarks", "Technova reported revenue growth of 9% in constant currency for Q2 FY27 with operating margin steady at 22.4%. Deal wins totalled $2.1 billion, and management said demand for cloud migration and AI services remains healthy."),
               ("Outlook", "FY27 revenue growth guidance was maintained at 8-10%. The CFO flagged the wage-hike cycle and a stronger rupee as headwinds to margins but expects them to be offset by utilisation gains.")],
        "LODR": [("Board outcome", "The Board declared an interim dividend of ₹28 per share. There were no changes in key managerial personnel during the quarter."),
                 ("Shareholding pattern", "Promoter holding at 72.3% with no pledge. Institutional holding was broadly unchanged.")],
        "AR": [("Management discussion", "Technova continued to diversify its client base, with the top-10 clients contributing 41% of revenue. Attrition improved to 12.8%, the lowest in three years."),
               ("Risk factors", "Client concentration, visa regulation changes and currency movements are the key risks. Management sees no material litigation outstanding.")],
    },
    "NAVBANK": {
        "ER": [("Opening remarks", "Navbharat Bank reported net profit growth of 14% with loan book growth of 16% year on year. Asset quality improved as gross NPA fell to 1.3%, the lowest in eight quarters."),
               ("Margins", "Net interest margin compressed 18 bps to 3.9% on higher deposit costs, and management cautioned that margin pressure may persist for two more quarters. Credit cost remained low and the bank expects loan growth to stay healthy.")],
        "LODR": [("Board outcome", "The Board approved a fundraise of up to ₹5,000 crore through long-term infrastructure bonds. No adverse observations were received in the latest supervisory review."),
                 ("Shareholding pattern", "Promoter holding stood at 26.1% with no pledge; FII holding rose 0.5% during the quarter.")],
        "AR": [("Management discussion", "Deposit mobilisation remained strong and the CASA ratio held at 42%. The bank expects a recovery in retail unsecured lending to support margins in FY27."),
               ("Risk factors", "Sensitivity to rate cuts, rising competition for deposits and slippages in the agriculture book are key risks. Management flagged stress in a few microfinance geographies.")],
    },
    "INFOSPHERE": {
        "ER": [("Opening remarks", "Infosphere Digital posted record revenue of $5.3 billion and beat estimates, with margin expansion of 90 bps on better pricing and utilisation. Management raised guidance for FY27 revenue growth to 7-9% from 5-7%."),
               ("Deal pipeline", "Large deal wins reached a record $4.6 billion in Q2, and the order book provides visibility into FY28. Demand for generative-AI programmes is robust, and no client has reduced committed spends.")],
        "LODR": [("Board outcome", "The Board approved a buyback of ₹12,000 crore and a special dividend. The company confirmed that there is no pending regulatory action."),
                 ("Shareholding pattern", "Promoter holding at 14.8% with no pledge; mutual funds increased their stake during the quarter.")],
        "AR": [("Management discussion", "The company gained market share in digital services and improved utilisation to 85%. The strategy of building AI platforms is expected to drive growth over FY27-FY28."),
               ("Risk factors", "Key risks include discretionary spending cuts in the US and wage inflation; management notes that attrition has improved and bench strength is healthy.")],
    },
    "SAMRIDDHI": {
        "ER": [("Opening remarks", "Samriddhi Finance reported assets under management of ₹1.9 lakh crore, up 12%. Disbursements were stable, and net profit was in line with estimates."),
               ("Asset quality", "Collection efficiency held at 97%. Management noted that slippages rose modestly in the two-wheeler segment but credit costs remain within the guided range.")],
        "LODR": [("Board outcome", "The Board approved the raising of ₹2,500 crore via non-convertible debentures. The proposed dividend is unchanged at ₹12 per share."),
                 ("Shareholding pattern", "Promoter holding stood at 56.0%; pledged shares were 1.2% of promoter holding, reduced from 2.0%.")],
        "AR": [("Management discussion", "The company opened 140 new branches and launched a gold-loan product. Management expects steady growth through FY27."),
               ("Risk factors", "Funding costs, regulatory changes on microfinance and competition from banks are key risks. A weaker rural monsoon could affect collections.")],
    },
    "KISANFOODS": {
        "ER": [("Opening remarks", "Kisan Foods reported revenue growth of 6% and stable margins. Volumes in the packaged foods portfolio were flat, while the hotels segment saw strong demand during the festive season."),
               ("Costs", "Input cost inflation in edible oils and wheat is creating margin pressure, which management expects to offset through price increases and cost savings in the second half.")],
        "LODR": [("Board outcome", "The Board approved the demerger of the hotels business into a separate listed entity, subject to shareholder approval. An interim dividend of ₹7.5 was declared."),
                 ("Shareholding pattern", "The company is professionally managed with no promoter; institutional holding is 78.4% with no material change.")],
        "AR": [("Management discussion", "The company continues to invest in brands and distribution, with rural reach expanded to 2.1 million outlets. Management expects steady growth from premiumisation."),
               ("Risk factors", "Agricultural commodity price swings, regulatory changes on packaged food labelling and a slowdown in discretionary spending are the main risks.")],
    },
    "RASHTRIYA": {
        "ER": [("Opening remarks", "Rashtriya Bank reported its highest-ever quarterly profit, up 31% year on year, driven by record treasury gains and lower provisions. Gross NPA fell to 2.8% and net NPA to 0.6%."),
               ("Capital and dividend", "Capital adequacy rose to 16.2%, and management said the bank is positioned to announce a higher dividend. Credit growth of 13% was supported by robust corporate demand.")],
        "LODR": [("Board outcome", "The Board approved a dividend of ₹6.5 per share. The Government of India indicated it may sell up to 3% of its holding through an offer for sale in FY27."),
                 ("Shareholding pattern", "Government holding at 64.2%; FII holding rose to 8.9%. The bank was not subject to any regulatory action during the quarter.")],
        "AR": [("Management discussion", "Recovery in the corporate book and lower slippages drove profit improvement. The bank targets a return on assets above 1.1% in FY27."),
               ("Risk factors", "The government stake-sale overhang, wage revisions and concentration in public-sector infrastructure loans are key risks. Management flagged weaker treasury income if yields rise.")],
    },
    "SANCHARNET": {
        "ER": [("Opening remarks", "Sanchar Telecom reported a record ARPU of ₹238, up 9% year on year, with 4.1 million net subscriber additions. EBITDA margin expanded to 54%, which beat estimates."),
               ("Capex and debt", "Capex intensity is falling as the 5G rollout nears completion. Net debt fell by ₹6,200 crore and management expects free cash flow to turn robust in FY27. Spectrum instalments of ₹9,400 crore fall due next year, which management is comfortable funding from internal cash flow.")],
        "LODR": [("Board outcome", "The Board approved a tariff increase of 8-10% across prepaid plans effective next month. The Board also approved the monetisation of tower assets, which is accretive to leverage."),
                 ("Shareholding pattern", "Promoter holding stands at 52.0% with no pledge; FII holding rose 0.7%.")],
        "AR": [("Management discussion", "The company gained market share in urban 5G and expects continued ARPU growth from tariff hikes and premium data plans."),
               ("Risk factors", "Spectrum payment obligations, regulatory changes on licence fees and intense competition are key risks. A tariff war would cause margin pressure.")],
    },
    "VIKRAMINFRA": {
        "ER": [("Opening remarks", "Vikram Infra reported a record order book of ₹48,000 crore, but execution at two port projects was delayed by environmental clearances. EBITDA margin was lower than expected and the company missed estimates on net profit."),
               ("Debt", "Net debt rose to 3.8x EBITDA. Management acknowledged elevated leverage and said it is evaluating asset sales to fund obligations, with no clear timeline.")],
        "LODR": [("Regulatory notice", "The company received a show-cause notice from the securities regulator regarding alleged non-disclosure of related-party transactions. The company said it is examining the notice and has engaged external counsel; a penalty may be imposed if the allegations are upheld."),
                 ("Shareholding pattern", "Promoter pledge increased to 38% of promoter holding from 24% last quarter. FII holding fell 1.4%.")],
        "AR": [("Auditor's report", "The statutory auditor issued a qualified opinion on the financial statements of a material subsidiary and highlighted going concern doubts for one project SPV."),
               ("Risk factors", "Contingent liabilities of ₹6,900 crore, ongoing litigation with port authorities and refinancing risk on near-term bonds are the principal risks identified by the Board.")],
    },
    "GATIMOTORS": {
        "ER": [("Opening remarks", "Gati Motors reported a sharp decline in net profit as domestic demand weakened and export volumes were hit by a shortfall in shipments. The company cut guidance for FY27 volume growth to 2-4% from 6-8%."),
               ("Margins", "EBITDA margin contracted 220 bps on commodity costs and higher discounts; management warned of continued margin pressure into Q3. The EV launch has been delayed by one quarter.")],
        "LODR": [("Product recall", "The company announced a voluntary recall of 41,000 vehicles to replace a faulty airbag component, with an estimated cost of ₹320 crore."),
                 ("Rating action", "A rating agency downgraded the outlook on the company's long-term debt to negative, citing weak demand and rising inventory.")],
        "AR": [("Management discussion", "The company retains leadership in commercial vehicles and is investing in electric platforms, though returns on these investments are not expected before FY29."),
               ("Risk factors", "Cyclical demand, commodity prices, emissions-regulation costs and an ongoing probe by the competition regulator into dealer practices are key risks. Headwinds from high inventory and rising interest rates may continue.")],
    },
    "AROGYAPHARMA": {
        "ER": [("Opening remarks", "Arogya Pharma reported domestic formulations growth of 13%, ahead of the market, and received approval for two new launches in the US. Net profit was in line with estimates."),
               ("Regulatory", "The company received a warning letter from the US drug regulator for its Unit-4 plant, which contributes 18% of US sales. Management expects remediation to take 9-12 months and flagged revenue risk from delayed approvals.")],
        "LODR": [("Regulatory update", "The company disclosed receipt of a warning letter relating to data integrity observations at Unit-4. The company stated that it is taking corrective steps; no impact on existing supplies is expected for now."),
                 ("Shareholding pattern", "Promoter holding at 48.6% with no pledge; domestic funds increased their stake.")],
        "AR": [("Management discussion", "The specialty portfolio and strong domestic franchise provide resilient earnings. Research spend rose to 8.5% of revenue with a healthy pipeline of complex generics."),
               ("Risk factors", "Pricing pressure in US generics, regulatory scrutiny of manufacturing sites and litigation over patent challenges are key risks.")],
    },
    "HINDCONSUMER": {
        "ER": [("Opening remarks", "Hind Consumer Products reported underlying volumes flat year on year, and revenue grew only 3%, which missed estimates. Management cited weak demand in rural markets and a slowdown in the personal care category."),
               ("Margins", "Gross margin faced margin pressure from palm oil and packaging inflation, though premiumisation and cost savings partly offset it. The Board maintained the dividend.")],
        "LODR": [("Board outcome", "The Board approved the final dividend of ₹24 per share and the appointment of a new independent director."),
                 ("Shareholding pattern", "Parent company holding is 61.9%; mutual funds trimmed holdings slightly during the quarter.")],
        "AR": [("Management discussion", "The company remains the market leader in several categories and is investing in direct-to-consumer channels. Management expects a recovery in rural demand in FY28."),
               ("Risk factors", "Commodity inflation, competition from regional brands and royalty payments to the parent are key risks; prolonged rural weakness could pressure volumes.")],
    },
}

MACRO = [
    Document("MACRO-RBI-2610", None, ["Banking", "Financials"], "RBI Monetary Policy Statement - October 2026",
             "Macro / Regulator", dt.date(2026, 10, 7),
             [("Policy stance", "The Monetary Policy Committee kept the repo rate unchanged and maintained a neutral stance. Inflation is projected to ease, and growth is expected to remain resilient at 6.8% for FY27."),
              ("Liquidity", "The central bank announced measures to improve system liquidity, which is supportive for bank deposit costs.")]),
    Document("MACRO-IT-2610", None, ["IT"], "IT Services Sector Note - Q2 FY27", "Sector Note", dt.date(2026, 10, 9),
             [("Demand outlook", "US discretionary technology spending showed a recovery in Q2 and deal pipelines are healthy, but a stronger rupee and wage inflation create margin headwinds for the sector.")]),
    Document("MACRO-AUTO-2610", None, ["Auto"], "Auto Sector Note - Festive Season 2026", "Sector Note", dt.date(2026, 10, 8),
             [("Demand", "Passenger-vehicle retail demand was strong over the festive season, but commercial-vehicle volumes remain weak, with high dealer inventory and rising financing costs.")]),
]


def build_corpus(history: bool = True) -> list[Document]:
    docs: list[Document] = []
    for i, st in enumerate(UNIVERSE):
        f = FACTS[st.symbol]
        docs.append(Document(f"{st.symbol}-ER-Q2FY27", st.symbol, [st.sector],
                             f"Q2 FY27 Earnings Call Transcript - {st.name}", "Earnings Transcript",
                             dt.date(2026, 10, 2 + i % 6), f["ER"], period="Q2 FY27"))
        docs.append(Document(f"{st.symbol}-LODR-R30", st.symbol, [st.sector],
                             f"SEBI LODR Reg. 30 Disclosure - {st.name}", "SEBI Filing",
                             dt.date(2026, 10, 6 + i % 7), f["LODR"], period="Q2 FY27"))
        docs.append(Document(f"{st.symbol}-AR-FY26", st.symbol, [st.sector],
                             f"Annual Report FY26 (MD&A and Risk Factors) - {st.name}", "Annual Report",
                             dt.date(2026, 6, 10 + i), f["AR"], period="FY26"))
    docs.extend(MACRO)
    if history:
        from .corpus_history import build_history
        docs.extend(build_history())
    return docs
