"""Domain glossary used only to *train* the LSA embedder (never indexed or cited).

Each line groups paraphrases that appear in real filings and analyst notes, so that the latent
space places "watchdog questions undisclosed dealings with affiliates" next to "regulator show-cause
notice on related-party transactions". It plays the role that pre-training plays for a neural embedder.
"""

GLOSSARY = [
    "the regulator watchdog authority SEBI issued a show-cause notice questioning undisclosed non-disclosed dealings with affiliates group companies related-party transactions",
    "the securities authority watchdog may impose a penalty fine sanction if the allegations violation breach of disclosure rules are upheld",
    "creditors lenders bondholders banks are owed borrowings debt obligations; high leverage raises refinancing risk and default concerns",
    "the company is in financial trouble distress stress with weak cash flow, rising debt, pledged shares and possible default",
    "promoters pledged their shares as collateral for loans, a pledge increase signals financial stress and lender pressure",
    "auditor accountant qualified opinion flagged going concern doubts, irregularities, accounting concerns and contingent liabilities",
    "lawsuit legal dispute court case litigation claims against the company may lead to damages provisions and settlement costs",
    "rating agency downgraded the credit rating outlook negative citing leverage weak demand and deteriorating coverage",
    "profit warning guidance cut lowered outlook missed estimates disappointed analysts earnings shortfall revenue miss",
    "strong results record profit beat estimates raised guidance upgraded outlook better than expected earnings surprise",
    "sales revenue turnover growth increased expanded year on year demand volumes orders strong healthy robust",
    "slowdown weak demand falling volumes declining sales softness muted consumption lower orders headwinds",
    "margins profitability operating margin EBITDA improved expansion efficiency cost savings pricing power",
    "margin pressure squeeze input costs inflation commodity prices raw material wage costs eroded profitability",
    "dividend payout shareholder returns buyback repurchase capital return cash distribution to investors",
    "deleveraging debt reduction repayment lower borrowings stronger balance sheet net debt to EBITDA improved",
    "order book order inflow contracts wins pipeline visibility backlog multi-year deals secured",
    "project delays execution slippage cost overruns environmental clearances postponed completion",
    "product recall defect safety issue faulty component quality problem regulatory inspection warning letter",
    "drug regulator FDA inspection observations data integrity plant compliance remediation approval launch",
    "bank loan growth deposits net interest margin credit cost asset quality gross NPA slippages provisions",
    "telecom subscribers ARPU tariff hike spectrum 5G churn market share data usage",
    "software services IT deal wins attrition utilisation clients discretionary spending digital AI demand",
    "consumer goods FMCG rural demand urban premiumisation volumes pricing distribution brands",
    "oil refining petrochemicals retail energy margins crude prices refining spreads green hydrogen",
    "vehicles auto sales inventory dealers discounts exports electric EV launch emissions rules",
    "infrastructure ports roads projects construction contracts government spending execution",
    "foreign institutional investors FII flows buying selling stake holding increased reduced institutional ownership",
    "promoter holding shareholding pattern stake sale offer for sale government divestment overhang",
    "interest rates repo rate central bank RBI monetary policy inflation liquidity stance bank funding costs",
    "sector outlook industry demand analysts brokerages recommend upgrade downgrade target price estimates",
    "management commentary outlook guidance expects confident cautious uncertain visibility next quarter",
    "board approved resolution appointment resignation director auditor key managerial personnel change",
]
