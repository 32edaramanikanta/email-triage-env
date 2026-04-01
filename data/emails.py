"""
Synthetic email dataset for the Email Triage environment.
Covers billing, technical, complaints, refunds, spam, and general inquiries.
Designed to cover easy → medium → hard scenarios.
"""

from typing import Dict, List
from models import Email

# ---------------------------------------------------------------------------
# TASK 1 — Easy: Basic triage (priority + label assignment)
# ---------------------------------------------------------------------------

TASK1_EMAILS: List[Email] = [
    Email(
        id="t1_001",
        subject="URGENT: Cannot login to my account — production down",
        sender="cto@bigcorp.com",
        sender_domain="bigcorp.com",
        body=(
            "Our entire team is locked out of the platform. We have a client demo "
            "in 45 minutes. This is a CRITICAL issue. Please escalate immediately. "
            "Account: bigcorp-enterprise. We pay $50k/month for enterprise SLA."
        ),
        received_at="2024-03-15T09:03:00Z",
        thread_length=1,
        metadata={"account_tier": "enterprise", "mrr": 50000},
    ),
    Email(
        id="t1_002",
        subject="Question about pricing",
        sender="john.doe@gmail.com",
        sender_domain="gmail.com",
        body=(
            "Hi, I was wondering what the pricing is for your Pro plan. "
            "Do you offer any discounts for startups? Thanks!"
        ),
        received_at="2024-03-15T09:05:00Z",
        thread_length=1,
        metadata={"account_tier": "free"},
    ),
    Email(
        id="t1_003",
        subject="Get rich quick — make $5000/day from home!!!",
        sender="promo@totallylegit-money.biz",
        sender_domain="totallylegit-money.biz",
        body=(
            "CONGRATULATIONS! You have been selected to participate in our "
            "exclusive money-making program. Click here NOW to claim your $5000 bonus! "
            "Limited time offer!!! ACT NOW!!!"
        ),
        received_at="2024-03-15T09:07:00Z",
        thread_length=1,
        metadata={"spam_score": 0.97},
    ),
    Email(
        id="t1_004",
        subject="Invoice #INV-2024-0312 — Payment Overdue",
        sender="billing@acmecorp.com",
        sender_domain="acmecorp.com",
        body=(
            "Hello, I noticed our invoice from last month still shows as unpaid in your "
            "system but we sent the wire transfer on March 10th. Reference: WT-445892. "
            "Could you please confirm receipt and update our account status? "
            "Our subscription renews on March 20th and we don't want service interruption."
        ),
        received_at="2024-03-15T09:12:00Z",
        thread_length=2,
        metadata={"account_tier": "business", "mrr": 2000},
    ),
    Email(
        id="t1_005",
        subject="Feature request: dark mode",
        sender="user123@outlook.com",
        sender_domain="outlook.com",
        body=(
            "Hi support team! Love the product. One thing I'd love to see is a dark mode "
            "option for the dashboard. My eyes get tired at night. Just a suggestion :) "
            "Keep up the great work!"
        ),
        received_at="2024-03-15T09:20:00Z",
        thread_length=1,
        metadata={"account_tier": "pro"},
    ),
]

TASK1_GROUND_TRUTH = {
    "t1_001": {"priority": "urgent",  "label": "technical",  "action": "escalate"},
    "t1_002": {"priority": "low",     "label": "inquiry",    "action": "draft_reply"},
    "t1_003": {"priority": "spam",    "label": "other",      "action": "mark_spam"},
    "t1_004": {"priority": "high",    "label": "billing",    "action": "draft_reply"},
    "t1_005": {"priority": "low",     "label": "feedback",   "action": "archive"},
}

# ---------------------------------------------------------------------------
# TASK 2 — Medium: Draft appropriate replies + handle ambiguous cases
# ---------------------------------------------------------------------------

TASK2_EMAILS: List[Email] = [
    Email(
        id="t2_001",
        subject="Re: Re: Re: Refund request — 3 weeks waiting",
        sender="angry.customer@hotmail.com",
        sender_domain="hotmail.com",
        body=(
            "This is absolutely unacceptable. I've been waiting 3 WEEKS for my refund "
            "of $349. I've sent 4 emails and nobody responds. If this isn't resolved "
            "TODAY I'm disputing with my credit card company AND posting reviews everywhere. "
            "Order #ORD-20240224-8871."
        ),
        received_at="2024-03-15T10:00:00Z",
        thread_length=4,
        metadata={"account_tier": "pro", "open_refund_ticket": "REF-8871", "days_waiting": 21},
    ),
    Email(
        id="t2_002",
        subject="API rate limits documentation unclear",
        sender="developer@techstartup.io",
        sender_domain="techstartup.io",
        body=(
            "Hey, I've been reading your API docs and I'm confused about rate limits. "
            "The docs say '1000 requests/min' but my team hit a 429 error at 800 req/min. "
            "Is there a burst limit? Also, are limits per API key or per account? "
            "We're on the Growth plan. This is blocking our production launch."
        ),
        received_at="2024-03-15T10:15:00Z",
        thread_length=1,
        metadata={"account_tier": "growth", "api_calls_today": 45000},
    ),
    Email(
        id="t2_003",
        subject="Data export for GDPR compliance",
        sender="dpo@eucompany.de",
        sender_domain="eucompany.de",
        body=(
            "Dear Support, We are a German company and our Data Protection Officer "
            "requires a full data export for GDPR audit purposes. We need: all user data, "
            "processing logs, and third-party data sharing records for the past 24 months. "
            "Please respond within 72 hours as required by GDPR Article 15. "
            "Our DPA ref: GDPR-2024-DE-0089."
        ),
        received_at="2024-03-15T10:30:00Z",
        thread_length=1,
        metadata={"account_tier": "enterprise", "region": "EU", "legal_flag": True},
    ),
    Email(
        id="t2_004",
        subject="Unauthorized charges on my account",
        sender="victim@yahoo.com",
        sender_domain="yahoo.com",
        body=(
            "I just checked my bank statement and there are 3 charges from your company "
            "that I did NOT authorize: $29.99 on Feb 15, $29.99 on Mar 1, $29.99 on Mar 15. "
            "I never signed up for a subscription. I want these reversed immediately "
            "and my account deleted. I'm considering reporting this as fraud."
        ),
        received_at="2024-03-15T10:45:00Z",
        thread_length=1,
        metadata={"account_tier": "unknown", "fraud_flag": True},
    ),
    Email(
        id="t2_005",
        subject="Partnership opportunity — AI integration",
        sender="bd@aipartner.com",
        sender_domain="aipartner.com",
        body=(
            "Hi, I'm the Business Development lead at AIPartner. We've built an AI layer "
            "that integrates with platforms like yours to boost user engagement by 40%. "
            "We have 50+ integrations and $10M ARR. Would love 20 minutes to explore "
            "a potential partnership. Are you the right person to talk to?"
        ),
        received_at="2024-03-15T11:00:00Z",
        thread_length=1,
        metadata={"is_vendor": True},
    ),
]

TASK2_GROUND_TRUTH = {
    "t2_001": {
        "priority": "urgent", "label": "refund", "action": "draft_reply",
        "reply_must_include": ["refund", "apologize", "order"],
        "reply_must_avoid": ["delay", "wait", "process"],
        "tone": "apologetic_and_resolving",
    },
    "t2_002": {
        "priority": "high", "label": "technical", "action": "draft_reply",
        "reply_must_include": ["rate limit", "burst", "api key"],
        "tone": "technical_and_helpful",
    },
    "t2_003": {
        "priority": "urgent", "label": "billing", "action": "escalate",
        "escalation_keywords": ["legal", "gdpr", "compliance", "dpo"],
    },
    "t2_004": {
        "priority": "urgent", "label": "billing", "action": "escalate",
        "escalation_keywords": ["fraud", "unauthorized", "security"],
    },
    "t2_005": {
        "priority": "low", "label": "other", "action": "archive",
    },
}

# ---------------------------------------------------------------------------
# TASK 3 — Hard: Multi-constraint triage with SLA rules and templates
# ---------------------------------------------------------------------------

TASK3_EMAILS: List[Email] = [
    Email(
        id="t3_001",
        subject="[CRITICAL] Data breach suspected — immediate action required",
        sender="security@financeinstitution.com",
        sender_domain="financeinstitution.com",
        body=(
            "Our security team has detected anomalous data access patterns in your platform "
            "using our API key. We believe there may be unauthorized access to our customer "
            "financial data. We are a regulated financial institution (FINRA/SEC). "
            "Under 17 CFR 248.30, we require incident notification within 3 hours. "
            "Contact our CISO directly: security@financeinstitution.com. "
            "This is not a drill. Account: fin-enterprise-001."
        ),
        received_at="2024-03-15T14:00:00Z",
        has_attachment=True,
        thread_length=1,
        metadata={
            "account_tier": "enterprise",
            "mrr": 150000,
            "regulated_industry": "financial",
            "security_incident": True,
            "sla_hours": 1,
        },
    ),
    Email(
        id="t3_002",
        subject="Re: Account suspension — appeal",
        sender="ceo@smallbiz.com",
        sender_domain="smallbiz.com",
        body=(
            "My account was suspended yesterday with no warning. I run a legitimate e-commerce "
            "business and depend entirely on your platform. I have 200 orders pending that I "
            "cannot fulfill. Your automated system flagged me for 'suspicious activity' — "
            "that was me testing new payment flows for our holiday campaign. "
            "I've been a customer for 3 years and pay $800/month. Please reinstate immediately "
            "or explain what I need to do. This is causing real financial damage."
        ),
        received_at="2024-03-15T14:10:00Z",
        thread_length=2,
        metadata={
            "account_tier": "business",
            "mrr": 800,
            "account_age_months": 36,
            "suspension_reason": "fraud_detection_flag",
            "appeal_count": 1,
        },
    ),
    Email(
        id="t3_003",
        subject="Accessibility complaint — ADA violation",
        sender="legal@disabilityadvocates.org",
        sender_domain="disabilityadvocates.org",
        body=(
            "We represent a client with visual impairment who has been unable to use your "
            "platform due to lack of screen reader support and missing ARIA labels. "
            "Under the Americans with Disabilities Act (ADA) and WCAG 2.1 AA standards, "
            "your platform may be in violation. We request: (1) acknowledgment within 48h, "
            "(2) remediation plan within 30 days, (3) immediate interim accommodation. "
            "Failure to respond may result in formal complaint to the DOJ."
        ),
        received_at="2024-03-15T14:20:00Z",
        thread_length=1,
        metadata={
            "account_tier": "free",
            "legal_flag": True,
            "regulatory_risk": "high",
            "sla_hours": 48,
        },
    ),
    Email(
        id="t3_004",
        subject="Bulk export of 2M records — performance issue",
        sender="dataeng@unicorn-startup.com",
        sender_domain="unicorn-startup.com",
        body=(
            "We're trying to export 2 million records via your API for a data migration. "
            "The export keeps timing out after ~50k records. We've tried: pagination with "
            "page_size=1000, async export endpoint, and direct DB dump (not available on our plan). "
            "We're on Enterprise and this migration needs to complete by EOD Friday for our "
            "board presentation. Can you enable a one-time full export or increase our rate limits?"
        ),
        received_at="2024-03-15T14:30:00Z",
        thread_length=3,
        metadata={
            "account_tier": "enterprise",
            "mrr": 25000,
            "deadline": "2024-03-17T18:00:00Z",
            "technical_escalation_needed": True,
        },
    ),
    Email(
        id="t3_005",
        subject="Re: Churn — cancelling after 5 years",
        sender="owner@loyalcustomer.com",
        sender_domain="loyalcustomer.com",
        body=(
            "After 5 years and paying over $60,000 total, I'm cancelling. The recent price "
            "increase of 40% with no notice, combined with the new feature restrictions on "
            "my legacy plan, makes this untenable for my small team. I'm moving to [Competitor]. "
            "If you want to retain me, I need: (1) grandfather pricing, (2) feature parity with "
            "my old plan. Otherwise please confirm cancellation and export my data."
        ),
        received_at="2024-03-15T14:40:00Z",
        thread_length=1,
        metadata={
            "account_tier": "legacy_pro",
            "lifetime_value": 60000,
            "tenure_years": 5,
            "churn_risk": "critical",
            "retention_authority_needed": True,
        },
    ),
]

TASK3_SLA_RULES = {
    "enterprise": {"response_hours": 4, "escalate_threshold": "high"},
    "business": {"response_hours": 24, "escalate_threshold": "urgent"},
    "growth": {"response_hours": 24, "escalate_threshold": "urgent"},
    "pro": {"response_hours": 48, "escalate_threshold": "urgent"},
    "free": {"response_hours": 72, "escalate_threshold": "urgent"},
    "legacy_pro": {"response_hours": 24, "escalate_threshold": "high"},
}

TASK3_GROUND_TRUTH = {
    "t3_001": {
        "priority": "urgent", "label": "technical", "action": "escalate",
        "escalation_keywords": ["security", "breach", "incident", "ciso", "legal"],
        "sla_violation_if_not_escalated": True,
    },
    "t3_002": {
        "priority": "high", "label": "account", "action": "escalate",
        "escalation_keywords": ["suspension", "appeal", "reinstate"],
        "context_matters": True,  # 3-year customer, legitimate reason
    },
    "t3_003": {
        "priority": "urgent", "label": "complaint", "action": "escalate",
        "escalation_keywords": ["ada", "legal", "accessibility", "doj"],
        "sla_violation_if_not_escalated": True,
    },
    "t3_004": {
        "priority": "high", "label": "technical", "action": "escalate",
        "escalation_keywords": ["migration", "export", "rate limit", "enterprise"],
    },
    "t3_005": {
        "priority": "urgent", "label": "account", "action": "escalate",
        "escalation_keywords": ["churn", "cancel", "retention", "grandfather", "ltv"],
        "retention_opportunity": True,
    },
}

TASK3_CONTEXT = {
    "sla_rules": TASK3_SLA_RULES,
    "escalation_team": {
        "security": "security-response@company.com",
        "legal": "legal@company.com",
        "retention": "customer-success@company.com",
        "technical": "eng-oncall@company.com",
    },
    "response_templates": {
        "acknowledge_legal": (
            "Thank you for reaching out. We take this matter seriously and have "
            "escalated your request to our specialized team who will contact you "
            "within the required timeframe."
        ),
        "acknowledge_urgent": (
            "Thank you for contacting us. We understand the urgency and have "
            "flagged your case as high priority. Our team will respond shortly."
        ),
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, dict] = {
    "task1_basic_triage": {
        "description": (
            "Basic email triage: assign the correct priority level (urgent/high/medium/low/spam) "
            "and category label to each incoming email. Choose the right action type for each."
        ),
        "difficulty": "easy",
        "emails": TASK1_EMAILS,
        "ground_truth": TASK1_GROUND_TRUTH,
        "context": {},
        "max_steps": 5,
        "scoring_weights": {"priority": 0.4, "label": 0.4, "action": 0.2},
    },
    "task2_draft_replies": {
        "description": (
            "Handle escalated customer issues: draft appropriate replies for complaints and "
            "technical questions, escalate legal/fraud cases, and archive low-value vendor outreach. "
            "Replies are graded on tone, completeness, and professionalism."
        ),
        "difficulty": "medium",
        "emails": TASK2_EMAILS,
        "ground_truth": TASK2_GROUND_TRUTH,
        "context": {},
        "max_steps": 5,
        "scoring_weights": {"priority": 0.2, "label": 0.2, "action": 0.3, "reply_quality": 0.3},
    },
    "task3_sla_constrained": {
        "description": (
            "Complex triage under SLA constraints: process high-stakes emails from enterprise "
            "customers, regulated industries, and legal contacts. Must respect SLA rules, "
            "identify regulatory risk, and choose escalation paths correctly. "
            "Partial credit for correct priority even if action is wrong."
        ),
        "difficulty": "hard",
        "emails": TASK3_EMAILS,
        "ground_truth": TASK3_GROUND_TRUTH,
        "context": TASK3_CONTEXT,
        "max_steps": 5,
        "scoring_weights": {"priority": 0.25, "label": 0.15, "action": 0.35, "escalation_quality": 0.25},
    },
}
