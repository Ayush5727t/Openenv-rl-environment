from __future__ import annotations

from copy import deepcopy
from typing import Dict

from ..models import SupportTeam, TaskSpec, TicketCategory, TicketGroundTruth, TicketPriority


TASK_LIBRARY: Dict[str, TaskSpec] = {
    "easy": TaskSpec(
        task_id="easy",
        title="Single ticket triage",
        max_steps=12,
        tickets=[
            TicketGroundTruth(
                ticket_id="T-100",
                subject="Double charge on invoice",
                customer_message=(
                    "I was charged twice for my monthly plan and need a refund today."
                ),
                correct_category=TicketCategory.billing,
                correct_priority=TicketPriority.high,
                correct_team=SupportTeam.billing_ops,
                requires_escalation=False,
                required_response_keywords=["refund", "invoice", "review"],
            )
        ],
        reward_weights={
            "category": 0.25,
            "priority": 0.25,
            "team": 0.25,
            "resolved": 0.25,
        },
    ),
    "medium": TaskSpec(
        task_id="medium",
        title="Queue ordering under SLA",
        max_steps=28,
        tickets=[
            TicketGroundTruth(
                ticket_id="T-200",
                subject="Cannot log in after 2FA reset",
                customer_message="My account is locked and I cannot access production dashboards.",
                correct_category=TicketCategory.account,
                correct_priority=TicketPriority.urgent,
                correct_team=SupportTeam.account_support,
                requires_escalation=False,
                required_response_keywords=["identity", "unlock", "priority"],
            ),
            TicketGroundTruth(
                ticket_id="T-201",
                subject="Webhook timeouts",
                customer_message="Webhook retries keep failing with timeout after latest deploy.",
                correct_category=TicketCategory.technical,
                correct_priority=TicketPriority.high,
                correct_team=SupportTeam.tech_support,
                requires_escalation=False,
                required_response_keywords=["logs", "reproduce", "incident"],
            ),
            TicketGroundTruth(
                ticket_id="T-202",
                subject="Request invoice pdf copy",
                customer_message="Please send a PDF invoice for March for accounting records.",
                correct_category=TicketCategory.billing,
                correct_priority=TicketPriority.medium,
                correct_team=SupportTeam.billing_ops,
                requires_escalation=False,
                required_response_keywords=["invoice", "pdf", "email"],
            ),
        ],
        reward_weights={
            "category": 0.20,
            "priority": 0.20,
            "team": 0.20,
            "queue": 0.20,
            "resolved": 0.20,
        },
    ),
    "hard": TaskSpec(
        task_id="hard",
        title="Complex triage with escalation and compliant response",
        max_steps=40,
        tickets=[
            TicketGroundTruth(
                ticket_id="T-300",
                subject="Harassment report in user messages",
                customer_message="A user is threatening me in direct messages, please help immediately.",
                correct_category=TicketCategory.abuse,
                correct_priority=TicketPriority.urgent,
                correct_team=SupportTeam.trust_safety,
                requires_escalation=True,
                required_response_keywords=["safety", "report", "investigation"],
            ),
            TicketGroundTruth(
                ticket_id="T-301",
                subject="API latency spike",
                customer_message="Response times are above 5s in our main region since 09:10 UTC.",
                correct_category=TicketCategory.technical,
                correct_priority=TicketPriority.high,
                correct_team=SupportTeam.tech_support,
                requires_escalation=True,
                required_response_keywords=["incident", "mitigation", "update"],
            ),
            TicketGroundTruth(
                ticket_id="T-302",
                subject="Need account ownership transfer",
                customer_message="Our admin left and we need ownership moved to a new lead.",
                correct_category=TicketCategory.account,
                correct_priority=TicketPriority.medium,
                correct_team=SupportTeam.account_support,
                requires_escalation=False,
                required_response_keywords=["verification", "ownership", "process"],
            ),
            TicketGroundTruth(
                ticket_id="T-303",
                subject="Refund for failed add-on purchase",
                customer_message="The add-on failed to activate but the charge posted to my card.",
                correct_category=TicketCategory.billing,
                correct_priority=TicketPriority.high,
                correct_team=SupportTeam.billing_ops,
                requires_escalation=False,
                required_response_keywords=["refund", "billing", "timeline"],
            ),
        ],
        reward_weights={
            "category": 0.15,
            "priority": 0.15,
            "team": 0.15,
            "queue": 0.15,
            "escalation": 0.20,
            "response_quality": 0.10,
            "resolved": 0.10,
        },
    ),
}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASK_LIBRARY:
        raise KeyError(f"Unknown task id: {task_id}")
    return deepcopy(TASK_LIBRARY[task_id])
