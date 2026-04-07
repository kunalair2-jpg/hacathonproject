"""Generate 1001 synthetic emails across 7 categories and 3 difficulty levels."""
import json, random, os

random.seed(42)

CATEGORY_TEMPLATES = {
    "spam": {
        "subjects": [
            "Congratulations! You've won $1,000,000!!!",
            "FREE gift card inside — claim now",
            "You have been selected for an exclusive offer",
            "Win a free iPhone 15 — limited time",
            "URGENT: Your account has been suspended",
            "Make $5000/week from home, guaranteed",
            "Your shipment is held — pay $3.99 to release",
            "Verify your account or it will be deleted",
            "Investment opportunity — 10000% ROI",
        ],
        "bodies": [
            "Click here immediately to claim your prize. Limited time only. Provide your bank details.",
            "You have been pre-approved. Send us your SSN and date of birth to verify your identity.",
            "Our crypto token is launching. Early investors get 10000% returns. Buy now before it sells out!",
            "Hello friend, I am a Nigerian prince needing to transfer $45M. I will give you 50% for helping.",
            "Your package is waiting. Pay a small delivery fee of $3.99 via gift cards to release it.",
        ],
        "senders": [
            "promo@fake-deals.biz", "winner@lottery9821.com", "noreply@account-verify.net",
            "deals@getrichtoday.ru", "offer@crypto-moon.io", "prize@click-now.net",
        ],
        "priority": "spam", "priority_level": 1, "expected_action": "archive", "routing": "spam-filter",
    },
    "urgent": {
        "subjects": [
            "PRODUCTION DOWN - All hands on deck",
            "Security breach detected - immediate response needed",
            "Database corruption - data loss risk",
            "CEO meeting prep needed in 15 minutes",
            "Client threatening legal action NOW",
            "Critical bug causing revenue loss",
            "System outage - 100% error rate",
            "Unauthorized root access detected",
        ],
        "bodies": [
            "Our main production API is returning 500 errors. 100% failure rate. Revenue impact $10k/min. All engineers report immediately.",
            "Unauthorized access detected. Attacker has root shell. Initiating emergency shutdown protocol. Need response NOW.",
            "The CEO is presenting to the board in 20 minutes and needs the Q3 deck updated. It's CRITICAL.",
            "Client XYZ is threatening $2M lawsuit unless we respond with a resolution plan within the hour.",
            "Critical SQL injection vulnerability discovered in prod. Customer data may be exposed. Need security team immediately.",
        ],
        "senders": [
            "oncall@company.com", "devops@company.com", "ceo-assistant@headoffice.com",
            "security@company.com", "client-success@company.com", "alerts@monitoring.io",
        ],
        "priority": "urgent", "priority_level": 5, "expected_action": "escalate", "routing": "incident-response",
    },
    "billing": {
        "subjects": [
            "Invoice #4920 attached - Q2 services",
            "Payment failed for your subscription",
            "Refund request for duplicate charge",
            "Billing discrepancy - please review",
            "Outstanding balance notice",
            "Requesting itemized invoice for audit",
            "Auto-renewal confirmation",
        ],
        "bodies": [
            "Please find the attached invoice for services rendered this quarter. Payment due in 30 days.",
            "Your subscription payment of $299 has failed. Please update your payment method to avoid service interruption.",
            "I was charged twice for the same order #48291. Please issue a refund for the duplicate charge.",
            "Our finance team flagged a $500 discrepancy between your invoice and our purchase order. Please advise.",
            "Requesting an itemized invoice for all charges in Q2 for our annual audit. Deadline is Friday.",
        ],
        "senders": [
            "finance@client.com", "accounts@vendor.com", "billing@service.io",
            "ap@enterprise.com", "invoices@partner.co",
        ],
        "priority": "normal", "priority_level": 3, "expected_action": "reply", "routing": "finance-team",
    },
    "technical": {
        "subjects": [
            "Bug report: login fails on Safari",
            "API integration - getting 401 errors",
            "SDK not compatible with Python 3.12",
            "Feature request: dark mode support",
            "Performance degradation on mobile",
            "Webhook not firing on event X",
            "Rate limiting hitting our use case",
        ],
        "bodies": [
            "The login button does nothing on Safari 16.3. Works fine on Chrome. Version: latest.",
            "Getting a 401 Unauthorized when calling your REST API with a valid token. Works in Postman but not in code.",
            "Your Python SDK throws TypeError on import in Python 3.12. This is a blocker for our team.",
            "Would it be possible to add dark mode? Our users have been asking for this for months.",
            "Response times on mobile have increased from 300ms to 4s after your last release. Please investigate.",
        ],
        "senders": [
            "dev@startup.io", "engineer@bigcorp.com", "techsupport@user.com",
            "api-user@company.net", "mobile-dev@agency.com",
        ],
        "priority": "normal", "priority_level": 3, "expected_action": "reply", "routing": "engineering-team",
    },
    "complaint": {
        "subjects": [
            "Very disappointed - 3 week wait for response",
            "Your update broke our entire workflow",
            "I want to cancel my account",
            "Terrible customer experience",
            "Unacceptable service levels",
            "I will be moving to a competitor",
        ],
        "bodies": [
            "I have been waiting 3 weeks for a response to my support ticket. This is completely unacceptable.",
            "Your last update completely broke our integration. We lost an entire week of engineering work.",
            "If this is not resolved by Friday I will be cancelling and moving to your competitor.",
            "I am extremely disappointed. I've been a customer for 3 years and this is how you treat loyal users?",
            "Your support team has been useless. I need to speak to a manager immediately.",
        ],
        "senders": [
            "angry.customer@gmail.com", "frustrated@enterprise.com", "unhappy@user.net",
            "escalation@client.org", "ceo@angryclient.com",
        ],
        "priority": "urgent", "priority_level": 4, "expected_action": "escalate", "routing": "customer-success",
    },
    "inquiry": {
        "subjects": [
            "Pricing question - enterprise plan",
            "Interested in your product - quick call?",
            "Do you offer nonprofit discounts?",
            "Request for product demo",
            "Evaluating solutions - need info",
            "Partnership opportunity",
        ],
        "bodies": [
            "Could you send me your enterprise pricing? We have a team of 500 and are evaluating solutions.",
            "I saw your product on HackerNews and I am very interested. Can we schedule a 30-min call?",
            "Do you offer a nonprofit discount? We are a registered 501c3 with a limited budget.",
            "I am evaluating 3 vendors for our company. Could you share a case study and arrange a demo?",
            "We are interested in reselling your product. Could you share your partner program details?",
        ],
        "senders": [
            "potential@customer.com", "sales-inquiry@company.com", "partnership@firm.co",
            "procurement@enterprise.net", "cto@startup.io",
        ],
        "priority": "normal", "priority_level": 2, "expected_action": "reply", "routing": "sales-team",
    },
    "feedback": {
        "subjects": [
            "Love the new dashboard update!",
            "Suggestion for small improvement",
            "Monthly feedback from our team",
            "Great product - keep it up",
            "Feature feedback after 3 months",
        ],
        "bodies": [
            "The new dashboard is fantastic! Our team productivity went up 30%. Thank you!",
            "One small suggestion: could the export button be more prominent? Users keep missing it.",
            "We really appreciate the fast response times lately. The team is happy. Keep it up!",
            "The mobile app could use some UX polish but overall the product is excellent. Great work.",
            "After 3 months of use, we are very happy. The analytics features are best in class.",
        ],
        "senders": [
            "happy@customer.com", "team@partner.org", "feedback@client.net",
            "user@loyalcustomer.com", "nps@survey.co",
        ],
        "priority": "normal", "priority_level": 1, "expected_action": "archive", "routing": "product-team",
    },
}

def generate_emails(n=1001):
    emails = []
    categories = list(CATEGORY_TEMPLATES.keys())
    per_cat = n // len(categories)

    email_id = 1
    for cat, tmpl in CATEGORY_TEMPLATES.items():
        count = per_cat + (1 if cat == "urgent" else 0)
        for i in range(count):
            base_priority = tmpl["priority_level"]

            # Difficulty: easy=clean, medium=slight variation, hard=ambiguous
            r = random.random()
            if r < 0.15:
                difficulty = "easy"
                priority_noise = 0
            elif r < 0.55:
                difficulty = "medium"
                priority_noise = random.choice([-1, 0, 0])
            else:
                difficulty = "hard"
                priority_noise = random.choice([-1, 0, 1])

            priority_level = max(1, min(5, base_priority + priority_noise))

            emails.append({
                "id": f"email_{email_id:04d}",
                "subject": random.choice(tmpl["subjects"]),
                "body": random.choice(tmpl["bodies"]),
                "sender": random.choice(tmpl["senders"]),
                "priority": tmpl["priority"],
                "priority_level": priority_level,
                "category": cat,
                "routing": tmpl["routing"],
                "expected_action": tmpl["expected_action"],
                "difficulty": difficulty,
                "thread_id": f"th_{cat}_{i % 30}" if random.random() > 0.4 else None,
            })
            email_id += 1

    random.shuffle(emails)
    return emails

if __name__ == "__main__":
    emails = generate_emails(1001)
    os.makedirs("data", exist_ok=True)
    with open("data/emails.json", "w", encoding="utf-8") as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)
    cats = {}
    diffs = {}
    for e in emails:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
        diffs[e["difficulty"]] = diffs.get(e["difficulty"], 0) + 1
    print(f"Generated {len(emails)} emails")
    print("Categories:", cats)
    print("Difficulties:", diffs)
