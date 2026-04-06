"""Prompts and schema definitions for Arctic Text2SQL model evaluation.

This module contains the system prompt, database schema, test questions,
and prompt template used for evaluating the Arctic Text2SQL model.
"""

# System prompt that instructs the model to act as a SQL generation expert
SYSTEM_PROMPT = (
    "You are a data science expert. Below, you are provided with a database "
    "schema and a natural language question. Your task is to understand the "
    "schema and generate a valid SQL query to answer the question."
)

# Database schema with 6 interconnected tables covering organizations, cloud resources, and billing
SCHEMA = """CREATE TABLE organizations (
    org_id INTEGER PRIMARY KEY,
    org_name TEXT NOT NULL,
    industry TEXT, -- 'Finance', 'Healthcare', 'Tech'
    hq_country TEXT,
    tier TEXT, -- 'Startup', 'Mid-Market', 'Fortune 500'
    created_at TIMESTAMP
);

CREATE TABLE cloud_resources (
    resource_id INTEGER PRIMARY KEY,
    org_id INTEGER,
    resource_type TEXT, -- 'Compute', 'Storage', 'Database'
    region TEXT, -- 'us-east-1', 'eu-west-1', 'ap-southeast-1'
    status TEXT, -- 'Active', 'Provisioning', 'Terminated'
    hourly_cost REAL,
    FOREIGN KEY (org_id) REFERENCES organizations(org_id)
);

CREATE TABLE usage_metrics (
    metric_id INTEGER PRIMARY KEY,
    resource_id INTEGER,
    usage_date DATE,
    cpu_utilization_avg REAL, -- 0 to 1.0
    memory_usage_gb REAL,
    data_egress_gb REAL,
    FOREIGN KEY (resource_id) REFERENCES cloud_resources(resource_id)
);

CREATE TABLE invoices (
    invoice_id INTEGER PRIMARY KEY,
    org_id INTEGER,
    billing_period TEXT, -- '2026-01'
    total_amount REAL,
    tax_amount REAL,
    payment_status TEXT, -- 'Paid', 'Pending', 'Overdue'
    FOREIGN KEY (org_id) REFERENCES organizations(org_id)
);

CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    org_id INTEGER,
    full_name TEXT,
    role_id INTEGER,
    last_login TIMESTAMP,
    FOREIGN KEY (org_id) REFERENCES organizations(org_id)
);

CREATE TABLE roles (
    role_id INTEGER PRIMARY KEY,
    role_name TEXT, -- 'Admin', 'Developer', 'Billing_Reader'
    permission_level INTEGER -- 1 (Low) to 5 (Critical)
);"""

# 20 diverse SQL generation test questions ranging from basic queries to complex analytical tasks
QUESTIONS = [
    "List all organizations in the 'Healthcare' industry located in 'Germany'.",
    "Count the total number of 'Active' resources currently running in the 'us-east-1' region.",
    "Retrieve the names of all users who have a 'permission_level' of 5.",
    "Find the top 5 most expensive resources based on their hourly_cost.",
    "List all organizations that have 'Overdue' invoices.",
    "Calculate the total data_egress_gb for the entire month of February 2026.",
    "Display each organization name and the total number of users they have assigned to the 'Admin' role.",
    "Find the average cpu_utilization_avg for all 'Compute' resources over the last 7 days.",
    "Identify organizations that have at least one resource of every resource_type (Compute, Storage, and Database).",
    "Calculate the Effective Tax Rate for each organization (Tax Amount / Total Amount) for the '2026-01' period.",
    "List all resources that have never recorded any usage in the usage_metrics table.",
    "Find the total billing amount per industry for the first quarter of 2026.",
    "List resources where the cpu_utilization_avg was consistently below 10% (0.1) for the last 30 days.",
    "Find organizations that have 'Active' resources but have not been invoiced for the current billing period.",
    "Identify the Top User for each organization (the user who logged in most recently).",
    "For each region, find which resource_type accounts for the highest total hourly_cost.",
    "List 'Startup' tier organizations whose total resource spend in a single month exceeded $5,000.",
    "Calculate the percentage of organizations created in 2025 that still have 'Active' resources in 2026.",
    "Find resources whose data_egress_gb on any given day was more than 3 standard deviations above the average egress for that specific resource type.",
    "Create a monthly summary for each organization showing: Total Spend, Peak CPU utilization, and the Year-over-Year growth in data usage using the formula (Current_Usage - Previous_Usage) / Previous_Usage.",
]

USER_PROMPT_TEMPLATE = """Database Engine:
SQLite

Database Schema:
```sql
{schema}
```

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question.
- If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```"""


def build_user_prompt(question: str) -> str:
    """Build a user prompt by inserting schema and question into template.
    
    Args:
        question (str): SQL generation question to answer.
        
    Returns:
        str: Formatted prompt with schema, question, and instructions.
    """
    return USER_PROMPT_TEMPLATE.format(schema=SCHEMA, question=question)
