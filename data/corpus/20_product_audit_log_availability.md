# Audit Log Availability

**Owner:** Security Product Team
**Last reviewed:** 2026-06-08
**Applies to:** Northwind Systems employees and relevant customer workspaces

## Purpose

This reference defines event coverage, retention by plan, and export behavior for audit logs. It is the controlled internal source for this topic. Northwind Systems uses it to make decisions consistently across Portland, Austin, Dublin, and remote teams. Readers should check the owner and review date before acting, particularly when a customer contract, regional rule, or active incident may affect the normal workflow.

## Authoritative facts

- Starter retains audit events for 30 days.
- Growth retains audit events for 90 days.
- Enterprise retains audit events for 365 days.
- Audit log timestamps are stored in UTC and exports use newline-delimited JSON.

These values are deliberate operating limits, not examples. Teams must not round a deadline, substitute a different threshold, or promise an exception unless the listed owner approves it through the documented channel. Where two records appear to disagree, the signed customer agreement controls commercial terms and the newer approved internal policy controls general operations.

## Operating procedure

Product, Sales, Support, and Finance use this reference when describing the service to customers. Contract-specific terms may change a commercial entitlement only when the signed order form states the change explicitly. Product Operations records approved exceptions against the workspace. Customer-facing answers must quote the applicable plan and effective date instead of relying on memory or an old sales deck.

Before closing the work, the responsible person verifies the relevant identifier, date, amount, plan, region, or severity and attaches enough evidence for another employee to reproduce the decision. Automated reminders support the process but do not replace owner accountability. A missed target is escalated on the same business day rather than hidden by changing a timestamp.

## Exceptions and escalation

Exceptions are narrow, time-bound, and written. The request describes the business reason, risk, affected people or workspaces, proposed end date, and the control that reduces exposure. Silence is not approval. Emergency action may proceed to protect people, customer data, or service availability, but the acting lead records the decision within one business day and informs the policy owner.

## Evidence and maintenance

The owning team validates these facts during the monthly catalog review and before pricing or entitlement changes. Evidence comes from billing configuration, product telemetry, signed order forms, and release records. Status incidents do not silently amend a subscription term. Proposed changes require Product, Finance, Support, and Legal review before the effective date is announced.
