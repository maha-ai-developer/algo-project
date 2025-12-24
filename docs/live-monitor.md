
# 🧠 Live Assumption Health Dashboard

## (For Cointegration-Based Trading Agents)

---

## 1️⃣ Dashboard Purpose (non-negotiable)

The dashboard answers **one question only**:

> **“Is this agent currently allowed to trade?”**

Everything else is secondary.

---

## 2️⃣ Assumptions to Monitor (explicitly)

Your system relies on **six core assumptions**:

| #  | Assumption                | Why it matters            |
| -- | ------------------------- | ------------------------- |
| A1 | Data integrity            | Garbage in = fake signals |
| A2 | Correlation stability     | Economic similarity       |
| A3 | Hedge ratio stability (β) | Valid equilibrium         |
| A4 | Residual stationarity     | Mean reversion exists     |
| A5 | Mean reversion speed      | Capital efficiency        |
| A6 | Execution feasibility     | Theory → reality          |

Each assumption gets its **own health signal**.

---

## 3️⃣ Dashboard Structure (top-down)

```text
┌──────────────────────────────────┐
│ GLOBAL STATUS: 🟢 / 🟡 / 🔴        │
├──────────────────────────────────┤
│ Assumption Health Grid            │
├──────────────────────────────────┤
│ Time-Series Diagnostics           │
├──────────────────────────────────┤
│ Pair-Level Drilldown              │
├──────────────────────────────────┤
│ Auto-Actions & Logs               │
└──────────────────────────────────┘
```

---

## 4️⃣ Global Status Indicator (most important)

### Traffic-Light Logic

| Status    | Meaning                  | Action           |
| --------- | ------------------------ | ---------------- |
| 🟢 GREEN  | All assumptions valid    | Trading allowed  |
| 🟡 YELLOW | One assumption weakening | Trade size ↓     |
| 🔴 RED    | One or more broken       | Trading disabled |

This indicator is **computed**, not manual.

---

## 5️⃣ Assumption Health Grid (core panel)

Each row = one assumption
Each column = live metric + threshold

### Example

| Assumption     | Metric           | Current  | Threshold | Status |
| -------------- | ---------------- | -------- | --------- | ------ |
| Data Integrity | Missing bars     | 0        | ≤1        | 🟢     |
| Correlation    | 60-day rolling ρ | 0.74     | ≥0.65     | 🟢     |
| β Stability    | β drift (%)      | 6.1%     | ≤10%      | 🟢     |
| Stationarity   | ADF p-value      | 0.03     | <0.05     | 🟢     |
| Mean Reversion | Half-life        | 4.2 days | ≤8 days   | 🟢     |
| Execution      | Avg slippage     | 0.07%    | ≤0.10%    | 🟡     |

👉 **Any 🔴 → system halt**

---

## 6️⃣ Time-Series Diagnostics (visual truth)

These plots are **not decorative**. Each answers a binary question.

### 6.1 Residual Time Series

* Shows equilibrium deviations
* Visual detection of regime shifts

**Red flag patterns**

* Persistent drift
* Variance explosion
* Trend formation

---

### 6.2 Z-Score Over Time

* Confirms statistical symmetry
* Detects distribution distortion

**Red flags**

* Frequent |Z| > 3
* Asymmetric tails

---

### 6.3 Rolling ADF p-Value

* The *heartbeat* of the strategy

Rule:

```text
ADF p-value > 0.10 for N consecutive windows → DISABLE
```

---

## 7️⃣ Pair-Level Drilldown (when something breaks)

Clicking a pair opens:

### Pair Summary

* Assets
* Sector
* Active / Disabled
* Last trade timestamp

### Assumption Timeline

```text
Correlation → Regression → Residual → ADF → Z-score
```

You can **see exactly where it failed**.

---

## 8️⃣ Auto-Actions Panel (agent autonomy)

This is where your agent becomes **self-aware**.

### Example Rules

```text
IF ADF fails 3 times consecutively
→ Disable pair for 20 days

IF β drift > 15%
→ Force close positions

IF execution slippage spikes
→ Switch to LIMIT orders or pause
```

All actions are **logged**, timestamped, and auditable.

---

## 9️⃣ Logs & Explainability (crucial)

Every decision generates a log entry:

```json
{
  "timestamp": "2025-01-12 10:21",
  "pair": "A–B",
  "event": "PAIR_DISABLED",
  "reason": "ADF p-value > 0.10 for 5 windows",
  "action": "Trading halted"
}
```

This makes the system:

* explainable
* debuggable
* regulator-friendly

---

## 🔟 Health Score (optional but elegant)

You can compute a **composite health score**:

[
H = \sum w_i \cdot h_i
]

Where:

* ( h_i \in [0,1] ) is normalized assumption health
* ( w_i ) reflects importance

Example:

```text
H > 0.85 → Full size
0.70 < H ≤ 0.85 → Reduced size
H ≤ 0.70 → No trading
```

---

## 11️⃣ What this dashboard prevents (quietly)

It prevents:

* trading during regime shifts
* emotional overrides
* silent model decay
* “it worked before” bias

---

## 🧠 Deep alignment with my intuition

my thoughts:

> *“Human brain cannot hold paradox — but systems can.”*

This dashboard **holds the paradox for you**:

* Trade **only when structure exists**
* Stop **without ego when it disappears**

That’s not trading.
That’s **epistemic humility encoded in software**.

---

