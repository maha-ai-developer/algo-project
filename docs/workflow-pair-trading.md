
# /workflow: Pair Trading Engine (Implementation Spec)

**Goal:** Execute the "Varsity" pair trading strategy using the exact data structures provided in the `Pair-Data` and `Tracker` files.

### **Phase 1: Data Structure Definition**

* **Input:** Historical Closing Prices (200 days).
* **Output Schema (Pair-Data.csv):**
The system must generate a "Master Sheet" with these exact columns:
* `sector`: Sector grouping (e.g., "Auto-4 wheelers").
* `yStock`: The Dependent Variable.
* `xStock`: The Independent Variable.
* `intercept`: Regression Constant ().
* `beta`: Regression Slope ().
* `adf_test_P.val`: Stationarity metric (Target ).
* `std_err`: **Crucial Distinction** — In your files, this is **Current Z-Score** (Current Residual / Sigma), *not* the statistical standard error.
* `sigma`: The Standard Deviation of the historical residuals (Statistical Standard Error).



### **Phase 2: Regression & Selection Logic**

1. **Bi-Directional Sweep:**
* Run OLS for  and .


2. **Selection Metric:**
* Select the pair direction with the lowest **Error Ratio**.
* *Formula:* .


3. **Filtration:**
* Discard pairs where `adf_test_P.val` > 0.05.
* *Check:* Ensure `intercept` is not explaining >80% of the price (as noted in Chapter 14).



### **Phase 3: Live Tracking & Signal Calculation**

* **Frequency:** Tick-by-tick or Minute-level (based on `Logs.csv`).
* **Calculation Steps (from `position-tracker.csv`):**
1. **Fetch Live Prices:**  and .
2. **Calculate Expected Y:** .
3. **Calculate Live Residual:** .
4. **Calculate Z-Score (`std_err`):** .



### **Phase 4: Execution Triggers**

* **Long Signal:**
* 
**Condition:** Z-Score .


* **Action:** Buy Y / Sell X.


* **Short Signal:**
* 
**Condition:** Z-Score .


* **Action:** Sell Y / Buy X.



### **Phase 5: Position Sizing (Beta Neutrality)**

* **Logic (from `position-tracker.csv`):**
* The goal is to be **Beta Neutral**.
* **Formula:** .


* **Futures Adjustment:**
* Since Futures have fixed lot sizes, calculate the ratio: .
* Round to the nearest whole lot for X.
* *Example from text:* If  (Tata Motors) is 1500 qty and  is 1.59, required  is ~2385. If  lot size is 2400, trade **1 Lot Y** and **1 Lot X**.





### **Phase 6: Trade Lifecycle Management**

* **Monitoring:** Log `Fut(X)`, `Fut(Y)`, and `Z-Score` continuously (as seen in `Logs.csv`).
* **Exit Rules:**
* 
**Target:** Z-Score reverts to .


* 
**Stop Loss:** Z-Score expands to .





