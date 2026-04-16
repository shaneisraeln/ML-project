# Smart Grid Dataset: Feature Contributions to Grid Stress

This document provides a comprehensive breakdown of every feature present in the `smart_grid_master_dataset.csv` and explains how they contribute to the target variable (`is_bad_stress`).

## 🎯 Target Variables

*   **`grid_stress`**: A continuous numerical index representing the real-time operational strain on the grid network. It isn't used as an input feature during predictive modeling to prevent data leakage.
*   **`is_bad_stress`**: The absolute target class used for our classification models. This is a binary variable (0 or 1) representing whether the grid is experiencing critical strain (calculated as `grid_stress` being in the 80th percentile or higher).

---

## ⚡ Direct Energy Generation & Loads

These features define the physical electricity entering and exiting the grid at any given moment.

*   **`grid_demand`**: The instantaneous, total electrical power required by consumers. 
    *   **Contribution**: *Highly Positive*. Spikes in grid demand directly push infrastructural limits, sharply increasing the probability of entering bad grid stress.
*   **`EV_load`**: The subset of grid demand attributed strictly to Electric Vehicle (EV) charging.
    *   **Contribution**: *Positive*. Widespread EV charging, particularly when uncoordinated or occurring during peak evening hours, creates massive, concentrated draws on the grid that can instigate bad stress.
*   **`solar`**: The total electrical generation supplied by solar farms/panels.
    *   **Contribution**: *Negative*. Higher solar generation naturally offsets base grid demand, feeding energy back into the system and significantly lowering the chances of grid stress.
*   **`wind`**: The total electrical generation supplied by wind turbines.
    *   **Contribution**: *Negative*. Similar to solar, higher wind yields provide clean baseline energy that reduces the burden on traditional power plants, mitigating stress.

---

## 📊 Derived & Rolling Metrics

Engineered features that capture proportions, momentums, and variations over time, helping the model understand *context* rather than just instantaneous snapshots.

*   **`renewable_share`**: The ratio calculated as `(solar + wind) / grid_demand`.
    *   **Contribution**: *Strong Negative*. Often one of the highest variance indicators. When grids run mostly on renewables, operational stress is very low. However, low renewable shares mean peaker-plants and external imports must be engaged, severely raising grid stress.
*   **`demand_lag_1h`**: Grid demand from exactly one hour prior.
    *   **Contribution**: *High Variance/Positive*. Grid stress has momentum. High demand an hour ago strongly predicts high demand (and potential failure limits) in the current hour.
*   **`demand_lag_24h`**: Grid demand from exactly 24 hours prior.
    *   **Contribution**: *High Variance/Positive*. Helps the model account for the cyclic nature of human energy consumption (a hot Monday afternoon maps well to a hot Tuesday afternoon).
*   **`demand_rolling_mean_24h`**: The average demand over the past 24 hours.
    *   **Contribution**: *Positive*. Indicates sustained strain. While short spikes can be mitigated with batteries, a high 24h rolling mean implies heavy sustained base-load (like during extreme heatwaves), heightening vulnerability.
*   **`demand_rolling_std_24h`**: The volatility/standard deviation of demand over 24 hours.
    *   **Contribution**: *Positive*. High volatility forces grid operators to quickly spin up/shut down peaker plants. Erratic demand is harder to fulfill efficiently and increases operational grid stress.
*   **`solar_change`**: Hour-over-hour difference in solar generation.
    *   **Contribution**: *Mixed/Contextual*. A sharply negative change (the sun setting) without a concurrent drop in demand leads to the dangerous "Duck Curve," rapidly spiking grid stress.
*   **`wind_change`**: Hour-over-hour difference in wind generation.
    *   **Contribution**: *Mixed/Contextual*. Sudden unexpected drop-offs in wind put immense pressure on baseload reserve stations.

---

## 🕒 Temporal Variables

Time-based integers that inform the model precisely *when* an event is happening, allowing it to correlate human behavioral patterns to grid consequences.

*   **`hour`** (0-23): The hour of the day.
    *   **Contribution**: *Non-linear but significant*. Hours 17 to 21 (5 PM - 9 PM) generally have the highest contribution to grid stress due to the intersection of the sun setting (low solar) and people arriving home. Nighttime hours reduce stress.
*   **`day_of_week`** (0-6): Monday through Sunday.
    *   **Contribution**: *Moderate*. Helps separate commercial-heavy weekdays from residential-heavy weekends.
*   **`month`** (1-12): The month of the year.
    *   **Contribution**: *Moderate (Seasonal)*. Summer months (high AC use) and deep winter months (high heating use) historically correlate with higher probabilities of systemic grid stress. 
*   **`is_weekend`**: A binary flag explicitly marking Saturdays and Sundays.
    *   **Contribution**: *Negative*. Weekends typically experience softer load curves with fewer extreme industrial/commercial peaks, generally correlating with lower grid stress.
*   **`is_peak_hour`**: A binary flag outlining typical peak grid usage blocks.
    *   **Contribution**: *Strong Positive*. Serves as an express highway for the model to directly associate a specific timeframe with the highest historic likelihood of "bad stress".
