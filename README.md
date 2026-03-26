# A/B Testing Intelligence Platform
Welcome to the A/B Testing Intelligence Platform. This interactive dashboard is designed to take you from raw experiment data to statistically sound business decisions in minutes.
This guide will walk you through navigating the platform, configuring your test parameters, and interpreting the results across 7 analytical modules.

# Step 1: Configuration & Data 
 UploadAll global settings for your analysis are managed in the left-hand sidebar.Select Data Source: 
 * Use Kaggle Dataset: Loads the default Kaggle Marketing A/B Test dataset to help you explore the platform's features.
 * Upload Your Own CSVs: Allows you to upload custom control_group.csv and test_group.csv files.( Note: Files must be semicolon-delimited (;) and contain matching column headers like Spend [USD], Impressions, and Purchase.)
 * Set Test Parameters: Adjust the Significance Level ($\alpha$) slider (between 0.01 and 0.10) to define the probability threshold for rejecting the null hypothesis. This dynamically updates the confidence level across    all statistical tests
 * Choose Primary Metric: Select the core KPI (e.g., Purchase, Add to Cart, Reach) that determines the ultimate "winner" of your experiment. 
            
<img width="209" height="550" alt="step1" src="https://github.com/user-attachments/assets/507cc0f0-0e7a-4aa0-9678-9336f3abf339" />

# Step 2: Navigating the 7 Analysis Modules
 Once your data is loaded, navigate through the tabs at the top of the main window to conduct a full-spectrum analysis.
 <img width="1911" height="946" alt="step2" src="https://github.com/user-attachments/assets/007e6ac7-f1a2-45cd-b4b9-66dd2333e674" />

  ## 1. Overview
  Start here for a high-level executive summary of your campaign.
  
  Campaign Snapshot: View top-line KPIs (Spend, Impressions, Clicks, Purchases) and their percentage delta vs. the control group.Head-to-Head & Relative 
  
  Performance: Review derived metrics like CTR and Cost/Click in a tabular format, and use the bar chart to visualize exact uplift percentages across the funnel.
  
  Daily Trend: Verify traffic and metric stability over time using the interactive line chart.
  <img width="1568" height="886" alt="1" src="https://github.com/user-attachments/assets/92eefe51-a4c5-42da-8fe7-15e71142e2d4" />

 ## 2. Metric Deep Dive
  Inspect the underlying distribution of your data before trusting the statistical outputs. Select any metric to view its daily distribution via Histograms and Box Plots. Track how the metric evolved using the Cumulative Average chart. Use the Correlation Heatmap to see how different user actions relate to one another.
  <img width="1627" height="889" alt="2" src="https://github.com/user-attachments/assets/9035359a-d89e-4a36-b871-c355ef1373fc" />

 ## 3. Statistical Tests (Frequentist)
  Evaluate if the differences between your campaigns are statistically significant. The platform runs a Welch's t-test on your selected metrics, outputting $p$-values, confidence intervals, and Cohen's $d$ (effect size).
   Look for the Verdict Boxes: **Green** indicates a significant win for the test, **red**  for the control, and **orange** for an insignificant difference.
  <img width="1864" height="903" alt="3" src="https://github.com/user-attachments/assets/6f32123a-b41b-485a-9b7b-06b2ee7a53cc" />

 ## 4. Bayesian Analysis
  Translate rigid $p$-values into intuitive probabilities using a Beta-Binomial model.P(Test > Control): See the exact probability that your test variant is superior.Expected Loss: Understand the financial or conversion risk of choosing the wrong variant.ROPE Analysis: Adjust the Region of Practical Equivalence to determine if the difference, even if real, is large enough to actually care about.
  <img width="1870" height="920" alt="4" src="https://github.com/user-attachments/assets/cf589b42-2679-4794-b901-3f24b6cc8633" />
  
 ## 5. Sequential Monitor
 If you are monitoring a test that is currently running, use this tab to safely check results without inflating your false-positive rate.The chart plots your daily Z-scores against O'Brien-Fleming boundaries.If your Z-score crosses the boundary, the platform will trigger a ⚡ Early Stop alert, indicating you can safely conclude the experiment.
 <img width="1866" height="778" alt="5" src="https://github.com/user-attachments/assets/542ad6b3-0086-4521-a8d0-390721c6e4c4" />

 ## 6. Funnel Analysis
 Identify exactly where users are dropping off in the conversion journey. Compare the interactive funnel visualizations for both the Control and Test groups.Review the Stage Drop-off Rates table to pinpoint micro-conversion friction (e.g., View Content $\rightarrow$ Add to Cart).
 <img width="1866" height="884" alt="6" src="https://github.com/user-attachments/assets/26f9a9d3-a3ad-4db5-9df2-ca00cae32006" />

 ## 7. Sample Size Planner
 Plan your next experiment before launching it. Input your target Minimum Detectable Effect (MDE) and desired Power. The engine uses your current baseline mean and standard deviation to calculate the exact number of participants needed per group.  Consult the Quick Reference Table for multiple MDE and Power scenarios.
 <img width="1864" height="887" alt="7" src="https://github.com/user-attachments/assets/76720d68-942b-470c-9c34-a3d17312b271" />
