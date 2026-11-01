# Volatility forecasting: design

Written and committed before any of these models were run, so the choices
below could not be steered by the results.

## Question

The main study found that next-day *returns* of gold, silver and copper
cannot be forecast better than "no change", while a simple HAR model does
forecast the *size* of the next move. So the fair test of whether neural
networks add anything is on volatility, where there is something to learn:

> With the same information, do an LSTM, a Transformer or a pretrained
> time-series foundation model forecast next-day variance better than HAR?

## Target and data

- Target: the squared next-day log return, r²(t+1), settlement to
  settlement. It is a noisy but unbiased proxy for the variance of the
  next day's return. Every model forecasts that variance, h(t+1), using
  data up to and including day t.
- Test years 2020-2024, walk-forward by year: every model is refit on data
  from March 2010 to the end of the previous year (the same sample as the
  event study and the ablation) and then forecasts each day of the test
  year. Nothing is tuned on the test years.

## Models

| Model | What it is |
|---|---|
| Historical mean | mean of r² over the training window |
| EWMA | RiskMetrics, λ = 0.94, no fitting |
| HAR | OLS of r²(t+1) on r²(t) and the mean of r² over the last 5 and 22 days (Corsi 2009) |
| HAR-X | HAR plus the extra inputs the networks get: squared high-low range, VIX² (lagged one day), and flags for a CPI, payroll or FOMC release inside the forecast window |
| GARCH | GARCH(1,1), Student-t errors, zero mean; parameters refit each year, then filtered through the test year |
| GJR-GARCH | as GARCH plus a leverage term |
| LSTM | 22-day window of the inputs below, one LSTM layer of 32 units, dense layer of 16, output log h |
| Transformer | same inputs, linear embedding to 32 with positional encoding, two encoder blocks (4 heads, feed-forward 64, dropout 0.1), last time step, dense layer of 16, output log h |
| Chronos-Bolt small, base | pretrained foundation models, zero-shot: context of the last 512 squared returns, mean forecast; no training on these data |

Network inputs per day: log return, absolute return, high-low range,
5- and 20-day rolling volatility, ATR as a share of price, VIX and its
change, and the three release flags. Scaled with statistics from the
training window only and clipped at 5 standard deviations.

Network training, fixed in advance with no search: trained directly on
QLIKE (below) with log h as output, AdamW (learning rate 1e-3, weight decay
1e-4, gradient clipping at 1), batch 64, up to 60 epochs, early stopping
with patience 8 on the last 15% of each training window, three seeds whose
variance forecasts are averaged.

All forecasts are floored at 1e-8 (a daily volatility of 0.01%).

## Evaluation

- Primary loss: QLIKE, L = r²/h + log h. It is robust to the noise in the
  r² proxy for ranking forecasts (Patton 2011) and punishes forecasting too
  little risk more than too much.
- Secondary loss: mean squared error of h against r².
- Main comparison: every model against HAR, Diebold-Mariano test on the
  QLIKE differences with Newey-West variance (5 lags), one-sided, Holm
  correction across all model and metal pairs.
- HAR-X against HAR shows what the extra information is worth in a linear
  model. LSTM and Transformer against HAR-X show what the network adds on
  the same information.
- Model Confidence Set (90%, T_max, block bootstrap) on QLIKE per metal,
  over all models.
- Breakdowns by test year and for release days against other days.

## What I expect

HAR is a hard benchmark for daily volatility and GARCH is usually close to
it. I expect HAR-X to improve on HAR, mostly through the high-low range,
and the networks to match HAR-X at best. The foundation models see only the
squared returns, so I expect them to land near EWMA. Whatever the result,
it is reported as it comes out.
