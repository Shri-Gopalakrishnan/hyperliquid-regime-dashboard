# Hyperliquid Funding Rate Regime Dashboard

Live regime detection on Hyperliquid perpetual funding rates using a Hidden Markov Model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens in your browser at http://localhost:8501

## Deploy free on Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Set main file path to `app.py`
5. Deploy — get a public URL instantly

No API keys needed. Hyperliquid API is fully public.

## What it does

Pulls live hourly funding rate history from Hyperliquid for any perpetual asset (BTC, ETH, SOL etc), runs a calibrated Hidden Markov Model to classify each observation into one of three regimes, and surfaces a trading signal based on the current regime.

**NEUTRAL** — funding near zero, balanced positioning, no strong signal  
**ELEVATED** — moderate imbalance, monitor for escalation  
**EXTREME** — crowded positioning, mean reversion expected, fade the direction  

## The model

Three-state HMM with Gaussian emissions and Bayesian posterior update.

On each new hourly funding rate observation:
1. Prior = transition matrix weighted by current belief state
2. Likelihood = Gaussian PDF of observed rate under each regime's emission distribution
3. Posterior = prior × likelihood, normalised
4. Regime = argmax(posterior)

Parameters calibrated from historical data using percentile segmentation and empirical transition matrix estimation.

## Why funding rates

Extreme funding rates are economically unsustainable. When positive funding is very high, longs are paying shorts a significant hourly fee. This creates pressure for longs to close — which pushes price and funding lower. The HMM identifies when funding has entered the EXTREME regime where this reversion is structurally likely.

This is the same mean reversion logic applied in the IV regime analysis dashboard, now applied to crypto's most transparent signal.
