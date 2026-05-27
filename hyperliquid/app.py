"""
Hyperliquid Funding Rate Regime Dashboard
==========================================
Live regime detection on Hyperliquid perpetual funding rates using a
Hidden Markov Model — the same model architecture as the IB live dashboard,
now pointed at crypto's most transparent derivatives market.

Run with: streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Hyperliquid Regime Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border-radius: 8px; padding: 12px; border: 1px solid #30363d; }
    .regime-high { color: #f85149; font-weight: bold; font-size: 1.4em; }
    .regime-med  { color: #d29922; font-weight: bold; font-size: 1.4em; }
    .regime-low  { color: #7ee787; font-weight: bold; font-size: 1.4em; }
    .signal-box  { background-color: #161b22; border-radius: 8px; padding: 16px;
                   border: 1px solid #30363d; margin: 8px 0; }
    h1, h2, h3  { color: #58a6ff; }
    .stSelectbox label { color: #8b949e; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HYPERLIQUID API
# =============================================================================
HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_funding_history(coin: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch historical funding rates from Hyperliquid.

    Hyperliquid pays funding every hour at 1/8 of the 8-hour computed rate.
    We annualise by multiplying by 8760 (hours per year) for comparison.

    Parameters
    ----------
    coin  : asset symbol e.g. 'BTC', 'ETH', 'SOL'
    days  : number of days of history to fetch

    Returns
    -------
    DataFrame with columns: time, funding_rate, funding_rate_annualised
    """
    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    payload = {
        "type": "fundingHistory",
        "coin": coin,
        "startTime": start_ms
    }

    try:
        response = requests.post(HYPERLIQUID_API, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['funding_rate'] = df['fundingRate'].astype(float)

        # Annualise: hourly rate × 8760 hours per year
        df['funding_rate_annualised'] = df['funding_rate'] * 8760

        # 8-hour equivalent (standard market convention)
        df['funding_rate_8h'] = df['funding_rate'] * 8

        return df[['time', 'funding_rate', 'funding_rate_8h', 'funding_rate_annualised']].sort_values('time')

    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)  # Cache for 1 minute
def fetch_current_market_data() -> dict:
    """
    Fetch current market context for all perpetuals.
    Returns open interest, current funding, mark price.
    """
    try:
        response = requests.post(
            HYPERLIQUID_API,
            json={"type": "metaAndAssetCtxs"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if len(data) < 2:
            return {}

        meta    = data[0]['universe']
        contexts = data[1]

        result = {}
        for i, asset in enumerate(meta):
            if i < len(contexts):
                ctx = contexts[i]
                result[asset['name']] = {
                    'funding':       float(ctx.get('funding', 0)),
                    'open_interest': float(ctx.get('openInterest', 0)),
                    'mark_price':    float(ctx.get('markPx', 0)),
                    'premium':       float(ctx.get('premium', 0) or 0)
                }
        return result

    except Exception as e:
        return {}


# =============================================================================
# HIDDEN MARKOV MODEL — REGIME DETECTION
# =============================================================================
class FundingRateHMM:
    """
    Three-state Hidden Markov Model for funding rate regime classification.

    States
    ------
    0 = NEUTRAL   : funding near zero, balanced long/short positioning
    1 = BULLISH   : positive funding, longs paying — crowded long
    2 = EXTREME   : extreme positive or negative funding — mean reversion expected

    The model uses the same Bayesian update architecture as the IB live
    regime dashboard, adapted for funding rate dynamics rather than
    price bar volatility.

    Update step per observation
    ---------------------------
    1. Prior = transition_matrix @ current_belief
    2. Likelihood = Gaussian PDF(observed_funding | regime_params)
    3. Posterior = prior × likelihood, normalised
    4. Regime = argmax(posterior)
    """

    def __init__(self):
        self.n_states = 3
        self.state_names  = ['NEUTRAL', 'ELEVATED', 'EXTREME']
        self.state_colors = ['#7ee787',  '#d29922',   '#f85149']

        # Transition matrix — regimes are sticky
        # Rows = from, columns = to
        self.transition_matrix = np.array([
            [0.85, 0.12, 0.03],   # From NEUTRAL
            [0.15, 0.75, 0.10],   # From ELEVATED
            [0.05, 0.15, 0.80],   # From EXTREME
        ])

        # Emission parameters (calibrated from data)
        self.emission_means = np.array([0.0,   0.0003,  0.001])
        self.emission_stds  = np.array([0.0002, 0.0003, 0.001])

        # Belief state — starts uniform
        self.state_probs = np.array([1/3, 1/3, 1/3])
        self.current_state = 0

    def calibrate(self, funding_rates: np.ndarray):
        """
        Estimate emission parameters from data using percentile segmentation.

        Bottom 40% by absolute value → NEUTRAL
        40th–75th percentile         → ELEVATED
        Top 25% by absolute value    → EXTREME
        """
        abs_rates = np.abs(funding_rates)
        abs_rates = abs_rates[abs_rates > 0]

        if len(abs_rates) < 10:
            return

        p40 = np.percentile(abs_rates, 40)
        p75 = np.percentile(abs_rates, 75)

        neutral_mask  = abs_rates < p40
        elevated_mask = (abs_rates >= p40) & (abs_rates < p75)
        extreme_mask  = abs_rates >= p75

        for mask, i in [(neutral_mask, 0), (elevated_mask, 1), (extreme_mask, 2)]:
            regime_vals = funding_rates[mask] if len(funding_rates) == len(abs_rates) else abs_rates[mask]
            if mask.sum() >= 3:
                # Use absolute values for emission means to handle negative funding
                vals = abs_rates[mask]
                self.emission_means[i] = np.mean(vals)
                self.emission_stds[i]  = max(np.std(vals), 1e-7)

        # Estimate transition matrix from sequence
        # Assign each observation to nearest mean
        assignments = np.zeros(len(abs_rates), dtype=int)
        assignments[abs_rates >= p40] = 1
        assignments[abs_rates >= p75] = 2

        counts = np.zeros((3, 3))
        for t in range(1, len(assignments)):
            counts[assignments[t-1], assignments[t]] += 1

        for i in range(3):
            row_sum = counts[i].sum()
            if row_sum > 0:
                self.transition_matrix[i] = (counts[i] + 0.1) / (row_sum + 0.3)

        self.state_probs = np.array([1/3, 1/3, 1/3])

    def _gaussian_pdf(self, x: float, mean: float, std: float) -> float:
        """Gaussian probability density — emission likelihood."""
        coeff = 1.0 / (std * np.sqrt(2 * np.pi))
        return coeff * np.exp(-0.5 * ((x - mean) / std) ** 2)

    def update(self, funding_rate: float) -> int:
        """
        Bayesian regime update for one new funding rate observation.

        P(regime | data) ∝ P(data | regime) × P(regime)
        posterior        ∝ likelihood       × prior

        Uses absolute value of funding rate — both extreme positive and
        extreme negative funding signal crowded positioning.
        """
        abs_rate = abs(funding_rate)

        # Prior: where transition matrix says we should be
        prior = self.transition_matrix.T @ self.state_probs

        # Likelihood: how probable is this funding rate in each regime?
        likelihoods = np.array([
            self._gaussian_pdf(abs_rate, self.emission_means[i], self.emission_stds[i])
            for i in range(self.n_states)
        ])

        # Posterior: Bayes update
        posterior = prior * likelihoods
        total = posterior.sum()
        if total > 0:
            posterior /= total
        else:
            posterior = prior

        self.state_probs = posterior
        self.current_state = int(np.argmax(posterior))
        return self.current_state

    def classify_series(self, funding_rates: np.ndarray) -> np.ndarray:
        """Classify a full series of funding rates, returning regime for each."""
        self.calibrate(funding_rates)
        regimes = []
        for rate in funding_rates:
            regimes.append(self.update(rate))
        return np.array(regimes)

    def get_signal(self, funding_rate: float, regime: int) -> tuple:
        """
        Translate current regime and funding rate into a trading signal.

        Returns (signal_text, signal_color, confidence)
        """
        abs_rate = abs(funding_rate)
        direction = "LONG" if funding_rate > 0 else "SHORT"
        opposite  = "SHORT" if funding_rate > 0 else "LONG"

        if regime == 2:  # EXTREME
            return (
                f"FADE THE {direction} — {opposite} signal",
                "#f85149",
                f"{self.state_probs[2]*100:.0f}%"
            )
        elif regime == 1:  # ELEVATED
            return (
                f"ELEVATED {direction} BIAS — monitor for reversion",
                "#d29922",
                f"{self.state_probs[1]*100:.0f}%"
            )
        else:  # NEUTRAL
            return (
                "NEUTRAL — no strong directional signal",
                "#7ee787",
                f"{self.state_probs[0]*100:.0f}%"
            )


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def compute_regime_metrics(df: pd.DataFrame, regimes: np.ndarray) -> dict:
    """
    Compute statistics about regime distribution and mean reversion.

    Key metric: does funding rate in EXTREME regime revert faster than NEUTRAL?
    This validates the core trading thesis.
    """
    df = df.copy()
    df['regime'] = regimes
    df['abs_funding'] = df['funding_rate'].abs()

    # Forward reversion: after N periods, how much has funding reverted?
    for n in [4, 8, 24]:  # 4h, 8h, 24h lookahead
        df[f'forward_{n}h'] = df['funding_rate'].shift(-n)
        df[f'reversion_{n}h'] = df[f'forward_{n}h'] - df['funding_rate']

    metrics = {}
    for regime_id, regime_name in enumerate(['NEUTRAL', 'ELEVATED', 'EXTREME']):
        mask = df['regime'] == regime_id
        if mask.sum() < 5:
            continue

        regime_df = df[mask]
        metrics[regime_name] = {
            'count':       int(mask.sum()),
            'pct_time':    f"{mask.mean()*100:.1f}%",
            'mean_abs_funding': f"{regime_df['abs_funding'].mean()*100:.4f}%",
            'std_funding': f"{regime_df['funding_rate'].std()*100:.4f}%",
        }

        # Mean reversion speed
        for n in [4, 8, 24]:
            col = f'reversion_{n}h'
            if col in regime_df.columns:
                valid = regime_df[col].dropna()
                if len(valid) > 3:
                    # Negative reversion mean = funding reverted toward zero
                    # For extreme positive funding: we expect negative reversion
                    # Use correlation between level and subsequent change
                    level = regime_df['funding_rate'].loc[valid.index]
                    if len(level) > 3:
                        corr = np.corrcoef(level, valid)[0, 1]
                        metrics[regime_name][f'reversion_corr_{n}h'] = f"{corr:.3f}"

    return metrics


def forward_regression(df: pd.DataFrame, forward_hours: int = 8) -> dict:
    """
    Regress forward funding rate change against current level.
    Negative slope = mean reversion confirmed.
    """
    df = df.copy()
    df['future_rate'] = df['funding_rate'].shift(-forward_hours)
    df['rate_change'] = df['future_rate'] - df['funding_rate']
    df = df.dropna()

    if len(df) < 20:
        return {}

    from scipy import stats

    slope, intercept, r, p, se = stats.linregress(
        df['funding_rate'], df['rate_change']
    )

    return {
        'slope':     slope,
        'intercept': intercept,
        'r_squared': r**2,
        'p_value':   p,
        'n_obs':     len(df),
        'df':        df
    }


# =============================================================================
# CHARTS
# =============================================================================
def plot_funding_regimes(df: pd.DataFrame, regimes: np.ndarray,
                          hmm: FundingRateHMM, coin: str) -> go.Figure:
    """
    Main chart: funding rate time series coloured by regime.
    Background shading shows regime periods.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
        subplot_titles=[
            f"{coin} Funding Rate — Markov Regime Classification",
            "Regime Probability (P(EXTREME))"
        ]
    )

    colors = [hmm.state_colors[r] for r in regimes]

    # Funding rate bars coloured by regime
    fig.add_trace(go.Bar(
        x=df['time'],
        y=df['funding_rate_8h'] * 100,
        marker_color=colors,
        name='Funding Rate (8h)',
        opacity=0.8,
        hovertemplate='%{x}<br>Rate: %{y:.4f}%<extra></extra>'
    ), row=1, col=1)

    # Zero line
    fig.add_hline(y=0, line_color='#8b949e', line_width=1,
                  line_dash='dash', row=1, col=1)

    # Percentile bands
    p75 = df['funding_rate_8h'].quantile(0.75) * 100
    p25 = df['funding_rate_8h'].quantile(0.25) * 100
    fig.add_hline(y=p75, line_color='#f85149', line_width=1,
                  line_dash='dot', annotation_text='75th pct',
                  annotation_font_color='#f85149', row=1, col=1)
    fig.add_hline(y=p25, line_color='#7ee787', line_width=1,
                  line_dash='dot', annotation_text='25th pct',
                  annotation_font_color='#7ee787', row=1, col=1)

    # Regime probability (EXTREME state)
    # Recompute probability series
    hmm_prob = FundingRateHMM()
    hmm_prob.calibrate(df['funding_rate'].values)
    probs_extreme = []
    for rate in df['funding_rate'].values:
        hmm_prob.update(rate)
        probs_extreme.append(hmm_prob.state_probs[2])

    fig.add_trace(go.Scatter(
        x=df['time'],
        y=probs_extreme,
        fill='tozeroy',
        fillcolor='rgba(248,81,73,0.2)',
        line=dict(color='#f85149', width=1.5),
        name='P(EXTREME)',
        hovertemplate='%{x}<br>P(EXTREME): %{y:.2f}<extra></extra>'
    ), row=2, col=1)

    fig.add_hline(y=0.5, line_color='#d29922', line_width=1,
                  line_dash='dash', row=2, col=1)

    fig.update_layout(
        height=600,
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', family='JetBrains Mono, monospace'),
        showlegend=True,
        legend=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(color='#c9d1d9', size=13)
        ),
        margin=dict(l=60, r=40, t=60, b=40),
        xaxis2=dict(
            gridcolor='#30363d',
            showgrid=True
        )
    )

    fig.update_yaxes(
        gridcolor='#30363d',
        showgrid=True,
        zerolinecolor='#30363d'
    )

    fig.update_xaxes(gridcolor='#30363d', showgrid=True)

    return fig


def plot_regime_distribution(df: pd.DataFrame, regimes: np.ndarray,
                               hmm: FundingRateHMM) -> go.Figure:
    """Distribution of funding rates within each regime — validates separation."""
    fig = go.Figure()

    for i, (name, color) in enumerate(zip(hmm.state_names, hmm.state_colors)):
        mask = regimes == i
        if mask.sum() < 2:
            continue
        vals = df.loc[mask, 'funding_rate_8h'].values * 100

        fig.add_trace(go.Histogram(
            x=vals,
            name=name,
            marker_color=color,
            opacity=0.7,
            nbinsx=30,
            hovertemplate=f'{name}<br>Rate: %{{x:.4f}}%<br>Count: %{{y}}<extra></extra>'
        ))

    fig.update_layout(
        title='Funding Rate Distribution by Regime',
        xaxis_title='Funding Rate (8h, %)',
        yaxis_title='Count',
        barmode='overlay',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1, font=dict(color='#c9d1d9', size=13)),
        height=350,
        margin=dict(l=60, r=40, t=50, b=50)
    )
    fig.update_xaxes(gridcolor='#30363d', showgrid=True)
    fig.update_yaxes(gridcolor='#30363d', showgrid=True)
    return fig


def plot_reversion_scatter(reg_result: dict) -> go.Figure:
    """
    Scatter of 8h forward rate change vs current level.
    Negative slope confirms mean reversion.
    """
    if not reg_result or 'df' not in reg_result:
        return go.Figure()

    df  = reg_result['df']
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['funding_rate'] * 100,
        y=df['rate_change'] * 100,
        mode='markers',
        marker=dict(color='#58a6ff', size=4, opacity=0.5),
        name='Observations',
        hovertemplate='Current: %{x:.4f}%<br>8h Change: %{y:.4f}%<extra></extra>'
    ))

    # Regression line
    x_range = np.linspace(df['funding_rate'].min(), df['funding_rate'].max(), 100)
    y_pred  = reg_result['slope'] * x_range + reg_result['intercept']
    fig.add_trace(go.Scatter(
        x=x_range * 100,
        y=y_pred * 100,
        mode='lines',
        line=dict(color='#f85149', width=2),
        name=f"Regression (slope={reg_result['slope']:.2f}, R²={reg_result['r_squared']:.3f})"
    ))

    # y=0 line
    fig.add_hline(y=0, line_color='#8b949e', line_width=1, line_dash='dash')

    fig.update_layout(
        title=f"Mean Reversion: 8h Forward Change vs Current Funding",
        xaxis_title='Current Funding Rate (8h, %)',
        yaxis_title='8h Forward Change (%)',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1, font=dict(color='#c9d1d9', size=13)),
        height=350,
        margin=dict(l=60, r=40, t=50, b=50)
    )
    fig.update_xaxes(gridcolor='#30363d', showgrid=True)
    fig.update_yaxes(gridcolor='#30363d', showgrid=True)
    return fig


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # ----- HEADER -----
    st.markdown("""
    <h1 style='font-family: JetBrains Mono, monospace; color: #58a6ff;'>
        ◈ Hyperliquid Funding Rate Regime Dashboard
    </h1>
    <p style='color: #8b949e; font-size: 0.95em;'>
        Hidden Markov Model regime detection on live Hyperliquid perpetual funding rates.
        Identifies crowded positioning and mean reversion signals across three regimes:
        NEUTRAL, ELEVATED, and EXTREME.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ----- SIDEBAR -----
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        coin = st.selectbox(
            "Asset",
            ["BTC", "ETH", "SOL", "ARB", "AVAX", "DOGE", "MATIC", "LINK", "BNB", "OP"],
            index=0
        )

        days = st.slider("History (days)", min_value=7, max_value=90, value=30, step=7)

        st.markdown("---")
        st.markdown("### 📖 Model")
        st.markdown("""
        **Three states**
        - 🟢 NEUTRAL — funding near zero
        - 🟡 ELEVATED — moderate imbalance
        - 🔴 EXTREME — crowded, expect reversion

        **Bayesian update**
        ```
        posterior ∝ likelihood × prior
        ```
        Transition matrix keeps regime
        sticky — consistent with how
        markets actually cluster.

        **Signal logic**
        Extreme positive funding →
        longs paying → fade the long.

        Extreme negative funding →
        shorts paying → fade the short.
        """)

        st.markdown("---")
        st.markdown("### 🔗 Data")
        st.markdown("Live via [Hyperliquid API](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api)")
        st.markdown("Refreshes every 5 minutes")

        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()

    # ----- FETCH DATA -----
    with st.spinner(f"Fetching {coin} funding history..."):
        df = fetch_funding_history(coin, days)

    if df.empty:
        st.error("No data returned. Check your connection or try a different asset.")
        return

    # ----- FETCH LIVE MARKET DATA -----
    market_data = fetch_current_market_data()
    current_data = market_data.get(coin, {})

    # ----- RUN HMM -----
    hmm = FundingRateHMM()
    regimes = hmm.classify_series(df['funding_rate'].values)
    df['regime'] = regimes
    df['regime_name'] = [hmm.state_names[r] for r in regimes]

    # Current state
    current_rate   = df['funding_rate'].iloc[-1]
    current_regime = regimes[-1]
    signal_text, signal_color, confidence = hmm.get_signal(current_rate, current_regime)

    # ----- TOP METRICS ROW -----
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Current Funding (1h)",
            f"{current_rate*100:.4f}%",
            delta=f"{(current_rate - df['funding_rate'].iloc[-2])*100:.4f}%"
        )

    with col2:
        st.metric(
            "8h Equivalent",
            f"{current_rate*8*100:.4f}%"
        )

    with col3:
        regime_name = hmm.state_names[current_regime]
        regime_color_map = {'NEUTRAL': '🟢', 'ELEVATED': '🟡', 'EXTREME': '🔴'}
        st.metric(
            "Current Regime",
            f"{regime_color_map[regime_name]} {regime_name}"
        )

    with col4:
        st.metric("Model Confidence", confidence)

    with col5:
        if current_data:
            oi = current_data.get('open_interest', 0)
            st.metric("Open Interest", f"${oi:,.0f}")
        else:
            st.metric("Data Points", f"{len(df):,}")

    st.markdown("---")

    # ----- SIGNAL BOX -----
    st.markdown(f"""
    <div class='signal-box'>
        <span style='color:#8b949e; font-size:0.85em;'>TRADING SIGNAL</span><br>
        <span style='color:{signal_color}; font-weight:bold; font-size:1.3em;'>
            {signal_text}
        </span>
        <br><span style='color:#8b949e; font-size:0.85em;'>
            Model confidence: {confidence} &nbsp;|&nbsp;
            {len(df)} observations over {days} days &nbsp;|&nbsp;
            Last updated: {df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ----- MAIN CHART -----
    st.plotly_chart(
        plot_funding_regimes(df, regimes, hmm, coin),
        use_container_width=True
    )

    # ----- ANALYSIS ROW -----
    col_left, col_right = st.columns(2)

    with col_left:
        st.plotly_chart(
            plot_regime_distribution(df, regimes, hmm),
            use_container_width=True
        )

    with col_right:
        reg_result = forward_regression(df, forward_hours=8)
        if reg_result:
            fig_rev = plot_reversion_scatter(reg_result)
            st.plotly_chart(fig_rev, use_container_width=True)

            slope = reg_result['slope']
            r2    = reg_result['r_squared']
            p     = reg_result['p_value']
            if slope < 0:
                st.markdown(f"""
                <div class='signal-box'>
                    ✅ <strong>Mean reversion confirmed</strong><br>
                    Slope = {slope:.3f} (negative = reversion)<br>
                    R² = {r2:.3f} &nbsp;|&nbsp; p = {p:.4f}<br>
                    <span style='color:#8b949e; font-size:0.85em;'>
                    High funding predicts lower future funding.
                    Statistically {"significant" if p < 0.05 else "weak"} relationship.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='signal-box'>
                    ⚠️ <strong>No clear mean reversion in this window</strong><br>
                    Slope = {slope:.3f} | R² = {r2:.3f}<br>
                    <span style='color:#8b949e; font-size:0.85em;'>
                    Try a longer history window.
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # ----- REGIME STATISTICS TABLE -----
    st.markdown("### Regime Statistics")

    metrics = compute_regime_metrics(df, regimes)
    if metrics:
        rows = []
        for regime_name, m in metrics.items():
            row = {
                'Regime':            regime_name,
                'Time in Regime':    m.get('pct_time', 'N/A'),
                'Observations':      m.get('count', 'N/A'),
                'Mean |Funding|':    m.get('mean_abs_funding', 'N/A'),
                'Funding Std':       m.get('std_funding', 'N/A'),
                '4h Reversion Corr': m.get('reversion_corr_4h', 'N/A'),
                '8h Reversion Corr': m.get('reversion_corr_8h', 'N/A'),
                '24h Reversion Corr':m.get('reversion_corr_24h', 'N/A'),
            }
            rows.append(row)

        stats_df = pd.DataFrame(rows)
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        <span style='color:#8b949e; font-size:0.85em;'>
        Reversion correlation: correlation between current funding rate and subsequent change.
        More negative = stronger mean reversion in that regime. EXTREME should show the most negative values.
        </span>
        """, unsafe_allow_html=True)

    # ----- RAW DATA EXPANDER -----
    with st.expander("📊 Raw Data"):
        display_df = df[['time', 'funding_rate', 'funding_rate_8h',
                          'funding_rate_annualised', 'regime_name']].copy()
        display_df.columns = ['Time', 'Hourly Rate', '8h Equivalent',
                               'Annualised', 'Regime']
        for col in ['Hourly Rate', '8h Equivalent', 'Annualised']:
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.5f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ----- FOOTER -----
    st.markdown("---")
    st.markdown("""
    <div style='color:#8b949e; font-size:0.8em;'>
    <strong>Model:</strong> 3-state Hidden Markov Model with Gaussian emissions and Bayesian posterior update.
    Calibrated from historical data using percentile-based regime segmentation and empirical transition matrix estimation.
    <br><br>
    <strong>Signal logic:</strong> Extreme funding rates are economically unsustainable —
    longs paying high positive funding will eventually close positions, driving price and funding lower.
    The HMM identifies when funding has entered the EXTREME regime where this reversion is most likely.
    <br><br>
    <strong>Data:</strong> Hyperliquid public API — no authentication required, updates every hour.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
