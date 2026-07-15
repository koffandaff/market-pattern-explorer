# Market Pattern Explorer

**I focused on learning and understanding the behaviour of features instead of trying to make a successful model.**

This is the central philosophy of this project. We did not set out to build a production-ready trading bot or a profitable strategy. Instead, we set out to understand *why* market prediction is hard, *how* features interact, and *where* models fail. The models themselves are tools for probing the data, not the end goal.

---

## What I Tried

I asked a simple question: **Can historical price data and technical indicators predict whether AAPL closes higher or lower tomorrow?**

I downloaded 5 years of Apple daily OHLCV data (Jan 2020 – Dec 2024), engineered 11 technical features from scratch, formulated binary classification targets, trained 4 models of increasing complexity, backtested them, and compared everything against a simple Buy & Hold benchmark.

The short answer: **No, not reliably.** The best model (LSTM) achieved 45.8% accuracy — barely above a coin flip. Every ML strategy was outperformed by Buy & Hold (+37% cumulative return). But the *why* behind these failures is where the real learning lives.

### Project Pipeline

```mermaid
graph LR
    A["Raw OHLCV<br/>(1258 rows)"] --> B["Feature Engineering<br/>11 technical indicators"]
    B --> C["Model Dataset<br/>16 features + Target<br/>(1209 rows)"]
    C --> D["Train/Test Split<br/>80/20"]
    D --> E["Scale (StandardScaler)"]
    E --> F["Train 4 Models"]
    F --> G["Evaluate Metrics<br/>Acc, Prec, Recall"]
    G --> H["Backtest Strategies"]
    H --> I{"Beat Buy & Hold?<br/>+37% return"}
    I -->|No| J["Learn Why"]
    I -->|No| K["Features > Models"]

    style A fill:#e8f4fd,stroke:#333
    style J fill:#fff3e0,stroke:#333
    style K fill:#e8f5e9,stroke:#333
```

---

## Key Learnings About Feature Engineering

### 1. The Noise-vs-Lag Tradeoff Is Everywhere

Every feature we engineered forced us to choose between **reactivity** and **reliability**:

```mermaid
graph LR
    subgraph Short Window
        S1["Return (1 day)"]
        S2["Change (1 day)"]
        S3["Momentum_5 (5 days)"]
    end
    subgraph Long Window
        L1["MA_50 (50 days)"]
        L2["Volatility (20 days)"]
    end

    S1 -->|"Reacts instantly<br/>But mostly noise"| Tradeoff
    L1 -->|"Smooth & reliable<br/>Lags ~25 days"| Tradeoff

    Tradeoff["NO FREE LUNCH<br/>You must pick one"]

    style Tradeoff fill:#fff3e0,stroke:#d84315,stroke-width:2px
```

- **Short-window features** (Return, Change, Momentum_5) react instantly to new price moves. But they are dominated by random noise — most day-to-day movement is meaningless.
- **Long-window features** (MA_50, Volatility) smooth away noise beautifully. But they lag reality by days or weeks. MA_50 trails price by ~25 days on average.
- There is **no free lunch**. You cannot have zero lag and zero noise simultaneously. Every modeling decision inherits this tradeoff from the feature engineering step.

### 2. Stationarity Matters More Than Complexity

Raw price is non-stationary — its mean and variance drift over time. Models trained on raw prices fail when the price level changes. **Percentage returns** solved this: they normalize price changes to be scale-independent and approximately stationary. This single transformation had more impact on model stability than switching from Logistic Regression to XGBoost.

```mermaid
graph LR
    A["Raw Price<br/>Non-stationary<br/>Mean drifts over time"] -->|"pct_change()"| B["Returns<br/>Stationary<br/>Mean ~ 0"]
    B --> C["Model trains on<br/>stable distribution"]
    A --> D["Model trains on<br/>shifting distribution"]
    D -->|"Fails when price<br/>level changes"| E["❌ Unstable"]
    C -->|"Works across<br/>all price levels"| F["✅ Stable"]

    style E color:#c62828
    style F color:#2e7d32
```

### 3. Feature Distributions Reveal the Market's Nature

Visualizing each feature taught us more than any model metric:

- **Returns** cluster near zero with fat tails — small moves dominate, big moves are rare.
- **RSI** rarely hits extreme zones (above 70 or below 30) during strong trends, limiting its signal value.
- **MACD components** are highly correlated with each other (MACD, Signal, Histogram) — including all three risks multicollinearity.
- **Volume_Change** is extremely volatile — 50%+ swings are common, making it a noisy predictor at daily frequency.

### 4. Feature Engineering Is Where the Signal Lives

Every model — from a simple linear classifier to a 53,825-parameter LSTM — received the *same* 16 features. The accuracy range across all models was just **40.1%–45.8%**.

```mermaid
graph LR
    subgraph Models
        LR["Logistic Regression<br/>11 parameters"]
        RF["Random Forest<br/>200 trees, depth 8"]
        XGB["XGBoost<br/>300 trees, lr=0.03"]
        LSTM["LSTM<br/>53,825 parameters"]
    end
    subgraph Input
        F["Same 16 Features"]
    end
    subgraph Result
        R["Acc: 40.1% – 45.8%<br/>Tight band"]
    end

    F --> LR
    F --> RF
    F --> XGB
    F --> LSTM
    LR --> R
    RF --> R
    XGB --> R
    LSTM --> R

    style R fill:#fff3e0,stroke:#d84315
```

This tight band tells us something profound: **the ceiling on performance is set by the features, not the model**. Adding better features (sentiment, order book, macro data) would likely improve performance far more than switching from XGBoost to a Transformer.

### 5. The Target Definition Shapes Everything

Our target was `(Close.shift(-1) > Close).astype(int)` — tomorrow up or not? This binary framing discards magnitude information. A day that closes +0.1% and a day that closes +5% are both labeled "1". A model that perfectly predicts direction but ignores size cannot produce a profitable strategy. This is a feature engineering decision (how we define the target) that directly causes the accuracy–profitability disconnect we later discovered in backtesting.

```mermaid
graph LR
    A["Close.tomorrow > Close.today?"] -->|Yes| B["Label: 1 (UP)<br/>+0.1% day = 1<br/>+5.0% day = 1"]
    A -->|No| C["Label: 0 (DOWN)<br/>-0.1% day = 0<br/>-5.0% day = 0"]
    B --> D["Model predicts direction<br/>but ignores magnitude"]
    C --> D
    D -->|"High accuracy possible<br/>but strategy loses money"| E["❌ Accuracy ≠ Profitability"]

    style E fill:#ffebee,stroke:#c62828
```

---

## The Data

| File                                                  | Description                                                                                                                 | Rows |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---- |
| [`data/btc_usd_daily.csv`](data/btc_usd_daily.csv)   | Raw AAPL OHLCV from Yahoo Finance (2020-01-02 to 2024-12-31)                                                                | 1258 |
| [`data/btc_features.csv`](data/btc_features.csv)     | Engineered features: Change, Return, Volume_Change, Momentum_5, MA_10, MA_50, Volatility, RSI, MACD, MACD_Signal, MACD_Hist | 1258 |
| [`data/btc_model_data.csv`](data/btc_model_data.csv) | Features + Target (binary direction label) after dropna()                                                                   | 1209 |

> **Note:** All three files are CSV. The file names contain "btc" but the data is Apple (AAPL) — a naming artifact from early exploration.

### Data Processing Flow

```mermaid
graph TD
    A["Raw OHLCV<br/>btc_usd_daily.csv<br/>1258 rows x 5 cols"] --> B["Feature Engineering<br/>11 new columns"]
    B --> C["btc_features.csv<br/>1258 rows x 16 cols"]
    C --> D["Add Target<br/>(shifted Close direction)"]
    D --> E["dropna()<br/>-49 rows with NaN<br/>(from rolling windows)"]
    E --> F["btc_model_data.csv<br/>1209 rows x 17 cols"]

    style A fill:#e3f2fd,stroke:#333
    style F fill:#e8f5e9,stroke:#333
```

---

## The Models

> **Benchmark:** Buy & Hold returned **+37%** with a Sharpe ratio of ~0.40. Every ML strategy underperformed this simple baseline.

| Model               | File                                                                | Accuracy                                       | Precision | Recall | Sharpe |
| ------------------- | ------------------------------------------------------------------- | ---------------------------------------------- | --------- | ------ | ------ |
| Logistic Regression | [`models/logistic_regression.pkl`](models/logistic_regression.pkl) | 40.91%                                         | 46.67%    | 20.14% | -1.94  |
| Random Forest       | [`models/random_forest.pkl`](models/random_forest.pkl)             | 43.0%                                          | 52.0%     | 17.0%  | 0.21   |
| XGBoost             | [`models/xgboost.pkl`](models/xgboost.pkl)                         | 40.08%                                         | 44.0%     | 15.83% | 0.13   |
| LSTM                | [`models/lstm_model.keras`](models/lstm_model.keras)               | 45.76%                                         | 53.47%    | 40.0%  | N/A    |
| Scaler              | [`models/scaler.pkl`](models/scaler.pkl)                           | StandardScaler (16 features, for classical ML) |           |        |        |

### Model Training Pipeline

```mermaid
graph TD
    subgraph Data Preparation
        A["Model Dataset<br/>1209 rows"] --> B["Train/Test Split<br/>80/20"]
        B --> C["Train: 967 rows"]
        B --> D["Test: 242 rows"]
    end

    subgraph Classical ML
        C --> E["Scale Features<br/>StandardScaler"]
        E --> F["Train Model<br/>(LR / RF / XGBoost)"]
        F --> G["Predict on Test Set"]
    end

    subgraph Deep Learning
        C --> H["Create Sequences<br/>30-day windows"]
        H --> I["Train LSTM<br/>64 units, dropout 0.2"]
        I --> J["Predict on Test Sequences"]
    end

    subgraph Evaluation
        G --> K["Classification Metrics<br/>Accuracy / Precision / Recall"]
        J --> K
        K --> L["Backtest Trading<br/>Strategy vs Buy & Hold"]
    end

    style L fill:#fff3e0,stroke:#d84315
```

---

## Step by Step: The Full Research Journey

### Step 1: Data Exploration — [`notebooks/01_dataexplorer.ipynb`](notebooks/01_dataexplorer.ipynb)

Downloaded AAPL daily data via `yfinance` for 2020–2024. Inspected structure, missing values (none), and basic statistics. The data was clean — no preprocessing needed beyond saving to CSV.

**Key cell — loading data:**

```python
import yfinance as yf
df = yf.download("AAPL", start="2020-01-01", end="2025-01-01", interval="1d")
```

> **Key observation:** `df.isna().sum()` returned all zeros. Clean data from the start.

```mermaid
graph LR
    A["yfinance<br/>AAPL"] -->|"download()"| B["1258 rows x 5 cols<br/>Shape: (1258, 5)"]
    B --> C["Missing values: 0<br/>Dtypes: all float64"]
    C --> D["Save to CSV<br/>btc_usd_daily.csv"]

    style A fill:#e3f2fd,stroke:#333
    style D fill:#e8f5e9,stroke:#333
```

### Step 2: Feature Engineering — [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb)

This was the heart of the project. We built 11 features from raw OHLCV, each with its own documented rationale, formula, visualization, and limitations:

| #  | Feature                 | Formula                      | Purpose                                |
| -- | ----------------------- | ---------------------------- | -------------------------------------- |
| 1  | **Change**        | `Close - Open`             | Candle sentiment — bullish or bearish |
| 2  | **Return**        | `Close.pct_change()`       | Scale-independent daily move           |
| 3  | **Volume_Change** | `Volume.pct_change()`      | Participation shift detection          |
| 4  | **Momentum_5**    | `Close - Close.shift(5)`   | 5-day price acceleration               |
| 5  | **MA_10**         | `Close.rolling(10).mean()` | Short-term trend                       |
| 6  | **MA_50**         | `Close.rolling(50).mean()` | Long-term trend                        |
| 7  | **Volatility**    | `Return.rolling(20).std()` | 20-day price fluctuation               |
| 8  | **RSI**           | 14-period RSI from`ta`     | Momentum oscillator (0-100)            |
| 9  | **MACD**          | `EMA_12 - EMA_26`          | Trend-momentum hybrid                  |
| 10 | **MACD_Signal**   | `EMA_9 of MACD`            | Signal line for crossovers             |
| 11 | **MACD_Hist**     | `MACD - Signal`            | Momentum acceleration                  |

**Key cell — computing RSI:**

```python
from ta.momentum import RSIIndicator
df['RSI'] = RSIIndicator(df['Close']).rsi()
```

Each feature includes a "Limitations" section in the notebook — e.g., for Change:

> *"A single candle does not describe the overall trend. Large candles may occur because of news or sudden volatility."*

```mermaid
graph LR
    OHLCV["Raw OHLCV<br/>Open, High, Low, Close, Volume"] --> FE["Feature Engineering"]
  
    FE --> C["Change<br/>Close - Open"]
    FE --> R["Return<br/>pct_change()"]
    FE --> VC["Volume_Change<br/>Volume.pct_change()"]
    FE --> M5["Momentum_5<br/>5-day change"]
    FE --> MA10["MA_10<br/>10-day avg"]
    FE --> MA50["MA_50<br/>50-day avg"]
    FE --> V["Volatility<br/>20-day std"]
    FE --> RSI["RSI<br/>14-period"]
    FE --> MACD["MACD<br/>EMA12 - EMA26"]
    FE --> MS["MACD_Signal<br/>9-day EMA of MACD"]
    FE --> MH["MACD_Hist<br/>MACD - Signal"]

    MA10 --> Combine["16 Features<br/>(5 OHLCV + 11 engineered)"]

    style OHLCV fill:#e3f2fd,stroke:#333
    style Combine fill:#e8f5e9,stroke:#333
```

### Step 3: Target Creation & Dataset Assembly — [`notebooks/03_target_and_dataset.ipynb`](notebooks/03_target_and_dataset.ipynb)

Formulated as binary classification:

```python
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
```

> **Key finding:** Class balance: ~53.5% UP, ~46.5% DOWN. Slightly imbalanced but not severely.

**Final feature set (16):** Open, High, Low, Close, Volume, Change, Return, Volume_Change, Momentum_5, MA_10, MA_50, Volatility, RSI, MACD, MACD_Signal, MACD_Hist.

```mermaid
graph TD
    A["16 Features<br/>(1209 rows)"] --> B["Target: tomorrow's direction"]
    B --> C["1 if Close.shift(-1) > Close<br/>0 otherwise"]
    A --> D["Model Dataset<br/>1209 x 17<br/>(16 features + 1 target)"]
    C --> D

    subgraph Class Balance
        E["53.5% UP"]
        F["46.5% DOWN"]
    end

    style D fill:#e8f5e9,stroke:#333
```

### Step 4: Logistic Regression Baseline — [`notebooks/05_logistic_regression.ipynb`](notebooks/05_logistic_regression.ipynb)

> **Rationale:** "Logistic Regression is simple, fast, and interpretable. It establishes a baseline performance against which more advanced models can be compared."

**Key output — coefficients revealed feature direction:**

```
Return coefficient: -0.30 (mean reversion signal)
```

When Return is very positive, the model predicts DOWN — large up-moves tend to reverse.

**Results:** Accuracy 40.91%, Precision 46.67%, Recall 20.14%.

### Step 5: Random Forest — [`notebooks/06_random_forest.ipynb`](notebooks/06_random_forest.ipynb)

> **Rationale:** "Logistic Regression assumes linearity. Markets are non-linear. Random Forest captures non-linear patterns."

**Configuration:** 200 estimators, `max_depth=8`. No feature scaling needed.

**Results:** Accuracy 43.0%, Precision 52.0%, Recall 17.0%. Slightly better accuracy and best precision. But recall dropped — RF caught fewer up-moves than LR.

### Step 6: XGBoost — [`notebooks/07_xgboost.ipynb`](notebooks/07_xgboost.ipynb)

> **Rationale:** "Each new tree attempts to correct the mistakes made by previous trees."

**Configuration:** 300 estimators, `learning_rate=0.03`, `max_depth=4`.

**Results:** Accuracy 40.08%, Precision 44.0%, Recall 15.83%. **Worst overall performance** — the sequential correction added variance without benefit on our small dataset (967 training rows).

### Step 7: Model Comparison — [`notebooks/08_model_comparison.ipynb`](notebooks/08_model_comparison.ipynb)

Side-by-side comparison with bar charts. The critical insight from this notebook:

> *"Model complexity did not improve performance. More advanced algorithms such as Random Forest and XGBoost did not significantly outperform Logistic Regression. This suggests that the limiting factor is not the choice of model but the information contained in the features."*

```mermaid
graph LR
    subgraph Models
        A["Logistic Regression"]
        B["Random Forest"]
        C["XGBoost"]
        D["LSTM"]
    end

    subgraph Performance
        A1["40.9%"]
        B1["43.0%"]
        C1["40.1%"]
        D1["45.8%"]
    end

    A --> A1
    B --> B1
    C --> C1
    D --> D1

    A1 --> Range["RANGE: 40.1% – 45.8%<br/>(just 5.7% spread)"]
    B1 --> Range
    C1 --> Range
    D1 --> Range

    Range --> Conclusion["Complexity ≠ Performance<br/>Features are the ceiling"]

    style Range fill:#fff3e0,stroke:#d84315
    style Conclusion fill:#e8f5e9,stroke:#2e7d32
```

**Four key findings:**

1. **Model complexity ≠ better performance** — LR, RF, XGBoost, and LSTM all landed within ~5% of each other
2. **Technical indicators alone provide limited predictive power** — no model crossed 50% accuracy
3. **The market signal appears weak** — all models near random guessing, suggesting efficient market effects at daily frequency
4. **Future work directions** — sentiment analysis, macroeconomic data, longer prediction horizons

### Step 8: Backtesting — [`notebooks/09_backtesting.ipynb`](notebooks/09_backtesting.ipynb)

This is where the project's most important lesson emerged. We simulated trading: predict UP → buy/hold, predict DOWN → stay in cash. No leverage, no transaction costs.

> **Key finding:** Cumulative returns chart showed **Buy & Hold (+37%) crushing every ML strategy**.

```mermaid
graph TD
    subgraph Trading Simulation
        A["Model predicts UP"] --> B["Buy and hold"]
        A1["Model predicts DOWN"] --> B1["Stay in cash"]
    end

    subgraph Results
        B --> C["Cumulative P&L tracked"]
        B1 --> C
    end

    subgraph Comparison
        C --> D["LR: Sharpe -1.94"]
        C --> E["RF: Sharpe 0.21"]
        C --> F["XGB: Sharpe 0.13"]
        C --> G["Buy & Hold: Sharpe ~0.40"]
    end

    G --> H["Buy & Hold wins<br/>+37% cumulative return"]

    style H fill:#e8f5e9,stroke:#2e7d32
    style D fill:#ffebee,stroke:#c62828
```

| Strategy            | Sharpe Ratio |
| ------------------- | ------------ |
| Logistic Regression | -1.94        |
| Random Forest       | 0.21         |
| XGBoost             | 0.13         |
| Buy & Hold          | ~0.40        |

> *"Even though Random Forest achieved better classification metrics, it failed to outperform simply holding Apple stock."*

### Step 9: LSTM Model — [`notebooks/10_lstm_model.ipynb`](notebooks/10_lstm_model.ipynb)

LSTM processes 30-day sequences rather than individual rows. Architecture: `LSTM(64 units) → Dropout(0.2) → Dense(1, sigmoid)`. **53,825 parameters on 967 training samples.**

**Results:** Accuracy 45.76% (best), Precision 53.47% (best), Recall 40.0% (best). But with 55x more parameters than training samples, overfitting risk is severe.

```mermaid
graph LR
    subgraph Sequence Creation
        A["Day 1-30"] --> B["LSTM Cell"]
        A1["Day 2-31"] --> B1["LSTM Cell"]
        A2["Day 3-32"] --> B2["LSTM Cell"]
    end

    subgraph Architecture
        B --> C["LSTM(64 units)"]
        B1 --> C
        B2 --> C
        C --> D["Dropout(0.2)"]
        D --> E["Dense(1, sigmoid)"]
    end

    subgraph Risk
        E --> F["53,825 params<br/>967 training samples<br/>55:1 ratio"]
        F --> G["⚠️ Severe overfitting<br/>risk"]
    end

    style G fill:#fff3e0,stroke:#d84315
```

### Step 10: Final Report — [`reports/final_report.ipynb`](reports/final_report.ipynb)

A comprehensive write-up consolidating all findings, including 3 key failed assumptions and future work directions.

---

## Where We Failed and How

### Failure 1: "More complexity = better results"

We assumed Random Forest would beat Logistic Regression, XGBoost would beat Random Forest, and LSTM would beat everything. The accuracy range across all 4 models was just **40.1%–45.8%**. Complexity added variance, not signal.

```mermaid
graph LR
    A["Assumption<br/>Complexity → Better"] --> B["Reality<br/>40.1% – 45.8%"]
    B --> C["Why?"]
    C --> D["Same 16 features<br/>fed to all models"]
    D --> E["Ceiling set by features<br/>not by algorithm"]

    style A fill:#ffebee,stroke:#c62828
    style E fill:#e8f5e9,stroke:#2e7d32
```

**Why:** The ceiling is set by the features, not the model. All models saw the same 16 features. No amount of algorithmic sophistication can extract information that simply isn't there.

### Failure 2: "Accuracy equals profitability"

Logistic Regression had 40.9% accuracy. Random Forest had 43.0%. By this measure, RF was "better." But in backtesting, LR's Sharpe was -1.94 (terrible) and RF's Sharpe was 0.21 (mediocre). Meanwhile Buy & Hold (+37% return, ~0.40 Sharpe) outperformed everything despite being "directionless" as a model.

```mermaid
graph LR
    subgraph Metrics
        A["RF: 43.0% accuracy"]
        B["LR: 40.9% accuracy"]
    end

    subgraph Reality
        C["RF: Sharpe 0.21"]
        D["LR: Sharpe -1.94"]
        E["Buy & Hold: Sharpe 0.40<br/>+37% return"]
    end

    A -->|"Looks better<br/>on paper"| C
    B --> D
    E --> F["Best strategy<br/>has zero 'accuracy'"]

    style F fill:#fff3e0,stroke:#d84315
```

**Why:** Accuracy ignores magnitude, risk, and market participation. A model that is wrong often but badly wrong can destroy capital faster than a model that is wrong slightly less often but is right in bigger moves.

### Failure 3: "Technical indicators are enough"

We used 16 features including RSI, MACD, moving averages, and volatility — a comprehensive technical toolkit. The models still couldn't beat Buy & Hold.

```mermaid
graph TD
    subgraph What We Used
        A["Price Data<br/>(OHLCV)"]
        B["Volume Data"]
        C["Technical Indicators<br/>(RSI, MACD, MA, etc.)"]
    end

    subgraph What We Missed
        D["News & Earnings"]
        E["Macroeconomic Data"]
        F["Order Flow"]
        G["Market Microstructure"]
        H["Sentiment Data"]
    end

    A --> I["ALL features derived<br/>from same source"]
    B --> I
    C --> I
    I --> J["Only captures<br/>historical patterns"]

    D --> K["External factors<br/>drive price"]
    E --> K
    F --> K
    G --> K
    H --> K

    style I fill:#ffebee,stroke:#c62828
    style K fill:#e3f2fd,stroke:#1565c0
```

**Why:** Technical indicators are derived from the same data (price and volume). They capture only historical patterns. They do not capture news, earnings, macroeconomic shifts, order flow, or market microstructure. In efficient markets, these external factors are what drive price.

---


---

## Project Structure

```
Market Pattern Explorer/
├── data/
│   ├── btc_usd_daily.csv         # Raw AAPL OHLCV
│   ├── btc_features.csv          # Engineered features
│   └── btc_model_data.csv        # Features + Target
├── models/
│   ├── logistic_regression.pkl   # Trained LR
│   ├── random_forest.pkl         # Trained RF (200 trees, depth 8)
│   ├── xgboost.pkl               # Trained XGB (300 trees, lr=0.03)
│   ├── lstm_model.keras          # Trained LSTM (64 units, 30-day sequences)
│   └── scaler.pkl                # StandardScaler (16 features)
├── notebooks/
│   ├── 01_dataexplorer.ipynb     # Data download + inspection
│   ├── 02_feature_engineering.ipynb  # 11 features from scratch
│   ├── 03_target_and_dataset.ipynb   # Target + feature selection
│   ├── 05_logistic_regression.ipynb  # Baseline model
│   ├── 06_random_forest.ipynb        # Non-linear ensemble
│   ├── 07_xgboost.ipynb              # Gradient boosting
│   ├── 08_model_comparison.ipynb     # Side-by-side metrics
│   ├── 09_backtesting.ipynb          # Trading simulation
│   └── 10_lstm_model.ipynb           # Deep learning with sequences
├── reports/
│   └── final_report.ipynb        # Comprehensive write-up
└── README.md
```

### Architecture Flow

```mermaid
graph TD
    subgraph notebooks
        N1["01_dataexplorer"]
        N2["02_feature_engineering"]
        N3["03_target_and_dataset"]
        N4["05_logistic_regression"]
        N5["06_random_forest"]
        N6["07_xgboost"]
        N7["08_model_comparison"]
        N8["09_backtesting"]
        N9["10_lstm_model"]
        N10["reports/final_report"]
    end

    subgraph data
        D1["btc_usd_daily.csv"]
        D2["btc_features.csv"]
        D3["btc_model_data.csv"]
    end

    subgraph models
        M1["logistic_regression.pkl"]
        M2["random_forest.pkl"]
        M3["xgboost.pkl"]
        M4["lstm_model.keras"]
        M5["scaler.pkl"]
    end

    N1 --> D1
    N2 --> D1
    N2 --> D2
    N3 --> D2
    N3 --> D3
    N4 --> D3
    N4 --> M1
    N5 --> D3
    N5 --> M2
    N6 --> D3
    N6 --> M3
    N7 --> M1
    N7 --> M2
    N7 --> M3
    N8 --> M1
    N8 --> M2
    N8 --> M3
    N9 --> D3
    N9 --> M4
    N10 --> N7
    N10 --> N8
    N10 --> N9

    style N1 fill:#e3f2fd
    style N2 fill:#e3f2fd
    style N3 fill:#e3f2fd
    style D1 fill:#fff3e0
    style D2 fill:#fff3e0
    style D3 fill:#fff3e0
    style M1 fill:#e8f5e9
    style M2 fill:#e8f5e9
    style M3 fill:#e8f5e9
    style M4 fill:#e8f5e9
    style M5 fill:#e8f5e9
    style N10 fill:#f3e5f5
```

---

## Final Thoughts

This project did not produce a profitable trading strategy. It produced something more valuable: a deep, hands-on understanding of why market prediction is hard. The models taught us about the noise ceiling in financial data, the primacy of feature engineering over model selection, and the dangerous gap between classification metrics and real-world trading performance.

```mermaid
graph TD
    Start["We started with a question<br/>Can we predict AAPL direction?"] --> Pipeline["Built full pipeline<br/>Data → Features → Models → Backtest"]
    Pipeline --> Failed["Models failed<br/>(40-45% accuracy)"]
    Failed --> Learned["We learned WHY"]
    Learned --> L1["Noise vs Lag tradeoff"]
    Learned --> L2["Features > Models"]
    Learned --> L3["Accuracy ≠ Profitability"]
    Learned --> L4["Technical indicators are not enough"]
    L1 --> Conclusion["The features are the model"]
    L2 --> Conclusion
    L3 --> Conclusion
    L4 --> Conclusion

    style Start fill:#e3f2fd,stroke:#333
    style Conclusion fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

If we learned one thing, it is this: **the features are the model.**
