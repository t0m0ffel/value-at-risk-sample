from typing import List

import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from enum import Enum, unique, auto
from scipy.stats import norm

rng = np.random.default_rng(2021)


@unique
class Ticker(Enum):
    APPLE = "AAPL"
    SWISS_RE = "SREN.SW"
    TESLA = "TSLA"
    GOLD = "GC=F"
    BITCOIN = "BTC-USD"

    def format_name(self):
        return " ".join([w.capitalize() for w in self.name.split("_")])


class VaRMethod(Enum):
    HISTORICAL_DATA = auto()
    NORMAL_DISTRIBUTION = auto()
    NORMAL_DISTRIBUTION_SAMPLING = auto()


class RiskMetrics:
    def __init__(self, method, var, es, alpha, symbol=None):
        self.value_at_risk = var
        self.expected_shortfall = np.round(es, 4)
        self.method = method
        self.symbol = symbol
        self.alpha = alpha

    def __rep__(self):
        return self.__str__

    def __str__(self):
        return self.method + "\n" + "-" * 30 + "\n" + str(dict(
            value_at_risk="{}%".format(np.round(self.value_at_risk * 100, 2)),
            espected_shortfall="{}%".format(np.round(self.expected_shortfall * 100, 2))
        ))


def get_data(company_ticker: Ticker, start: datetime = None, end: datetime = None):
    return yf.download(company_ticker.value, start=start, end=end)


def risk_metrics(
        df: pd.DataFrame,
        alpha: float,
        method: VaRMethod,
        price_field: str = "Close",
        symbol: str = None
) -> RiskMetrics:
    log_returns: pd.Series = (np.log(df[price_field]) - np.log(df[price_field].shift(1))).dropna()

    if method == VaRMethod.NORMAL_DISTRIBUTION_SAMPLING:
        mu = log_returns.mean()
        sig = log_returns.std()
        samples = rng.normal(mu, sig, 10_000)
        var = np.quantile(samples, alpha)
        es = samples[samples < var].mean()

    elif method == VaRMethod.NORMAL_DISTRIBUTION:
        mu = log_returns.mean()
        sig = log_returns.std()
        var = norm.ppf(alpha, loc=mu, scale=sig)
        es = mu - sig * norm.pdf(norm.ppf(alpha, 0, 1), 0, 1) / alpha

    elif method == VaRMethod.HISTORICAL_DATA:
        var = np.quantile(log_returns, alpha)
        es = log_returns[log_returns < var].mean()
    else:
        raise ValueError("Method has to be one of {}".format(list(VaRMethod)))

    return RiskMetrics(method, abs(var), abs(es), alpha, symbol)


def all_risk_metrics(data: pd.DataFrame, alpha: float, symbol: str = None):
    return [risk_metrics(data, method=method, alpha=alpha, symbol=symbol) for method in VaRMethod]


def metrics_for_tickers(tickers: List[Ticker], start: datetime, end: datetime, alpha: float):
    return sum([
        all_risk_metrics(
            get_data(ticker, start=start, end=end),
            alpha,
            ticker.format_name()
        ) for ticker in tickers
    ], [])


def plot_method(df: pd.DataFrame, method: VaRMethod, title: str, sort_by: str = "value_at_risk",
                exclude: str = "alpha"):
    ax = df[df.method == method].sort_values(sort_by).drop(exclude, axis=1).plot.bar(rot=0)
    ax.set_title(title, fontsize=15)
    ax.get_figure().savefig(f'{"_".join([w.lower() for w in title.split(" ")])}.pdf')


def metrics_to_df(metrics):
    return pd.DataFrame(map(lambda m: m.__dict__, metrics))


if __name__ == '__main__':
    metrics = metrics_for_tickers(list(Ticker), start=datetime(2020, 2, 21), end=datetime(2020, 3, 30), alpha=0.05)

    risk_df_covid_impact = metrics_to_df(metrics)
    risk_df_covid_impact.to_csv("risk_metrics_covid_impact")

    metrics = metrics_for_tickers(list(Ticker), start=datetime(2015, 1, 1), end=datetime(2021, 8, 1), alpha=0.05)

    risk_df_long_term = metrics_to_df(metrics)

    risk_df_long_term.to_csv("risk_metrics_long-term.csv")

    risk_df_covid_impact = risk_df_covid_impact.set_index("symbol")
    risk_df_long_term = risk_df_long_term.set_index("symbol")

    plot_method(risk_df_long_term, VaRMethod.HISTORICAL_DATA, "Risk Metrics from 2015")
    plot_method(risk_df_covid_impact, VaRMethod.HISTORICAL_DATA, "Risk Metrics during Covid-19 Impact")

    print("Done")
