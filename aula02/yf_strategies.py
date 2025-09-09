#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YFinance Strategies Backtester (Educational)
===========================================
AVISO IMPORTANTE:
Este script é **educacional**. Nenhuma estratégia aqui garante lucro.
Mercados envolvem riscos. Faça sua própria pesquisa e, se possível,
consulte um profissional habilitado. Considere custos, impostos e slippage.
"""

import argparse
import math
import sys
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# Dependências opcionais: yfinance e matplotlib são importadas somente quando usados
# para facilitar testes em ambientes sem internet.
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ----------------------------- Utilidades ----------------------------------

def annualize_factor(freq: str) -> int:
    if freq == "D":
        return 252
    if freq == "W":
        return 52
    if freq == "M":
        return 12
    raise ValueError("freq deve ser 'D', 'W' ou 'M'")


def to_month_end_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Usa resample para pegar os últimos dias úteis de cada mês presentes nos dados
    s = pd.Series(1, index=index)
    return s.resample("M").last().index


def compute_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_dd = drawdown.min()
    # Datas aproximadas do pico e do vale
    end = drawdown.idxmin()
    start = equity_curve.loc[:end].idxmax()
    return float(max_dd), start, end


def downside_std(returns: pd.Series, rf: float = 0.0, ann_factor: int = 252) -> float:
    # Desvio padrão apenas dos retornos abaixo do rf/ann_factor
    threshold = rf / ann_factor
    downside = returns[returns < threshold]
    if downside.empty:
        return 0.0
    return downside.std(ddof=0)


def performance_metrics(returns: pd.Series, rf: float = 0.0, freq: str = "D") -> Dict[str, float]:
    ann = annualize_factor(freq)
    equity = (1 + returns.fillna(0)).cumprod()
    total_return = equity.iloc[-1] - 1.0
    # CAGR
    years = len(returns) / ann
    cagr = equity.iloc[-1] ** (1 / max(years, 1e-9)) - 1.0
    sharpe = 0.0 if returns.std(ddof=0) == 0 else ((returns.mean() - rf / ann) / returns.std(ddof=0)) * math.sqrt(ann)
    dstd = downside_std(returns, rf=rf, ann_factor=ann)
    sortino = 0.0 if dstd == 0 else ((returns.mean() - rf / ann) / dstd) * math.sqrt(ann)
    max_dd, dd_start, dd_end = compute_max_drawdown(equity)
    calmar = 0.0 if max_dd == 0 else cagr / abs(max_dd)
    return {
        "Total Return": float(total_return),
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Max Drawdown": float(max_dd),
        "Calmar": float(calmar),
        "DD Start": dd_start.value if isinstance(dd_start, pd.Timestamp) else None,
        "DD End": dd_end.value if isinstance(dd_end, pd.Timestamp) else None,
    }


def apply_transaction_costs(portfolio_pos: pd.Series, cost_bps: float) -> pd.Series:
    """
    portfolio_pos: série de posições (ex.: 0 -> fora, 1 -> comprado, -1 -> vendido)
    Retorna série de custos diários (negativos) em % aplicados quando há mudança de posição.
    """
    turnover = (portfolio_pos.fillna(0) - portfolio_pos.shift(1).fillna(0)).abs()
    cost = turnover * (cost_bps / 10000.0)
    return -cost


# ------------------------------ Dados --------------------------------------

def fetch_prices(tickers: List[str], start: Optional[str], end: Optional[str], interval: str = "1d") -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance não está instalado neste ambiente. Instale com: pip install yfinance")
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    # Normaliza em MultiIndex (ticker, field)
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=1)
        # Se o nível superior for ticker, as colunas são tickers
        # Garante a ordem dos tickers pedidos, se existirem
        close = close[[c for c in tickers if c in close.columns]]
    else:
        # Caso de um único ticker: vira coluna única
        close = data["Close"].to_frame(tickers[0])
    close = close.dropna(how="all").sort_index()
    return close


# --------------------------- Estratégias -----------------------------------

def strategy_sma_crossover(close: pd.DataFrame, short: int = 20, long: int = 100,
                           cost_bps: float = 5.0) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Long-only por ticker quando SMA curta > SMA longa. Portfolio = média igualitária das posições longas.
    """
    short_sma = close.rolling(short, min_periods=short).mean()
    long_sma = close.rolling(long, min_periods=long).mean()
    signal = (short_sma > long_sma).astype(float)  # 1 quando cruzado pra cima, 0 caso contrário
    # Retornos simples
    rets = close.pct_change().fillna(0.0)
    # Posição de portfólio: média dos sinais (equal weight)
    port_pos = signal.mean(axis=1)
    # Custos
    cost = apply_transaction_costs(port_pos, cost_bps=cost_bps)
    # Retorno da carteira: posição de ontem * retorno médio simples hoje - custo de hoje
    port_rets = port_pos.shift(1).fillna(0.0) * rets.mean(axis=1) + cost
    return port_rets, signal


def strategy_rsi_mean_reversion(close: pd.DataFrame, rsi_len: int = 14, low: int = 30, high: int = 70,
                                cost_bps: float = 5.0) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compra quando RSI < low, zera quando RSI > high. Equal weight entre tickers ativos.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(rsi_len).mean()
    avg_loss = loss.rolling(rsi_len).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)

    signal = (rsi < low).astype(float)  # entra comprado quando sobrevendido
    # zera posição quando sobrecomprado
    flat = (rsi > high).astype(float)
    # Constrói posição persistente por ticker
    pos = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for col in close.columns:
        active = 0.0
        series = []
        for t in close.index:
            if signal.loc[t, col] == 1.0:
                active = 1.0
            if flat.loc[t, col] == 1.0:
                active = 0.0
            series.append(active)
        pos[col] = series
    rets = close.pct_change().fillna(0.0)
    port_pos = pos.mean(axis=1)
    cost = apply_transaction_costs(port_pos, cost_bps=cost_bps)
    port_rets = port_pos.shift(1).fillna(0.0) * rets.mean(axis=1) + cost
    return port_rets, pos


def strategy_momentum_topN(close: pd.DataFrame, lookback_days: int = 252, skip_recent: int = 21,
                           top_n: int = 5, cost_bps: float = 5.0) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Momentum 12-1 aproximado: rankeia por retorno dos últimos 'lookback_days' excluindo 'skip_recent'.
    Rebalanceia no último dia útil de cada mês, equal weight nos Top-N.
    """
    # Retornos cumulativos para janelas
    ret_total = close.pct_change(lookback_days).shift(skip_recent)
    # Índices de rebalance = fim de mês nos dados
    month_ends = to_month_end_index(close.index)
    month_ends = close.index.intersection(month_ends)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    last_selection = None
    for t in month_ends:
        scores = ret_total.loc[t].dropna().sort_values(ascending=False)
        selected = list(scores.head(top_n).index)
        last_selection = selected
        # Define pesos para o mês seguinte até próximo rebalance
        next_idx = month_ends.get_loc(t)
        if next_idx < len(month_ends) - 1:
            next_t = month_ends[next_idx + 1]
            idx_range = close.index[(close.index > t) & (close.index <= next_t)]
        else:
            idx_range = close.index[close.index > t]
        if len(selected) > 0:
            w = 1.0 / len(selected)
            weights.loc[idx_range, selected] = w

    # Caso nunca tenha caído num fim de mês (dataset curto), usa última seleção disponível
    if weights.sum().sum() == 0.0 and last_selection:
        w = 1.0 / len(last_selection)
        weights[last_selection] = w

    # Retornos e custos
    rets = close.pct_change().fillna(0.0)
    # Custo proporcional ao turnover total do portfólio
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = -turnover * (cost_bps / 10000.0)
    port_rets = (weights.shift(1).fillna(0.0) * rets).sum(axis=1) + cost
    return port_rets, weights


# ------------------------------ Runner -------------------------------------

@dataclass
class Config:
    tickers: List[str]
    start: Optional[str]
    end: Optional[str]
    strategy: str
    cost_bps: float
    sma_short: int
    sma_long: int
    rsi_len: int
    rsi_low: int
    rsi_high: int
    mom_lookback: int
    mom_skip: int
    mom_topn: int
    plot: bool
    rf: float
    out_trades: Optional[str]
    freq: str


def run(cfg: Config) -> None:
    print("⬇️  Baixando dados do yfinance...")
    close = fetch_prices(cfg.tickers, cfg.start, cfg.end)
    close = close.asfreq("B").ffill()  # garante frequência diária útil

    print(f"✅ Dados recebidos: {close.shape[0]} dias, {close.shape[1]} tickers.")

    if cfg.strategy == "sma":
        port_rets, detail = strategy_sma_crossover(close, cfg.sma_short, cfg.sma_long, cfg.cost_bps)
    elif cfg.strategy == "rsi":
        port_rets, detail = strategy_rsi_mean_reversion(close, cfg.rsi_len, cfg.rsi_low, cfg.rsi_high, cfg.cost_bps)
    elif cfg.strategy == "momentum":
        port_rets, detail = strategy_momentum_topN(close, cfg.mom_lookback, cfg.mom_skip, cfg.mom_topn, cfg.cost_bps)
    else:
        raise ValueError("Estratégia inválida. Use: sma | rsi | momentum")

    # Métricas
    equity = (1 + port_rets.fillna(0)).cumprod()
    metrics = performance_metrics(port_rets, rf=cfg.rf, freq="D")

    print("\n📈 Métricas de Performance (brutas, com custos informados):")
    for k, v in metrics.items():
        if k in ("DD Start", "DD End"):
            if v is not None:
                print(f"- {k}: {pd.to_datetime(v)}")
            else:
                print(f"- {k}: {v}")
        else:
            print(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")

    # Exporta trades/posições se solicitado
    if cfg.out_trades:
        try:
            if isinstance(detail, pd.DataFrame):
                detail.to_csv(cfg.out_trades, index=True)
                print(f"\n💾 Detalhe da estratégia exportado para: {cfg.out_trades}")
        except Exception as e:
            print(f"Falha ao salvar {cfg.out_trades}: {e}", file=sys.stderr)

    # Gráfico
    if cfg.plot:
        if plt is None:
            print("matplotlib não disponível para plotar.")
        else:
            plt.figure(figsize=(10, 5))
            equity.plot()
            plt.title(f"Equity Curve - {cfg.strategy.upper()}")
            plt.xlabel("Data")
            plt.ylabel("Patrimônio (base=1.0)")
            plt.grid(True)
            plt.tight_layout()
            try:
                plt.show()
            except Exception:
                # Em ambientes headless, salva
                out_png = "equity_curve.png"
                plt.savefig(out_png, dpi=150)
                print(f"Gráfico salvo em {out_png}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yf_strategies.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Backtester simples usando dados do yfinance (educacional).

            Exemplos:
              - SMA Crossover diário em AAPL e MSFT desde 2015, com gráfico:
                  python yf_strategies.py --tickers AAPL MSFT --start 2015-01-01 --strategy sma --plot

              - RSI mean reversion no IBOV (via ^BVSP) e PETR4.SA, exportando posições:
                  python yf_strategies.py --tickers ^BVSP PETR4.SA --strategy rsi --rsi-low 25 --rsi-high 65 --out-trades rsi_pos.csv

              - Momentum Top-5 em 10 ativos brasileiros:
                  python yf_strategies.py --tickers VALE3.SA PETR4.SA ITUB4.SA BBDC4.SA ABEV3.SA MGLU3.SA B3SA3.SA WEGE3.SA SUZB3.SA LREN3.SA                       --strategy momentum --mom-topn 5 --start 2014-01-01 --plot
            """
        ),
    )
    p.add_argument("--tickers", nargs="+", required=True, help="Lista de tickers (yfinance). Ex.: AAPL MSFT PETR4.SA")
    p.add_argument("--start", type=str, default=None, help="Data inicial (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="Data final (YYYY-MM-DD)")
    p.add_argument("--strategy", choices=["sma", "rsi", "momentum"], default="sma", help="Estratégia a rodar")
    p.add_argument("--cost-bps", type=float, default=5.0, help="Custo por ida (bps) aplicado ao turnover")
    # SMA params
    p.add_argument("--sma-short", type=int, default=20, help="Período da média curta (SMA)")
    p.add_argument("--sma-long", type=int, default=100, help="Período da média longa (SMA)")
    # RSI params
    p.add_argument("--rsi-len", type=int, default=14, help="Janela do RSI")
    p.add_argument("--rsi-low", type=int, default=30, help="Compra abaixo deste nível de RSI")
    p.add_argument("--rsi-high", type=int, default=70, help="Zera posição acima deste nível de RSI")
    # Momentum params
    p.add_argument("--mom-lookback", type=int, default=252, help="Dias de lookback para momentum (aprox. 12m)")
    p.add_argument("--mom-skip", type=int, default=21, help="Dias ignorados mais recentes (aprox. 1m)")
    p.add_argument("--mom-topn", type=int, default=5, help="Número de ativos no Top-N")
    # Outros
    p.add_argument("--plot", action="store_true", help="Plota curva de patrimônio")
    p.add_argument("--rf", type=float, default=0.0, help="Taxa livre de risco anual decimal (ex.: 0.05 = 5% a.a.)")
    p.add_argument("--out-trades", type=str, default=None, help="Caminho CSV para salvar posições/trades")
    return p


@dataclass
class ArgsProxy:
    tickers: List[str]
    start: Optional[str]
    end: Optional[str]
    strategy: str
    cost_bps: float
    sma_short: int
    sma_long: int
    rsi_len: int
    rsi_low: int
    rsi_high: int
    mom_lookback: int
    mom_skip: int
    mom_topn: int
    plot: bool
    rf: float
    out_trades: Optional[str]


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = Config(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        strategy=args.strategy,
        cost_bps=args.cost_bps,
        sma_short=args.sma_short,
        sma_long=args.sma_long,
        rsi_len=args.rsi_len,
        rsi_low=args.rsi_low,
        rsi_high=args.rsi_high,
        mom_lookback=args.mom_lookback,
        mom_skip=args.mom_skip,
        mom_topn=args.mom_topn,
        plot=bool(args.plot),
        rf=args.rf,
        out_trades=args.out_trades,
        freq="D",
    )
    run(cfg)


if __name__ == "__main__":
    main()
