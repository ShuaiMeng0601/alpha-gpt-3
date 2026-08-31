"""LLM-powered alpha explainer: takes an expression + metrics, returns plain-language explanation."""

from openai import OpenAI

from alpha_gpt.evaluate.backtester import BacktestResult
from alpha_gpt.config import Config
from alpha_gpt.llm import _completion_params


def explain_alpha(
    expression: str,
    backtest: BacktestResult,
    ic_mean: float,
    icir: float,
    client: OpenAI | None = None,
    model: str = Config.llm_model,
) -> str:
    """Generate a plain-language explanation of an evolved alpha.

    Args:
        expression: The symbolic alpha expression string.
        backtest: BacktestResult from backtesting the alpha.
        ic_mean: Mean information coefficient.
        icir: Information ratio (IC / std(IC)).
        client: OpenAI client (pointed at OpenRouter).
        model: Model identifier on OpenRouter.

    Returns:
        Plain-language explanation string.
    """
    if client is None:
        return _fallback_explanation(expression, backtest, ic_mean, icir)

    prompt = f"""You are a quantitative finance researcher. Explain the following alpha expression
in plain language. Describe what trading signal it captures, why it might work,
and summarize its performance.

Alpha expression: {expression}

Performance metrics:
- Mean IC: {ic_mean:.4f}
- ICIR: {icir:.4f}
- Annualized Sharpe: {backtest.sharpe:.2f}
- Annualized Return: {backtest.annual_return:.2%}
- Max Drawdown: {backtest.max_drawdown:.2%}

Quantile returns (Q1=lowest alpha, Q5=highest alpha):
{backtest.quantile_returns.to_string()}

Keep the explanation concise (3-5 sentences). Focus on the intuition behind the signal."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **_completion_params(model, temperature=0.3, max_tokens=300),
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_explanation(expression, backtest, ic_mean, icir) + f"\n(LLM error: {e})"


def _fallback_explanation(
    expression: str, backtest: BacktestResult, ic_mean: float, icir: float
) -> str:
    """Simple template-based explanation when LLM is unavailable."""
    quality = "strong" if abs(ic_mean) > 0.03 else "moderate" if abs(ic_mean) > 0.01 else "weak"
    direction = "positive" if ic_mean > 0 else "negative"

    return (
        f"Alpha: {expression}\n"
        f"This alpha has a {quality} {direction} predictive signal "
        f"(mean IC={ic_mean:.4f}, ICIR={icir:.4f}). "
        f"The long-short portfolio achieves a Sharpe of {backtest.sharpe:.2f} "
        f"with {backtest.annual_return:.2%} annualized return "
        f"and {backtest.max_drawdown:.2%} max drawdown."
    )
