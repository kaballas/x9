run python3 scripts/inspect_race.py --verbose 10480071

For this resulted race, the arithmetic is mostly adding up, but the probability and edge logic is still wrong.

The race itself looks usable. Declared field is 18, active is 14, excluded is 4, and the model ranks 14 runners. That is internally consistent. The warning is not a blocker here, unlike the earlier race where `Active: 0` contradicted the ranking output.

The biggest issue is still `MktP%`.

Your `MktP%` is de-vigged market probability. It is not the breakeven probability at the actual price.

For this race, the raw market book is:

```text
Sum of 1 / price = 124.6%
```

So every displayed `MktP%` has been divided down by the overround. That is why `Global Turn` at $10.00 shows:

```text
ModelP = 9.5%
MktP   = 8.0%
Edge   = +1.5%
```

But the actual breakeven at $10.00 is:

```text
1 / 10.00 = 10.0%
```

So the true betting edge is:

```text
9.5% - 10.0% = -0.5%
EV = 0.095 * 10.00 - 1 = -5.0%
```

Your report says positive edge. The actual price says negative EV.

That same issue affects the watchlist. Corrected examples:

| Runner             |  Price | ModelP | Reported MktP | Raw breakeven | Reported edge | True raw edge |       EV |
| ------------------ | -----: | -----: | ------------: | ------------: | ------------: | ------------: | -------: |
| Global Turn        |  10.00 |   9.5% |          8.0% |         10.0% |         +1.5% |         -0.5% |    -5.0% |
| Empire Grace       |  16.00 |   8.7% |          5.0% |         6.25% |         +3.7% |        +2.45% |   +39.2% |
| She Rex            |  51.00 |   7.2% |          1.6% |         1.96% |         +5.6% |        +5.24% |  +267.2% |
| Immediate Response | 101.00 |   6.8% |          0.8% |         0.99% |         +6.0% |        +5.81% |  +586.8% |
| Avanzo             | 201.00 |   5.7% |          0.4% |         0.50% |         +5.3% |        +5.20% | +1045.7% |

This exposes the second issue: your probability distribution is too flat. A $201 runner is getting 5.7% model probability. That implies fair odds of about $17.54. That is not credible unless your model has very strong evidence, which it does not.

The compression is visible across the whole field:

```text
Top model probability:    10.3%
Bottom model probability:  4.7%
```

In a 14-runner race, that is too narrow. The model barely separates the field. The actual winner, `Invertational`, was market second/fav range at $4.80 but your model gave it only 6.4% and ranked it 10th.

That points to a score-to-probability calibration problem, not just a scoring problem.

Likely causes:

```text
1. Softmax temperature is too high, flattening probabilities.
2. Score spread is too compressed.
3. Market signal is too weak at weight 0.01.
4. Longshot probabilities are not being capped or calibrated.
5. The model is ranking by raw weighted score, but the score is not calibrated to actual win probability.
```

Your component scoring also shows a market contradiction. `Invertational` had:

```text
Price = 4.80
Steam score = 0.946
Mkt contribution = 0.0081
Model rank = 10
ModelP = 6.4%
```

The market and steam both liked the eventual winner, but your model barely cared because market only contributes 1%. If your goal is betting, not pure form modelling, that is too low. Market should not dominate, but 1% is close to decorative.

The winner was penalised heavily by:

```text
Form   = 0.069
Margin = 0.059
Conn   = 0.004
Fresh  = 0.034
```

But it had strong enough market support that the model should at least have treated it as a danger. It should not have been below multiple $21, $31, $51, $101, and $201 runners.

The third issue is the watchlist is misleading. These are not useful “edges”:

```text
Immediate Response at $101, ModelP 6.8%
She Rex at $51, ModelP 7.2%
Avanzo at $201, ModelP 5.7%
Whoops A Daisy at $31, ModelP 6.9%
```

Those are symptoms of overconfident longshot probabilities. The filters stop them becoming bets, but the watchlist still makes the model look better than it is.

Fix the report like this:

```python
raw_market_prob = 1 / live_price
book_overround = sum(1 / p for p in race_prices)
fair_market_prob = raw_market_prob / book_overround

raw_edge = model_prob - raw_market_prob
fair_edge = model_prob - fair_market_prob
ev = model_prob * live_price - 1
```

Then rename the columns:

```text
MktP%        -> FairMktP%
Edge%        -> FairEdge%
RawMktP%     -> 1 / price
RawEdge%     -> ModelP - RawMktP
EV%          -> ModelP * price - 1
```

For bet qualification, use this:

```python
candidate = (
    race_integrity_ok
    and ev >= 0.05
    and raw_edge >= 0.02
    and model_prob >= 0.10
    and model_rank <= 2
    and 2.0 <= live_price <= 15.0
)
```

For this resulted race, your “no bets qualify” result is correct under the current thresholds. But the reason is partly accidental. The edge/watchlist math is still inflated, and the model probabilities are too flat to trust for EV.

The next thing to fix is not the weighted score sum. It is probability calibration. Run a backtest bucketed by `ModelP` and compare actual win rate:

```text
ModelP 5–7%    actual win rate?
ModelP 7–10%   actual win rate?
ModelP 10–15%  actual win rate?
ModelP 15–25%  actual win rate?
```

If runners priced $51 to $201 with model probabilities around 6% are not winning anywhere near 6% of the time, your probability layer is overstating longshots and understating favourites.

