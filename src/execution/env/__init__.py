"""Real-data order-execution environment (Phase 2).

Replaces the QRM simulator as the order-book source: the RL agent executes a buy
meta-order against *replayed* real Hyperliquid L2 books (the frozen Stage-6
datasets) instead of synthetic books. Built as the user's own package; the
vendored QRM scaffold is left untouched and cited.
"""
