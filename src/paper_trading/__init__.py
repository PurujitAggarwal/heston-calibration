"""Autonomous paper-trading system for the Heston short-vol strategy.

Live counterpart to the historical ``src.heston`` research pipeline: pulls SPX
option chains from Interactive Brokers, recalibrates Heston daily, generates
short-vol signals, executes paper trades and reports by email. All modules
here read and reuse the existing ``src.heston`` logic without modifying it.
"""
