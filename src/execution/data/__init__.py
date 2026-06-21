"""Data pipeline: raw source -> canonical book -> minute features -> train/test.

The pipeline is data-source-agnostic. All source-specific logic (how to list and
fetch raw order-book data) lives behind the adapters in ``sources/``; every stage
downstream of parsing operates on the canonical book schema.
"""
