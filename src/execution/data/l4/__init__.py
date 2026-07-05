"""Stage L4 - order-book reconstruction from Oxford "Open Book" per-order data.

New track (does NOT touch the L2 pipeline). Reconstructs the visible BTC limit order
book from the raw book-diff stream so a Queue-Reactive Model can be calibrated on it.
Modules:

* :mod:`book_diffs_reader` - stream normalised events from one hourly file.
* :mod:`book_engine`       - apply events in sequence to maintain the book.
"""
