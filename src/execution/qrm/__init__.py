"""QRM calibration (Step 3) - fit the Queue-Reactive Model on the reconstructed BTC book.

Produces the inputs the vendored `qrm_core` engine consumes: an intensity table
(rates per depth x queue-size x side x event-type), the invariant queue distributions
(derived analytically from the intensities), and the average event size per level.
Calibrated on the real L4 event stream; regime-conditional (calm vs volatile). New code;
the vendored `qrm_optimal_execution` engine is reused as a library, not edited.
"""
