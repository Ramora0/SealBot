
## NNUE v3 design: LineNL (user directive — ONE simple representation)

Line = the unit, not the cell. Per maximal line (3 dirs x offset, ~420):
line_acc[K] = sum of its 6-window EW embeddings (already incremental);
ONE clamp NL per line; value head = MLP(sum of clamped line acts + g);
policy(cell) = MLP([its 3 line acts; global; g0; g1]). Replaces the cell
trunk entirely. Rationale: threats are line phenomena (building attacks
within a line = representable by line NL; cross-line coordination =
linear readout of summed line activations). Engine: maintain
line_acc[420][K] + acc = sum clamp(line_acc), diff per window change
(~18 windows x 3 lines per stone, ~trunk cost).
Status: v1.5 (soft labels) benching; DAgger 18230 positions prepped
(dagger_prep/); champion = trunk3+mode74+VCF15/k11/40k = 45/150 (-146).
Next: trunk_train3.py (LineNL), label DAgger, train on full mix, port
(TRK-style, new accumulator), gate on strix 150.
