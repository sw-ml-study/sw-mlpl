Step 002: lstm_cell builtin.

Implement lstm_cell(input, hidden, cell, W_i, W_f, W_c, W_o, b_i, b_f, b_c, b_o) in rnn_builtins.rs. Or simpler: lstm_cell(input, hidden, cell, weights, bias) where weights [4*hidden_dim, input_dim+hidden_dim] and bias [4*hidden_dim] are concatenated gates (input, forget, cell, output).

Returns (new_hidden, new_cell). Uses sigmoid for gates, tanh for cell candidate.

TDD: unit test with known values.