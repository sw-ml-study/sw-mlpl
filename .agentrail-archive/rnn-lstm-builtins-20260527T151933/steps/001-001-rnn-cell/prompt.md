Step 001: rnn_cell builtin.

Implement rnn_cell(input, hidden, W_ih, W_hh, bias) in a new crates/mlpl-runtime/src/rnn_builtins.rs. One step of a simple recurrent cell: new_hidden = tanh(W_ih @ input + W_hh @ hidden + bias). Input [N, input_dim], hidden [N, hidden_dim]. Returns new hidden [N, hidden_dim].

TDD: unit test with known small matrices verifying output matches manual tanh(W_ih @ input + W_hh @ hidden + bias) computation.