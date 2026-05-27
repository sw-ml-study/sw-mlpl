use crate::demos::Demo;

pub const RNN_SEQUENCE: Demo = Demo {
    category: "RNN",
    name: "RNN (sequence)",
    intro: "A simple recurrent neural network unrolled over a 5-step sequence. \
            At each step, rnn_cell reads one input and updates the hidden state. \
            The hidden state is the network's 'memory' of what it has seen so far. \
            Watch the hidden vector change at each time step.",
    takeaway: "The hidden state after step 5 encodes the entire sequence history. \
               Each rnn_cell call is one tick of the clock -- the same weights are \
               reused at every step (weight sharing). This is the core RNN idea: \
               fixed-size memory updated by each new input.",
    lines: &[
        "W_ih = randn(1, [4, 1])",
        "W_hh = randn(2, [4, 4])",
        "bias = reshape(zeros([4]), [4, 1])",
        "h = reshape(zeros([4]), [4, 1])",
        "h = rnn_cell(reshape([0.2], [1, 1]), h, W_ih, W_hh, bias)",
        "h = rnn_cell(reshape([0.8], [1, 1]), h, W_ih, W_hh, bias)",
        "h = rnn_cell(reshape([0.1], [1, 1]), h, W_ih, W_hh, bias)",
        "h = rnn_cell(reshape([0.9], [1, 1]), h, W_ih, W_hh, bias)",
        "h = rnn_cell(reshape([0.5], [1, 1]), h, W_ih, W_hh, bias)",
        "h",
    ],
};

pub const LSTM_SEQUENCE: Demo = Demo {
    category: "RNN",
    name: "LSTM (sequence memory)",
    intro: "An LSTM cell unrolled over a 10-step sequence -- twice as long as \
            the simple RNN demo. LSTM adds a separate 'cell state' that carries \
            information across many time steps without the vanishing gradient \
            problem that plagues vanilla RNNs. Watch hidden (h) and cell (c) \
            diverge as the sequence progresses.",
    takeaway: "After 10 steps the LSTM still has a functioning cell state (c) that \
               remembers early inputs. A vanilla RNN's hidden state would have \
               washed out by now -- each tanh squashing loses a little signal. \
               The LSTM's forget/input/output gates control what to keep, what to \
               add, and what to expose, giving it stable long-range memory.",
    lines: &[
        // W: [4*hd, id+hd] = [8, 3], bias: [4*hd, 1] = [8, 1]
        "W = randn(42, [8, 3])",
        "bias = reshape(zeros([8]), [8, 1])",
        // hidden and cell state, each [hd, 1] = [2, 1]
        "h = reshape(zeros([2]), [2, 1])",
        "c = reshape(zeros([2]), [2, 1])",
        // Step 1 -- result is [2*hd, 1]=[4,1]; reshape to [2,2] to split
        "hc = lstm_cell(reshape([0.9], [1, 1]), h, c, W, bias)",
        "h = reshape(take(reshape(hc, [2, 2]), 0, 0), [2, 1])",
        "c = reshape(take(reshape(hc, [2, 2]), 0, 1), [2, 1])",
        // Step 2
        "hc = lstm_cell(reshape([0.1], [1, 1]), h, c, W, bias)",
        "h = reshape(take(reshape(hc, [2, 2]), 0, 0), [2, 1])",
        "c = reshape(take(reshape(hc, [2, 2]), 0, 1), [2, 1])",
        // Step 3
        "hc = lstm_cell(reshape([0.8], [1, 1]), h, c, W, bias)",
        "h = reshape(take(reshape(hc, [2, 2]), 0, 0), [2, 1])",
        "c = reshape(take(reshape(hc, [2, 2]), 0, 1), [2, 1])",
        // Step 4
        "hc = lstm_cell(reshape([0.2], [1, 1]), h, c, W, bias)",
        "h = reshape(take(reshape(hc, [2, 2]), 0, 0), [2, 1])",
        "c = reshape(take(reshape(hc, [2, 2]), 0, 1), [2, 1])",
        // Step 5
        "hc = lstm_cell(reshape([0.7], [1, 1]), h, c, W, bias)",
        "h = reshape(take(reshape(hc, [2, 2]), 0, 0), [2, 1])",
        "c = reshape(take(reshape(hc, [2, 2]), 0, 1), [2, 1])",
        // Step 6
        "hc = lstm_cell(reshape([0.3], [1, 1]), h, c, W, bias)",
        "h = reshape(take(reshape(hc, [2, 2]), 0, 0), [2, 1])",
        "c = reshape(take(reshape(hc, [2, 2]), 0, 1), [2, 1])",
        // Step 7
        "hc = lstm_cell(reshape([0.6], [1, 1]), h, c, W, bias)",
        "h = reshape(take(reshape(hc, [2, 2]), 0, 0), [2, 1])",
        "c = reshape(take(reshape(hc, [2, 2]), 0, 1), [2, 1])",
        // Step 8
        "hc = lstm_cell(reshape([0.4], [1, 1]), h, c, W, bias)",
        "h = reshape(take(reshape(hc, [2, 2]), 0, 0), [2, 1])",
        "c = reshape(take(reshape(hc, [2, 2]), 0, 1), [2, 1])",
        // Step 9
        "hc = lstm_cell(reshape([0.5], [1, 1]), h, c, W, bias)",
        "h = reshape(take(reshape(hc, [2, 2]), 0, 0), [2, 1])",
        "c = reshape(take(reshape(hc, [2, 2]), 0, 1), [2, 1])",
        // Step 10
        "hc = lstm_cell(reshape([0.1], [1, 1]), h, c, W, bias)",
        "h = reshape(take(reshape(hc, [2, 2]), 0, 0), [2, 1])",
        "c = reshape(take(reshape(hc, [2, 2]), 0, 1), [2, 1])",
        // Final state
        "h",
        "c",
    ],
};
