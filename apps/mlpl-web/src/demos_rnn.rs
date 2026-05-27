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
