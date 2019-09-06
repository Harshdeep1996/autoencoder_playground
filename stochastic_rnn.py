import sonnet as snet
import tensorflow as tf
from collections import namedtuple

StochasticRNNState = namedtuple('StochasticRNNState', 'rnn_state latent_encoded')
TrainableStochasticRNNState = namedtuple('TrainableStochasticRNNState', StochasticRNNState._fields + ('rnn_out', ))

class StochasticRNN(object):

    def __init__(
        self, rnn_cell, data_encoder, transition,
        emission, latent_encoder, random_seed=None):
        self.rnn_cell = rnn_cell
        self.state_size = self.rnn_cell.state_size
        self.data_encoder = data_encoder
        self.transition_state = transition ## z_t <- z_t-1, h_t ## Normal
        self.emission_state = emission ## X_t <- z_t, h_t ## Logprob
        # Encoding z into something favorable
        self.latent_encoder = latent_encoder
        # Dimensions of z
        self.encoded_z_side = latent_encoder.output_size
        self.random_seed = random_seed

    def zero_state(self, batch_size, dtype):
        ## Make the initial state of the input and latent layer
        return StochasticRNNState(
            rnn_state=self.rnn_cell.zero_state(batch_size),
            latent_encoded=tf.zeros([batch_size, self.latent_encoder.output_size], dtype=dtype)
        )

    def run_rnn(self, previous_state, inputs):
        # Encode the data x_t into something favorable which could be fed to an RNN
        # Function to get the new hidden state
        rnn_inputs = self.data_encoder(tf.to_float(inputs))
        rnn_out, rnn_state = self.rnn_cell(rnn_inputs, previous_state)
        return rnn_out, rnn_state

    def transition(self, previous_latent, current_hidden_state):
        return self.transition_state(current_hidden_state, previous_latent)

    def emission(self, current_latent_state, current_hidden_state):
        latent_inputs = self.latent_encoder(tf.to_float(current_latent_state))
        return (
            self.emission_state(current_hidden_state, latent_inputs),
            latent_inputs
        )

    def sample_step(self, previous_state, inputs, unused_t):
        """
        args: previous_state: this is the previous rnn state and previous encoded latent.
        args: input: this is X_t-1 with [batch_size, data_size]
        args: the current time step (this can be used for filtering)
        """
        rnn_out, rnn_state = self.run_rnn(previous_state.rnn_state, inputs)
        z_t = self.transition(prev_state.latent_encoded, rnn_out)
        z_t = z_t.sample(seed=self.random_seed)
        x_t = self.emsission(z_t, rnn_out)
        new_state = StochasticRNNState(rnn_state=rnn_state, latent_encoded=latent_encoded)
        return new_state, tf.to_float(x_t)

class TrainableStochasticRNN(StochasticRNN):

    def __init__(
        self, rnn_cell, data_encoder, latent_encoder,
        transition, emission, proposal_type, proposal=None,
        rev_rnn_cell=None, tilt=None, random_seed=None):

    super(TrainableStochasticRNN, self).__init__(
        rnn_cell, data_encoder, transition, emission,
        latent_encoder, random_seed=random_seed
    )
    self.rev_rnn_cell = rev_rnn_cell
    self._tilt = tilt
    ## Callable for proposal to which inputs can be fed
    ## hidden state and encoded target of current timestamp
    self._proposal = proposal
    self.proposal_type = proposal_type

    def zero_state(self, batch_size, dtype):
        ## Assigning all 3 - input, hidden and latent layer to be zero
        super_state = super(TrainableStochasticRNN, self).zero_state(batch_size, dtype)
        return TrainableStochasticRNNState(
            ## Assigning hidden state to be zeros at the start
            rnn_out=tf.zeros([batch_size, self.rnn_cell.output_size], dtype=dtype).
            **super_state._asdict() ## Get the params from the StochasticRNNState
        )

    def ta_for_tensor(self, x, dynamic_size=False, clear_after_read=False):
        # Unstack the inputs and outputs based on sequence lengths which need to be fed into the RNN
        return tf.TensorArray(
            x.dtype, tf.shape(x)[0],
            dynamic_size=dynamic_size,
            clear_after_read=clear_after_read).unstack(x)

    def encode_all(inputs, encoder):
      """Encodes a timeseries of inputs with a time independent encoder.
      Args:
        inputs: A [time, batch, feature_dimensions] tensor.
        encoder: A network that takes a [batch, features_dimensions] input and
          encodes the input.
      Returns:
        A [time, batch, encoded_feature_dimensions] output tensor.
      """
      input_shape = tf.shape(inputs)
      num_timesteps, batch_size = input_shape[0], input_shape[1]
      reshaped_inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])
      inputs_encoded = encoder(reshaped_inputs)
      inputs_encoded = tf.reshape(inputs_encoded,
                                  [num_timesteps, batch_size, encoder.output_size])
      return inputs_encoded

    def set_observations(self, observations, seq_lengths):
        """
        args: observations: 2 tensors of shape (max_seq_len, batch_size, data_size)
            should be inputs and targets.
        args: seq_lengths: tensor of length batch size with all the lengths
        """
        inputs, targets = observations
        self.seq_lengths = seq_lengths
        self.max_len_seq = tf.reduce_max(seq_lengths)
        self.targets_ta = self.ta_for_tensor(targets)
        # Encode the targets for the LSTM
        targets_encoded = self.encode_all(targets, self.data_encoder)
        self.targets_encoded_ta = self.ta_for_tensor(targets_encoded)
        # Encode the inputs for the LSTM
        inputs_encoded = self.encode_all(inputs, self.data_encoder)

        ## Feed it into the RNN
        ## time major is letting if the inputs are of the form of [max__l]
        rnn_out, _ = tf.nn.dynammic_rnn(
            self.rnn_cell, inputs_encoded, time_major=True, dtype=tf.float32, scope='forward_rnn')
        self.rnn_ta = self.ta_for_tensor(rnn_out)

        # This is for SMOOTHING - learning the inference by reversing the inputs
        if self.rev_rnn_cell:
            targets_and_rnn_out = tf.concat([rnn_out, targets_encoded], 2) # Along 2nd axis/data points
            reversed_input = tf.reverse_sequence(targets_and_rnn_out, seq_lengths, seq_axis=0, batch_axis=1)
            reverse_rnn_out, _ = tf.nn.dynammic_rnn(
                self.rev_rnn_cell, reversed_input, time_major=True, dtype=tf.float32, scope='reverse_rnn'
            )
            reverse_rnn_out = tf.reverse_sequence(reverse_rnn_out, seq_lengths, seq_axis=0, batch_axis=1)
            self.reverse_rnn_ta = self.ta_for_tensor(reverse_rnn_out)

    def _filtering_proposal(self, rnn_out, prev_latent_encoded, prior, t):
        return self._proposal(
            rnn_out, prev_latent_encoded,
            self.targets_encoded_ta.read(t), prior_mu=prior.mean())

    def _smoothing_proposal(self, rnn_out, prev_latent_encoded, prior, t):
        return self._proposal(
            rnn_out, prev_latent_encoded,
            smoothing_tensors=[self.reverse_rnn_ta.read(t)]
            prior_mu=prior.mean())

    def proposal(self):
        pass

