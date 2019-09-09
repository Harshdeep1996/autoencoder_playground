import sonnet as snt
import tensorflow as tf
from collections import namedtuple

class ConditionalNormalDistribution(object):
  """A Normal distribution conditioned on Tensor inputs via a fc network."""

  def __init__(self, size, hidden_layer_sizes, sigma_min=0.0,
               raw_sigma_bias=0.25, hidden_activation_fn=tf.nn.relu,
               initializers=None, name="conditional_normal_distribution"):
    """Creates a conditional Normal distribution.
    Args:
      size: The dimension of the random variable.
      hidden_layer_sizes: The sizes of the hidden layers of the fully connected
        network used to condition the distribution on the inputs.
      sigma_min: The minimum standard deviation allowed, a scalar.
      raw_sigma_bias: A scalar that is added to the raw standard deviation
        output from the fully connected network. Set to 0.25 by default to
        prevent standard deviations close to 0.
      hidden_activation_fn: The activation function to use on the hidden layers
        of the fully connected network.
      initializers: The variable intitializers to use for the fully connected
        network. The network is implemented using snt.nets.MLP so it must
        be a dictionary mapping the keys 'w' and 'b' to the initializers for
        the weights and biases. Defaults to xavier for the weights and zeros
        for the biases when initializers is None.
      name: The name of this distribution, used for sonnet scoping.
    """
    self.sigma_min = sigma_min
    self.raw_sigma_bias = raw_sigma_bias
    self.name = name
    self.size = size
    if initializers is None:
      initializers = DEFAULT_INITIALIZERS
    self.fcnet = snet.nets.MLP(
        output_sizes=hidden_layer_sizes + [2 * size],
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_fcnet")

  def condition(self, tensor_list, **unused_kwargs):
    """Computes the parameters of a normal distribution based on the inputs."""
    inputs = tf.concat(tensor_list, axis=1)
    outs = self.fcnet(inputs)
    mu, sigma = tf.split(outs, 2, axis=1)
    sigma = tf.maximum(tf.nn.softplus(sigma + self.raw_sigma_bias),
                       self.sigma_min)
    return mu, sigma

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution conditioned on the inputs."""
    mu, sigma = self.condition(args, **kwargs)
    return tf.contrib.distributions.Normal(loc=mu, scale=sigma)

class NormalApproximatePosterior(ConditionalNormalDistribution):
  """A Normally-distributed approx. posterior with res_q parameterization."""

  def __init__(self, size, hidden_layer_sizes, sigma_min=0.0,
               raw_sigma_bias=0.25, hidden_activation_fn=tf.nn.relu,
               initializers=None, smoothing=False,
               name="conditional_normal_distribution"):
    super(NormalApproximatePosterior, self).__init__(
        size, hidden_layer_sizes, sigma_min=sigma_min,
        raw_sigma_bias=raw_sigma_bias,
        hidden_activation_fn=hidden_activation_fn, initializers=initializers,
        name=name)
    self.smoothing = smoothing

  def condition(self, tensor_list, prior_mu, smoothing_tensors=None):
    """Generates the mean and variance of the normal distribution.
    Args:
      tensor_list: The list of Tensors to condition on. Will be concatenated and
        fed through a fully connected network.
      prior_mu: The mean of the prior distribution associated with this
        approximate posterior. Will be added to the mean produced by
        this approximate posterior, in res_q fashion.
      smoothing_tensors: A list of Tensors. If smoothing is True, these Tensors
        will be concatenated with the tensors in tensor_list.
    Returns:
      mu: The mean of the approximate posterior.
      sigma: The standard deviation of the approximate posterior.
    """
    if self.smoothing:
      tensor_list.extend(smoothing_tensors)
    mu, sigma = super(NormalApproximatePosterior, self).condition(tensor_list)
    return mu + prior_mu, sigma


##### STOCHASTIC RNN ######

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
        ## Make the initial state of the hidden and latent layer - (h and z)
        return StochasticRNNState(
            rnn_state=self.rnn_cell.zero_state(batch_size, dtype=dtype),
            latent_encoded=tf.zeros([batch_size, self.latent_encoder.output_size], dtype=dtype)
        )

    def run_rnn(self, previous_state, inputs):
        # Encode the data x_t into something favorable which could be fed to an RNN
        # Function to get the new output and hidden state
        rnn_inputs = self.data_encoder(tf.to_float(inputs))
        rnn_out, rnn_state = self.rnn_cell(rnn_inputs, previous_state)
        return rnn_out, rnn_state # (h,c)

    def transition(self, previous_latent, current_hidden_state):
        return self.transition_state(current_hidden_state, previous_latent)

    def emission(self, current_latent_state, current_hidden_state):
        # Encoding the input of the variational autoencoder to the RNN
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
        x_t, latent_encoded = self.emsission(z_t, rnn_out)
        x_t = x_t.sample(seed=self.random_seed)
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
            rnn_out=tf.zeros([batch_size, self.rnn_cell.output_size], dtype=dtype),
            **super_state._asdict() ## Get the params from the StochasticRNNState
        )

    def ta_for_tensor(self, x, dynamic_size=False, clear_after_read=False):
        # Unstack the inputs and outputs based on sequence lengths which need to be fed into the RNN
        return tf.TensorArray(
            x.dtype, tf.shape(x)[0],
            dynamic_size=dynamic_size,
            clear_after_read=clear_after_read).unstack(x)

    def encode_all(self, inputs, encoder):
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
      inputs_encoded = tf.reshape(inputs_encoded, [num_timesteps, batch_size, encoder.output_size])
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
            smoothing_tensors=[self.reverse_rnn_ta.read(t)],
            prior_mu=prior.mean())

    def proposal(self, rnn_out, prev_latent_encoded, prior, t):
        ## Depending upon the proposal call the particular neural network to do
        ## an inference for the inputs
        if self.proposal_type == 'filtering':
            return self._filtering_proposal(rnn_out, prev_latent_encoded, prior, t)
        elif self.proposal_type == 'smoothing':
            return self._smoothing_proposal(rnn_out, prev_latent_encoded, prior, t)
        else:
            return self.transition(prev_latent_encoded, rnn_out)

    def tilt(self, rnn_out, latent_encoded, targets):
        # Calculating the log probability of the output
        # given in the hidden state and latent stat
        r_func = self._tilt(rnn_out, latent_encoded)
        return tf.reduce_sum(r_func.log_prob(targets), axis=-1)

    def propose_and_weight(self, state, t):
        """
        args: state: the previous state of the model - and an TrainableSRNNState 
        args: t: the current timestep on which the model is
        """
        ## For running the model and computes importance weights for one timestep
        
        ## One gets the targets and encoded input for one timestep
        targets = self.targets_ta.read(t)
        rnn_out = self.rnn_ta.read(t)

        # Get the latent space
        p_zt = self.transition(state.latent_encoded, rnn_out)
        q_zt = self.proposal(rnn_out, state.latent_encoded, p_zt, t)
        # Sampling the latent state
        z_t = q_zt.sample(random_seed=self.random_seed)

        ## Emission phase where we reconstruct the input
        p_xt_given_zt, latent_encoded = self.emsission(z_t, rnn_out)
        log_p_xt_given_zt = tf.reduce_sum(p_xt_given_zt.log_prob(targets), axis=-1)
        log_p_zt = tf.reduce_sum(p_zt.log_prob(z_t), axis=-1)
        log_q_zt = tf.reduce_sum(q_zt.log_prob(z_t), axis=-1)
        weights = log_p_zt + log_p_xt_given_zt - log_q_zt

        if self._tilt:
            ## Calculating the log_r values for the t and t+1 timesteps
            prev_log_r = tf.cond(tf.greater(t > 0), lambda: self.tilt(
                state.rnn_out, state.latent_encoded, targets), lambda: 0.)
            log_r = tf.cond(tf.less(t + 1, self.max_len_seq), lambda: self.tilt(
                rnn_out, latent_encoded, self.targets_ta.read(t + 1)), lambda: 0.)
            log_r *= tf.to_float(t < self.seq_lengths - 1)
            weights += log_r - prev_log_r

        ## reshaping rnn_out so that it reports correctly - from the tensor array
        rnn_out = tf.reshape(rnn_out, tf.shape(state.rnn_out))

        ## Setting new state of the SRNN for all the 3 variables - i,h,c
        new_state = TrainableSRNNState(
            rnn_out=rnn_out, latent_encoded=latent_encoded, rnn_state=state.rnn_state)

        return weights, new_state


## Creating the method which does all the calls
def create_stochastic_rnn(
    data_size, latent_size, rnn_hidden_size=None,
    fcnet_hidden_sizes=None, encoded_data_size=None, encoded_latent_size=None,
    sigma_min=0.0, raw_sigma_bias=0.25, emission_bias_init=0.0, use_tilt=False,
    proposal_type='filtering', random_seed=None):

    ## Default initialization for the weights and the biases
    INITIALIZERS = {
        'w': tf.contrib.layers.xavier_initializer(),
        'b': tf.zeros_initializer()
    }

    ## Set all values which are none to be latent_size if nothing given
    if rnn_hidden_size is None:
        rnn_hidden_size = latent_size
    if fcnet_hidden_sizes is None:
        fcnet_hidden_sizes = [latent_size]
    if encoded_data_size is None:
        encoded_data_size = latent_size
    if encoded_latent_size is None:
        encoded_latent_size = latent_size

    ## Encode the data where the output_sizes are given as a list
    data_encoder = snet.nets.MLP(
        output_sizes=fcnet_hidden_sizes + [encoded_data_size],
        initializers=INITIALIZERS,
        name='data_encoder'
    )
    latent_encoder = snet.nets.MLP(
        output_sizes=fcnet_hidden_sizes + [encoded_latent_size],
        initializers=INITIALIZERS,
        name='latent_encoder'
    )
    ## By applying Conditional Normal Distribution we are getting the latent
    ## states by giving it the data
    transition = ConditionalNormalDistribution(
        size=latent_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        sigma_min=sigma_min,
        raw_sigma_bias=raw_sigma_bias,
        initializers=INITIALIZERS,
        name='prior'
    )
    emission = ConditionalNormalDistribution(
        size=data_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        initializers=INITIALIZERS,
        name='generative'
    )

    ## Instantiating 
    proposal = None
    if proposal_type in ['filtering', 'smoothing']:
        proposal = NormalApproximatePosterior(
            size=latent_size,
            hidden_layer_sizes=fcnet_hidden_sizes,
            initializers=INITIALIZERS,
            smoothing=(proposal_type == 'smoothing'),
            name='approximate_posterior'
        )

    ## Instantiating tilt with the normal distribution or None
    tilt = None
    if use_tilt:
        tilt = ConditionalNormalDistribution(
            size=data_size,
            hidden_layer_sizes=fcnet_hidden_sizes,
            initializers=INITIALIZERS,
            name='tilt'
        )

    ## Instantiate rnn cell and reverse rnn cell
    rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size, initializer=INITIALIZERS['w'])
    rev_rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size, initializer=INITIALIZERS['w'])

    return TrainableStochasticRNN(
        rnn_cell, data_encoder, latent_encoder,
        transition, emission, proposal_type, proposal=proposal,
        rev_rnn_cell=rev_rnn_cell, tilt=tilt, random_seed=random_seed
    )
