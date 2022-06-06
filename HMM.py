import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf

tfd = tfp.distributions  # making a shortcut for later on
# 80% chance first day is cold
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
# cold after cold 70%, hot after cold 30%
# cold after hot 20%, hot after hot 80%
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                 [0.2, 0.8]])
# cold day: temp -5 to  5, mean 0
# hot day: temp 5 to 25, mean 15
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

# the loc argument represents the mean and the scale is the standard deviation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)


mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:
  print(mean.numpy())
