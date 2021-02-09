# AI learns to play snake

A genetic algorithm is used to train a neural network to play snake. The neural network has 24 input values, 2 hidden layers each with 16
neurons and 4 output values representing the four different directions, i.e
left, right, up and down. The 24 input values are as followed:
1.  In 4 directions distance to the wall
2.  In 8 directions if a body part is close to the snakes head (binary)
3.  In 8 directions if an apple is present (binary)
4.  Direction of its movement (binary)

### Crossover
The genetic algorithm uses a single point binary crossover between parent 1
and parent 2.

### Mutation
Radom values sampled from a Gaussian distribution with a standard deviation
of 0.5 and a mean of 0 are being used as random mutations. These mutations are
added to the DNA values of the individuals

### Fitness function
The fitness function that is being used is copied from https://chrispresso.io/AI_Learns_To_Play_Snake.

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>f</mi>
  <mo stretchy="false">(</mo>
  <mi>s</mi>
  <mi>t</mi>
  <mi>e</mi>
  <mi>p</mi>
  <mi>s</mi>
  <mo>,</mo>
  <mi>a</mi>
  <mi>p</mi>
  <mi>p</mi>
  <mi>l</mi>
  <mi>e</mi>
  <mi>s</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>s</mi>
  <mi>t</mi>
  <mi>e</mi>
  <mi>p</mi>
  <mi>s</mi>
  <mo>+</mo>
  <mo stretchy="false">(</mo>
  <msup>
    <mn>2</mn>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>a</mi>
      <mi>p</mi>
      <mi>p</mi>
      <mi>l</mi>
      <mi>e</mi>
      <mi>s</mi>
    </mrow>
  </msup>
  <mo>+</mo>
  <mn>500</mn>
  <mo>&#x2217;<!-- ∗ --></mo>
  <mi>a</mi>
  <mi>p</mi>
  <mi>p</mi>
  <mi>l</mi>
  <mi>e</mi>
  <msup>
    <mi>s</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mn>2.1</mn>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
  <mo>&#x2212;<!-- − --></mo>
  <mo stretchy="false">(</mo>
  <mn>0.25</mn>
  <mo>&#x2217;<!-- ∗ --></mo>
  <mi>s</mi>
  <mi>t</mi>
  <mi>e</mi>
  <mi>p</mi>
  <msup>
    <mi>s</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mn>1.3</mn>
    </mrow>
  </msup>
  <mo>&#x2217;<!-- ∗ --></mo>
  <mi>a</mi>
  <mi>p</mi>
  <mi>p</mi>
  <mi>l</mi>
  <mi>e</mi>
  <msup>
    <mi>s</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mn>1.2</mn>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
</math>

The fitness function takes into account the following aspects:
1. Reward snakes early on for exploration + finding a couple apples
2. Have an increasing reward for snakes as they find more apples
3. Penalize snakes for taking a lot of steps

### Dashboard
A simple is dashboard is generated to train and view the AI to play snake  
