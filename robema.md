
### Robust EMA estimation


The *denominator EMA term* of the learning rate can be interpreted as a norm of the input gradient signal, serving as a measure of its energy or magnitude {% cite boydLinearControllerDesign1991b %}. By normalizing the update through division by this gradient norm, the learning rule becomes scale‑normalized, always adjusted relative to the effective strength of the input. As a result, even when the squared gradient input to the denominator EMA is corrupted by heavy‑tailed noise or occasional outliers, the normalization absorbs these effects. Extreme values in the gradient are proportionally scaled down, preventing instability and ensuring that the update remains bounded and robust.

However, the same cannot be said for the *numerator EMA term*. Whereas the denominator term acts as a norm of the gradient signal and thus provides scale‑normalized robustness, the numerator term directly involves the correlation between a weight and a gradient component. This correlation is inherently more sensitive to noise and outliers: if the weight-gradient product is corrupted, the output of the numerator EMA can be distorted in both magnitude and sign. Unlike the denominator, which absorbs extreme values through normalization, the numerator reflects them directly, potentially leading to erratic updates. In practice, this means that while the denominator stabilizes the learning rate by bounding its scale, the numerator remains the primary channel through which input variability and heavy‑tailed disturbances distort the update step.

Classic mean estimators like the EMA assume a well-behaved noise model {% cite huberRobustEstimationLocation1992 zoubirRobustEstimationSignal2012 %}. **Heavy‑tailed stochastic correlations, noisy sign flips and occasional magnitude spikes can break this assumption** {% cite zoubirRobustStatisticsSignal2018a %} leading to breakdown. In other words, heavy-tailed gradient noise statistics induce misleading spikes that can dominate the EMA's estimate over many iterations by increasing its bias from the true mean estimate. 

>To handle such problems, a common approach in the robust estimation of location from data is to essentially apply **concentration inequality** techniques that **detect** if the input to the mean estimator is suspicious (an outlier), then **replace** (or **clip**) with an appropriate value using a measure of scale {% cite zoubirRobustEstimationSignal2012 zoubirRobustStatisticsSignal2018a %}.

In this case, we want the estimate of the **numerator EMA** to remain positive, well‑bounded, and avoid corruptions due to noisy, heavy-tailed inputs. Limiting the influence of outliers while preserving sensitivity to smaller signals.
In other words, we want to robustify the estimate from the numerator EMA without distorting the bulk of the signal observed via
its input
$$ \mathbf{u}(t,i) = \mathbf{w}(t,i) \, \bar{\mathbf{g}}(t,i) $$

#### 1. Input Clipping
Since, we do not know the probability distribution, **Markov's inequality** gives a rationale for how often large such values can occur. Let $$u$$ denote an instantaneous input signal, and $$c > 0$$ be a scale constant. Markov's inequality

$$
\mathbb{Pr}[|u| \ge c\,\mathbb{E}[|u|] ] \le \frac{1}{c},
$$

relates how large the magnitude of $$ u $$ can be relative to its expected magnitude 
$$ m = \mathbb{E} \bigl[|u|\bigr] $$. 

<!-- **Soft limiting** is a practical way to robustly mitigate such heavy-tailed values.  -->

By utilizing Markov's inequality, instead of naively passing an input $$u$$ though the EMA, the **Huber clipping function** $$\psi_{c}(u)$$ {% cite zoubirRobustEstimationSignal2012 Menon2020GradientClipping %} can be used to detect the most extreme outliers ($$ > c\times m$$) and replace with $$c \times m$$ before they are processed by the EMA. This avoids signal dead-zones of zero and allows moderate estimates to pass through untouched relative to the expected magnitude. 

$$
\psi_{c}(u) =
\begin{cases}
u, & |u| \le c\,\mathbb{E}[|u|] \\
\mathrm{sign}(u) \cdot c\,\mathbb{E}[|u|], & |u| > c\,\mathbb{E}[|u|].
\end{cases}
$$

The scale constant $$c$$ is used to clip the extreme outliers relative to $$m$$ For example, $$c=4$$ can be viewed as a *prior* that the probability $$p$$ of the magnitude $$|u|$$ exceeding four times its expectation is no more than $$25\%$$. Equivalently, the probability that $$|u|$$ remains below this threshold is at least $$1-p=75\%$$.
Therefore, the interval defined by the 25–75% quantiles capture a good percentage of the distribution, while the clipping function suppresses only the most extreme values. This yields a more **robust EMA estimator** that is less sensitive to heavy‑tailed noise and spurious magnitude spikes.

<!-- , and so the probability of the magnitude being less than $$4$$ times its mean is at least $$1-p=75\%$$. T -->

In the case of $$ \mathbf{u}(t,i) $$, we can iteratively estimate its expected magnitude, via the EMA estimate

$$
\hat{\mathbf{u}}(t,i) = \beta_a \, \hat{\mathbf{u}}(t-1,i) + (1 - \beta_a)\,|{\mathbf{u}}(t,i)|,
$$

where $$\hat{\mathbf{u}}(t,i)$$ adapts to the typical scale of $$\mathbf{u}(t,i)$$ in each layer.


#### 2. Output Clipping and Max-Normalization
Furthermore, small noisy inputs that slip through the input-clipping function can flip the signs of numerator EMA's estimate.
These **spurious sign flips** distort the learning‑rate ratio, sometimes inflating it and other times collapsing it to near zero. A simple input‑clipping strategy cannot prevent this problem: when the estimate spuriously turns negative, clipping to zero halts progress entirely.

Since the global learning‑rate constant $$\mu$$ already serves as a safety margin (a trust‑region) for the locally-optimal learning rate, a more robust limiting approach that utilizes $$\mu$$ is needed. Drawing on robust filtering principles {% cite zoubirRobustStatisticsSignal2018a %}, one approach is to **preserve sign information** only within the EMA update via the input clipping, while at the output stage, a **magnitude-only bound** is applied to produce the final estimate.

To achieve this, let $$s$$ be the upper-bound on a signal $$u$$. We can define a second clipping function $$\psi_{\mu}(u)$$ that allows small values of 
$$|u|$$ to pass unchanged, but limits large values to the threshold $$s$$ scaled by $$\mu$$.

$$
\psi_{\mu}(u) =
\begin{cases}
|u|, & |u| \le \mu\,s \\ \mu\,\mathbb{E}[|u|], & |u| > \mu\,s.
\end{cases}
$$

In our case, for the inequality $$ 0 \le (\mathbf{w}(t,i)-\bar{\mathbf{g}}(t,i))^2 $$, we have that
$$
\mathbf{w}(t,i) \,\bar{\mathbf{g}}(t,i) \ \le\ \frac{1}{2}\,\big(\mathbf{w}(t,i)^2 + \bar{\mathbf{g}}(t,i)^2\big),
$$
and so obtain the upper-bound for $$ \mathbf{u}(t,i) $$

$$
\mathbb{E}[\mathbf{w}(t,i) \,\bar{\mathbf{g}}(t,i)] \le \,\mathbb{E}[\mathbf{w}(t,i)^2] + \mathbb{E}[\bar{\mathbf{g}}(t,i)^2] = \mathbb{E}[\mathbf{w}(t,i)^2] + 1.
$$

To realize a dynamic max-threshold, we can realize $$\mathbb{E}[\mathbf{w}(t,i)^2]$$ by maintaining an EMA estimate 

$$
\mathbf{s}(t,i) = \beta_a \,\mathbf{s}(t-1,i) + (1 - \beta_a) \,\mathbf{w}(t,i)^2,
$$

then define the max-normalizer as $$\bar{\mathbf{s}}(t,i) = 1 + \mathbf{s}(t,i)$$.

We can now realize a robust numerator EMA estimate as follows:

$$
\tilde{\mathbf{a}}(t,i) = \beta_a \, \tilde{\mathbf{a}}(t-1,i) + \mu\, \bar{\mathbf{s}}(t,i)^{-1}\cdot{\psi_{c} (\mathbf{u}(t,i))},
$$

$$\mathbf{a}(t,i) = \psi_{\mu}\bigl(\tilde{\mathbf{a}}(t,i)\bigr).$$

<!-- $$
\mathbf{a}(t,i) = \max\bigl(\mu\,\bar{\mathbf{s}}(t,i)^{-1}\cdot\mathbf{d}(t,i),\, \min\bigl( |\tilde{\mathbf{a}}(t,i)|, \, \mu \bigr) \bigr).
$$ -->

Taken together, the small $$\epsilon
$$, the normalizers $$\bar{\mathbf{s}}(t,i)$$ and $$\mathbf{d}(t,i)$$ and robust input-output clipping functions help us realize a learning rate estimate that remains within a predictable dynamic range, while preventing large values that can lead to **breakdown**. 

> In terms of implementation, note that 
$$\psi_{c}\bigl(u\bigr) = \mathrm{sign}(u) \cdot \min\bigl( |u|, \, c\,m\bigr),$$ and  $$\psi_{\mu}\bigl(u\bigr) = \min\bigl( |u|, \, \mu\,s\bigr).$$

#### 3. Layer-wise Average 
In addition, to account for intra-layer structure and variability in deep neural networks, we observed that replacing the raw $$ \mathbf{a}(t,i) $$ estimates with their layerwise mean further ensured more numerically stable and uniform parameter adaptation within each layer. Specifically, for a given layer $$\ell$$, with parameter size $$n_\ell$$, the numerator estimates are averaged to yield a uniform estimate:

$$
\mathbf{a}(t,i) ← \frac{1}{n_\ell} \sum_{i=1}^{n_\ell} \mathbf{a}(t,i).
$$

<!-- ### Alternatives: Relaxed Upper-bound variant
Since
$$
\mathbf{u}(t,i) = \mathbf{w}(t,i) \,\bar{\mathbf{g}}(t,i) \ \le\ \frac{1}{2}\,\big(\mathbf{w}(t,i)^2 + \bar{\mathbf{g}}(t,i)^2\big),
$$
we may replace $$\mathbf{u}(t,i)$$ with a relaxed form of the symmetric upper-bound,
$$ \tilde{\mathbf{u}}(t,i) = \mathbf{w}(t,i)^2 + \bar{\mathbf{g}}(t,i)^2$$, weighted by $$ \mu < \frac{1}{2} $$. A proxy estimate of the partial-correlation then becomes

$$
\tilde{\mathbf{a}}(t,i)  = \beta_a \,\tilde{\mathbf{a}}(t-1,i)  + (1 - \beta_a) \,\mu\,\tilde{\mathbf{u}}(t,i).
$$

$$
\mathbf{a}(t,i) = \min\bigl( |\tilde{\mathbf{a}}(t,i)|, \, \mu \bigr).
$$

Layer-wise smoothing can be applied next. -->

<!-- --- -->

---

{% bibliography --cited %} 