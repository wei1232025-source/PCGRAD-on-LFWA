### Introduction to Multi-Objective Optimization Task Concepts
Multi-objective optimization (MOO) problems aim to find Pareto optimal solutions globally.

#### What is Pareto Optimality?

In multi-objective optimization (MOO) problems in machine learning, because different objectives are often in conflict (e.g., desiring both high model accuracy and a smaller model size with faster inference speed), a single solution that simultaneously optimizes all objectives to their absolute perfect states rarely exists.

To evaluate the "best" state under such multi-objective conflicts, we introduce the concept of the **Pareto Optimal Solution**.

---

### Our Work
We implement the gradient surgery algorithm and validate its performance when handling multi-objective conflicts. To achieve this, this experiment constructs a conflicting task using real-world data:

- Objective 1 (Utility): High-accuracy prediction of whether a face is smiling using shared backbone features.
- Objective 2 (Fairness/De-information): Prevent the shared backbone features from containing Male information that could be exploited by a gender attacker.

These two objectives may inherently conflict. If there is a statistical correlation between Smiling and Male in the LFWA dataset, the model might use gender as a shortcut feature to improve smiling prediction accuracy. Conversely, the fairness objective requires the backbone to eliminate such gender information.

#### Mathematical Principles of the Gradient Surgery Algorithm
When optimizing `smile_loss` (preserving smiling) and `degender_loss` (eliminating gender) simultaneously on the backbone network, the gradient directions of these two tasks often conflict (as shown on the right side of the results plot, where the cosine similarity remains negative over the long term).

The traditional `naive_adv` (direct addition: `smile_loss + degender_loss`) can cause gradients to cancel each other out, leading to unstable training or even degradation of smiling prediction accuracy.

PCGrad (Gradient Surgery) addresses this issue through the following steps (as implemented in the `pcgrad_two_task` function):

1. Conflict Detection
Calculate the dot product of the smiling task gradient $g_{\text{smile}}$ and the de-biasing task gradient $g_{\text{degender}}$:
$$\text{dot} = g_{\text{smile}} \cdot g_{\text{degender}}$$
* If the dot product $\text{dot} < 0$, the cosine similarity is negative, indicating that the two gradients point in opposite directions and thus **conflict**.

2. Gradient Projection (Surgery Process)
If a conflict occurs, we project the gradient of each task onto the orthogonal plane of the other task's gradient, thereby eliminating the opposing components:
$$g_{\text{smile}} \leftarrow g_{\text{smile}} - \frac{g_{\text{smile}} \cdot g_{\text{degender}}}{\|g_{\text{degender}}\|^2} g_{\text{degender}}$$
$$g_{\text{degender}} \leftarrow g_{\text{degender}} - \frac{g_{\text{degender}} \cdot g_{\text{smile}}}{\|g_{\text{smile}}\|^2} g_{\text{smile}}$$
* **Physical Intuition**: The modified $g_{\text{smile}}$ will no longer apply a force that degrades `degender_loss`. Similarly, the modified $g_{\text{degender}}$ will not disrupt the optimization process of `smile_loss`. The directions of both gradients are mathematically adjusted to be orthogonal or positively aligned.
* **Gradient Merging**: The final gradient applied to update the backbone network is the sum of the two projected gradients: $g_{\text{merged}} = g_{\text{smile}} + g_{\text{degender}}$.

---

#### LFWA Dataset Analysis
>Total samples: 13143
Male ratio: 77.46%
Gender majority-class baseline: 77.46%
Smiling ratio: 41.33%
Male & Smiling: 3477
Male & Not Smiling: 6704
Female & Smiling: 1955
Female & Not Smiling: 1007

1. Severe Imbalance in Sample Size (Quantity Bias)

First, let's look at the baseline comparison of male and female samples:
* **Total Samples**: $13,143$
* **Male Samples**: $13,143 \times 77.46\% = 10,181$
* **Female Samples**: $13,143 - 10,181 = 2,962$

**Cause of Bias:**
* **Male-Dominant Feature Extractor**: Since male samples account for a high percentage of **$77.46\%$** (nearly 3.5 times the female samples), without any intervention, the shared backbone network will naturally bias towards learning and fitting male facial features to minimize the overall training loss. Consequently, this can lead to relatively lower accuracy in the extracted feature representations when the model processes female facial images.

---

2. "Spurious Correlation" between Gender and Smiling (Association Bias)

This is the core reason why the model develops prediction bias (i.e., treating "gender" as a shortcut for "smiling" prediction). We quantify this using conditional probabilities:

A. Smiling probability in the male group $P(\text{Smile} \mid \text{Male})$
* Smiling males: $3,477$
* Non-smiling males: $6,704$
* Total males: $10,181$
* **Male Smiling Ratio**: $\frac{3477}{10181} \approx \mathbf{34.15\%}$

B. Smiling probability in the female group $P(\text{Smile} \mid \text{Female})$
* Smiling females: $1,955$
* Non-smiling females: $1,007$
* Total females: $2,962$
* **Female Smiling Ratio**: $\frac{1955}{2962} \approx \mathbf{66.00\%}$

**Cause of Bias:**
* **Highly Imbalanced Prior Distribution**: In the dataset, the **probability of females smiling ($66\%$) is nearly double that of males ($34.15\%$)**.
* **Strong Statistical Association**: This indicates a strong positive correlation between "female" and "smiling", and "male" and "not smiling".

#### Log Analysis
Based on the code implementation and the line charts generated during execution, we can derive the following analysis:

1. **Utility: Smiling Prediction (Main Task Accuracy)**:
   * Since `SMILE ONLY` (blue) does not need to consider any de-biasing task, its main task accuracy remains relatively high in the later stages (around 0.88).
   * Both `NAIVE ADV` (yellow) and `PCGRAD` (red) show slight fluctuations in the main task accuracy after introducing adversarial de-biasing. However, `PCGRAD` maintains a solid smiling prediction accuracy while providing privacy/de-biasing protection. Compared to the direct addition of `NAIVE ADV`, its performance is more stable across several epochs.

2. **Leakage: Gender Attack (Gender Leakage Status)**:
   * Our objective is to **reduce the balanced accuracy of gender to 0.50 (random guess)**, indicating that the features do not contain any gender information.
   * `SMILE ONLY` (blue) has the highest gender prediction accuracy (0.60 - 0.67), showing that training solely on the smiling prediction task severely leaks gender bias.
   * Both `NAIVE ADV` (yellow) and `PCGRAD` (red) significantly reduce gender leakage. Among them, **`PCGRAD` demonstrates a stronger de-biasing capability**, with its curve lying closer to the 0.50 "random guess" baseline, and even dropping below 0.50 in several instances (reaching as low as 0.475 during epoch 3). This indicates a more effective de-biasing outcome.

3. **Backbone Gradient Conflict (Gradient Conflict Analysis)**:
   * This plot directly validates the core motivation behind the PCGrad algorithm.
   * The cosine similarity of `PCGRAD` (red) on the backbone network consistently remains between $-0.10$ and $-0.40$. **This provides strong evidence of a deep gradient conflict between the two objectives of "preserving smiling" and "eliminating gender".**
   * Traditional direct addition (`naive_adv`) suffers from internal cancellation due to this negative correlation. In contrast, by orthogonalizing the gradient projections at each step, `PCGRAD` mitigates this internal conflict, thereby achieving a better trade-off between "Utility" and "Fairness/De-biasing".