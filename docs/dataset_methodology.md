# 🎓 Dataset Generation Methodology & Statistical Modeling Specification
### Course: IT462 Machine Learning | Course Project

---

## 📌 Executive Summary
Due to data privacy regulations (e.g., DPDP Act and FERPA) prohibiting public sharing of individual academic marks, backlog histories, and confidential corporate compensation contracts, this project uses an **institutional benchmark dataset of 100,000 university students** across 8 engineering disciplines.

Rather than generating arbitrary random numbers, every feature is governed by the specific statistical distribution that describes that phenomenon in real university and corporate recruitment environments.

The entire generation pipeline is implemented in [`scripts/generate_placement_data.py`](../scripts/generate_placement_data.py) and is 100% reproducible via `random_state=42`.

---

## 🔬 Mathematical Distributions & Real-World Justifications

### 1. Academic Performance (`cpi`)
* **Distribution:** Truncated Gaussian $\mathcal{N}(\mu=7.3, \sigma=1.1)$, clipped to $[4.5, 10.0]$.
* **Statistical Justification:** By the **Central Limit Theorem**, continuous aggregated student grades naturally follow a bell curve centered around 7.0–7.5.
* **Domain Realism:** Indian engineering institutions operate on a 10.0 scale, where 4.5 is the minimum passing threshold.

### 2. Branch Cohort Sizes (`branch`)
* **Distribution:** Discrete Categorical Selection with defined seat probabilities:
  * CSE (25%), ICT (20%), IT (15%), ECE (15%), MnC (8%), EE (7%), Mechanical (5%), Civil (5%).
* **Domain Realism:** Matches actual institutional seat matrices where computer and electronics branches have significantly higher enrollment than specialized core branches.

### 3. Backlog Distribution & CPI Impact (`backlogs`)
* **Distribution:** Discrete Empirical Frequency:
  * 0 Backlogs: 72% | 1 Backlog: 16% | 2 Backlogs: 7% | 3 Backlogs: 3% | 4 Backlogs: 2%.
* **Cross-Feature Penalty:** Students with $\ge 2$ backlogs receive a conditional negative shift on their CPI ($cpi - \Delta$).
* **Domain Realism:** You cannot fail multiple courses and maintain a 9.5 CPI. This introduces realistic correlation between course failure and overall grade index.

### 4. Technical & Coding Skills (`coding_score`, `dsa_score`)
* **Distribution:** Correlated Linear Projection with Gaussian Residuals.
  * $\text{Coding Score} = 0.5 \times \text{CPI} + \mathcal{N}(2.2, 1.8)$, clipped to $[1.0, 10.0]$.
  * $\text{DSA Score} = 0.7 \times \text{Coding Score} + \mathcal{N}(1.6, 1.4)$, clipped to $[1.0, 10.0]$.
* **Domain Realism:** Academic aptitude provides a foundation for programming, but does not strictly dictate it. Some 6.5 CPI students are extraordinary coders, while some 9.0 CPI students struggle with practical implementation. The residual variance models this divergence.

### 5. Algorithmic Problem Solving (`dsa_problems_solved`)
* **Distribution:** Linear Skill Baseline + **Exponential Distribution** ($\beta = 35$).
* **Statistical Justification:** Competitive programming (LeetCode, Codeforces) follows a **Power-Law / Pareto Distribution (80/20 Rule)**. The median student solves 100–180 questions, while an elite top 5% solves 500–800+ problems. The exponential distribution models this heavy right-tail.

### 6. Projects Portfolio (`projects_count`, `has_major_project`)
* **Distribution:** **Poisson Process** ($\lambda = 2.4$), bounded $[0, 6]$.
* **Statistical Justification:** In probability theory, the Poisson distribution is the standard model for counting discrete, independent events occurring within a fixed interval (4 years of undergraduate study).
* **Major Project Indicator:** Modeled via a Bernoulli trial whose success probability scales with project count and coding competence.

### 7. Industry Internships & Experience (`internships`, `experience_months`)
* **Distribution:** **Binomial Distribution** ($n=3$ trials).
* **Statistical Justification:** An undergraduate engineering program contains exactly 3 summer breaks. Each summer break represents an independent trial where success probability $p$ depends on CPI and coding ability.
* **Duration Multiplier:** Internships convert into months using authentic durations: 2 months (summer), 3 months (extended summer), or 6 months (final semester industrial training).

### 8. Aptitude & Soft Skills (`aptitude_score`, `communication_skills`)
* **Distribution:** Standardized Normal distributions.
  * `aptitude_score`: $\mathcal{N}(71.0, 13.5)$ on scale $[25, 100]$ (standard cognitive bell curve).
  * `communication_skills`: $\mathcal{N}(6.8, 1.5)$ on scale $[2.0, 10.0]$.

---

## 🎯 Target Variables: The Hiring Decision & Salary

### 1. `placement_status` (Classification Target)
* **Model Used:** **Multivariate Logistic Choice Model (Sigmoid)**
  $$\text{Logit}(z) = 1.25(\text{CPI} - 7) + 0.55(\text{Coding} - 5.5) + 0.75(\text{Internships} - 0.8) + 0.40(\text{Projects} - 2) - 1.60(\text{Backlogs}) + \dots + \epsilon$$
  $$P(\text{Placement}) = \frac{1}{1 + e^{-z}}$$
* **Why this function?**
  * Hiring is an econometric **Discrete Choice Problem** where recruiters assess a candidate's composite latent utility.
  * Gaussian noise $\epsilon \sim \mathcal{N}(0, 1.2)$ is added to model real-world interview variance (luck, interviewer bias, test questions).
  * This calibrates the baseline Random Forest classifier to a realistic **~82.6% accuracy and 0.897 ROC-AUC**, rather than an artificial, overfitted 100%.

### 2. `salary_lpa` (Package Target)
* **Model Used:** **Linear Baseline + Exponential Elite Compensation**
  $$\text{Base LPA} = 4.2 + 0.75(\text{CPI} - 6) + 1.0(\text{Coding} - 4.5) + 1.4(\text{Internships}) + \dots$$
  $$\text{Elite Bonus} = \text{Exponential}(\beta = 7.5) \quad \text{for top coders}$$
* **Domain Realism:** Indian tech campus hiring is bimodal:
  * Standard service/core offers: 4.5 to 8 LPA.
  * Mid-tier product offers: 10 to 18 LPA.
  * Elite FAANG / product offers: 25 to 48 LPA.
* **Missing Value Integrity:** Unplaced candidates have `NaN` values, giving the data science pipeline a realistic missing-data handling challenge.
