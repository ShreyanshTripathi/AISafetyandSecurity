# TML25_A3_03

# Assignment 3  
**Trustworthy Machine Learning**  
_Saarland University – Summer Semester 2025_  

**Team 49/3**  
- Shreyansh Tripathi (7056894)  
- Vishnuvasan Thadan Srinivasan (7075887)  

---

## Abstract

In this assignment, we implement methods to train a robust classifier with the highest possible accuracy for both clean and adversarial data points. This report discusses potential solutions, our implementations, and several experiments, along with the results and insights extracted from them.

---

## Methodology

We implemented two different modeling approaches:

1. **FGSM Adversarial Training**
2. **FAST**

### 1. FGSM Adversarial Training

We began by training a simple ResNet-18 model using the FGSM Adversarial Training approach.  
- **Epsilon:** 8/255 (≈ 0.031)
- FGSM perturbations were generated as follows:

    ```
    x_adv = x + epsilon * sign(grad)
    ```

- We calculated scores for:
  - FGSM adversarial accuracy
  - PGD adversarial accuracy (α = 0.007)
  - Clean accuracy

#### Results

| Evaluation                                 | Accuracy |
|---------------------------------------------|----------|
| Final Clean Accuracy                        | 0.6065   |
| FGSM (ε=0.03) Adversarial Accuracy          | 0.3924   |
| PGD (ε=0.03, iter=5) Adversarial Accuracy   | 0.3767   |

*Table 1: Clean and Adversarial Accuracies for Single-Step FGSM Adversarial Training*

![Accuracies on different sets for Single-Step FGSM Adversarial Training](image.png)

---

### 2. FAST

Another method we implemented is FAST.  
- **Epsilon:** 8/255  
- **Alpha:** 2/255  

#### Results

| Evaluation                                 | Accuracy |
|---------------------------------------------|----------|
| Final Clean Accuracy                        | 0.6092   |
| FGSM (ε=0.03) Adversarial Accuracy          | 0.2783   |
| PGD (ε=0.03, iter=5) Adversarial Accuracy   | 0.2782   |

*Table 2: Clean and Adversarial Accuracies for Mixed Adversarial Training*

![Clean Accuracies over different epochs for Mixed Adversarial Training](acc_chart.png)

---

## Directory Structure

The implementation includes a folder called **contrastive** with the following files:

- `fgsm.py` : Implements single step FGSM.
- `hybrid.py` : Implements FAST solution, mixed adversarial training.
- `shot.py` : Our solution that gave best results
- `logs/` : Logs for different trainings.

Training logs are available for various datasets.

**Repository:**  
[https://github.com/Cipher-unhsiV/TML25_A3_03/](https://github.com/Cipher-unhsiV/TML25_A3_03/)

---

## Acknowledgements

- Assignment lectures and paper by Dubiński et al. (2023)
- Reading from book by [jm3]
- Implementation using Python 3 and PyTorch [Paszke et al., 2019]
- Tables generated with [tablesgenerator.com](https://www.tablesgenerator.com/)

**References**  
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.  
- Dubiński, et al. (2023).  
- [tablesgenerator.com](https://www.tablesgenerator.com/)
