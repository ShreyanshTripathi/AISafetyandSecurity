# Model Stealing - Progress Log

## Approach Evolution Summary

| Approach | Model              | Training           | L2 Score | Notes                                              |
|----------|-------------------|--------------------|----------|----------------------------------------------------|
| 1        | Custom CNN         | One-shot           | 13.598   | Knockoff method, shallow model, no fine-tuning     |
| 2        | Custom CNN         | Full-batch         | 6.054    | One-time query but trained over multiple epochs    |
| 3        | ResNet-18          | Full training      | 5.704    | Used deeper model + standard augmentations         |
| 4        | Wide-ResNet-50     | Dual loss (O+T)    | ???      | Server crashed, but loss curve suggests < 5.0      |

---

## Evolution of Model Stealing Approaches - Technical Narrative

### **Approach 1 – Knockoff (Simple CNN, One-Shot Training)**
We began our model stealing attempt with the most straightforward strategy: a shallow custom CNN trained in a one-shot fashion using representations from a single API query of 1,000 proxy images. The goal was to understand the feasibility of the attack with minimal effort. However, this naive approach yielded an **L2 score of 13.598**, indicating poor approximation of the teacher model’s feature space. The architecture lacked sufficient depth, and the limited training strategy resulted in significant underfitting. This early failure revealed the need for deeper modeling capacity and a more effective learning procedure.

### **Approach 2 – Single-Shot Data Distillation (Same CNN, Full Batch Training)**
Building on the lessons from Approach 1, we retained the same architecture but significantly improved the training methodology. Instead of just a one-shot inference, we treated the 1,000 teacher representations as a full training set and performed several epochs of training using MSE loss. This yielded a **substantial reduction in L2 score to 6.054**, highlighting the benefits of distilling knowledge over time even from a limited dataset. Still, the architecture remained shallow, limiting how well the student could mimic complex representations. This drove our motivation to enhance the model’s representational power.

### **Approach 3 – Data Augmented Distillation (ResNet-18, Full Training)**  
To increase model capacity and generalization, we replaced the custom CNN with a deeper **ResNet-18** backbone. We also introduced **standard data augmentations** (like crops, flips, jitter) to improve robustness while still training on the same 1,000 teacher responses. The improvement was clear—**L2 dropped to 5.704**—but it plateaued soon after. Although deeper, the model was trained only on direct supervision from static labels, and there was no incentive to maintain consistency across input transformations. This limitation sparked the idea to explore training strategies that better utilized both views.

### **Approach 4 – Dual Loss + Deep Backbone (Wide-ResNet-50, Advanced Training)**
For the final and most advanced attempt, we adopted a **Wide-ResNet-50 backbone** to deepen the model's capacity, combined with **dual loss**: one on the original view and another on the augmented view. This retained consistency training from Approach 4 while unlocking greater representational power. Despite being unable to obtain the final **L2 score** due to a server crash during submission, the **training loss was the lowest of all methods**, indicating a likely performance **below 5.0**. This experiment showed that robust architecture and dual-supervision can extract remarkably accurate feature stealers even from limited API access.

### **Approach 5 – Contrastive Learning - SimCLR using Loss functions InfoNCE and SNN Loss**
Another method we implemented is a student-teacher architecture where the loss used are contrastive
learning losses. This approach is better suited because the API returns high dimensional feature
vectors instead of low dimensional labels and since contrastive losses use positive and negative
examples to place similar vectors together and dissimilar vectors far away in the representation space,
this approach is the state of the art in the literature.
To implement the solution, we took 4 different models, all of them being CNN based architecture since
the modality involved is images. These 4 models are: Simple 3 layered CNN, resnet18, resnet34 and
resnet50 backbones. Additionally, we used contrastive losses like Soft Nearest Neighbour Contrastive
Loss, Info-Noise-Contrastive Estimation (InfoNCE) loss which is another variant of SNN loss and is
shown to perform really well in some contractive learning problems. We implemented a trivial loss
called cosine similarity loss.
The API is given to be protected by B4B defence [1] which we hypothesised, can be countered by
using similar type of data in queries (i.e either belonging to the same class, or similar classes). We
did many experiments using

### **Experiments**
In contrastive learning we have several hyperparameters: like the choice of loss function, the choice of
student architecture used, the number of queries used and the number and type of transformations used.
Consequently, we carried out 4 experiments measuring the effect of change in these hyperparameters
on the L2 validation distance.

Experiment 1: In experiment 1, we changed backbones and measured its effect on L2 validation
distance. Figure 1 shows the metric on 30 epochs for all the backbones. As evident, resnet 18
performs the best. This may be because other backbones are either too complex or simple to capture
the essence of the teacher model.

Experiment 2: In experiment 2, we changed loss functions and measured its effect on L2 validation
distance. Figure 2 shows the metric on 30 epochs for all the backbones. As evident, InfoNCE loss
[ 2] performs the best. This may be because InfoNCE takes positive and negative samples and applies
softmax like function to separate pairs of postive and negative samples.

Experiment 3: In experiment 3, we changed number of queries used and measured its effect on
L2 validation distance. Figure 3 shows the metric on 30 epochs different number of queries. It is
very interesting that as less as 1000 queries are enough for a model like resnet18 to converge to a
good enough val distance. This suggests that contrastive learning is indeed a powerful technique and
augmentations can be used effectively in contrastive learning scenarios.

Experiment 4: In experiment 4, we changed transforms and measured its effect on L2 validation
distance. Figure 4 shows the metric on 30 epochs different transforms removed from the complete
dataset of original + 5 transforms (vertical and horizontal flips, rotations, inversion and affine trasnform). Removing transforms have some effect on the accuracy of the models. It is very interesting that the model without any transforms works the best. This may be because the number of
epochs we used is only 30 while the amount of data is 4 times. This suggests a need for larger, more
involved experiments

### Directory Structure
The implementation has a folder called contrastive where all the files for contrastive learning are
present:
1. attack.py : Implements the main attack class and functions
2. class_ratio.py
3. dataset.py: Implements the main dataset and query datasets class and functions
4. handler.py: the entrace to the code. Carries out the flow
5. loss.py: Implements loss functions.
6. model.py: Implements different model backbones
7. victimAPI.py: Functions related to victimAPI requests
8. utils.py: utility functions.
Other than than there is a file called knockoff.py that implements knockoff algorithm. We have
added a file called progress_log.txt which shows all the details of the progress of methods other than
contrastive loss

## Please take a look at the report for detailed results.
