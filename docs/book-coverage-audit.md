# Book coverage audit (docs/ml-topics.txt)

Audit date: 2026-08-05. CLOSED 2026-08-05: the glossary entries
(12 new + 4 extensions), the four diagrams, both findability
fixes, and nine of the ten demos shipped the same day (four
classical-ML demos in Classification; distributions/CLT, Bayes
grid, power iteration, augmentation, and toy segmentation across
Basics / Dim Reduction / Training and Learning / Vision). The
VAE demo was DESCOPED after testing: a one-dimensional-latent
toy with a linear encoder collapses to the data mean at
whiteboard scale (reconstruction MSE equals the data variance);
the Autoencoder demo and the VAE glossary entry carry the
concept. Source: the three eBook outlines in
`docs/ml-topics.txt` (Grokking Machine Learning 2nd ed., Deep
Learning with PyTorch 2nd ed., Math and Architectures of Deep
Learning), checked chapter by chapter against the four coverage
surfaces: glossary entries (427), demos (69), diagrams (48),
lessons (58).

Verdict: coverage of the neural-network, transformer, tensor,
and training-mechanics spine is essentially complete across all
four surfaces. The gaps cluster in five areas: (1) classical-ML
algorithms that have glossary entries and diagrams but no
runnable demo, (2) the probability/Bayes toolkit, (3) the
linear-algebra underpinnings (eigenvectors, SVD, Jacobian,
Hessian), (4) evaluation-metrics pedagogy, and (5) a handful of
task types (segmentation, detection, augmentation).

## Chapter-by-chapter status

### Grokking Machine Learning

| Ch | Topic | Status |
|----|-------|--------|
| 1-2 | What is ML; types of ML | COVERED (lessons, glossary) |
| 3 | Linear regression | COVERED -- the "How Gradient Descent Works" demo fits y = w*x + b by MSE; it does not SAY "linear regression" (findability fix queued) |
| 4 | Under/overfitting, testing, regularization | COVERED (three demos, glossary, val_split) |
| 5 | Perceptron | MOSTLY -- glossary + diagram; the logical-gates demo is a perceptron in fact but not in name |
| 6 | Logistic classifiers | COVERED (demo, lesson, diagram) |
| 7 | Accuracy and its friends | PARTIAL -- Precision vs Recall / F1 / ROC-AUC / confusion_matrix exist; no "Accuracy" glossary entry, and no metrics-focused demo (thresholds, precision-recall tradeoff) |
| 8 | Naive Bayes | MOSTLY -- demo + glossary; no diagram |
| 9 | Decision trees | PARTIAL -- glossary + diagram; no demo (no recursive structures in-language; a depth-2 stump demo is feasible) |
| 10 | Neural networks | COVERED (extensively) |
| 11 | SVM + kernel method | PARTIAL -- SVM glossary + diagram; NO kernel-method glossary entry, no demo (linear SVM via hinge loss is expressible today; kernel trick via explicit feature maps) |
| 12 | Ensemble learning | PARTIAL -- Random Forest / Gradient Boosting / XGBoost entries + RF diagram; no umbrella "Ensemble learning" entry (bagging/voting/stacking), no demo (a voting ensemble of small MLPs is expressible today) |
| 13 | Data engineering practice | MOSTLY -- the Data Forge category covers construction; no "Feature engineering" glossary entry |
| 14 | Generative AI (language + images) | COVERED (Tiny LM, GAN, Diffusion demos) |

### Deep Learning with PyTorch

| Ch | Topic | Status |
|----|-------|--------|
| 1 | Intro | COVERED |
| 2 | Pretrained networks | CONCEPT ONLY -- Transfer Learning + Fine-tuning entries, LoRA demos; actual model IMPORT is the queued interchange work (Track 7) |
| 3-4 | Tensors; data as tensors | COVERED (Structure Zoo, lessons, upload) |
| 5-6 | Mechanics of learning; fitting | COVERED |
| 7 | Image classification | COVERED (Pets demos) |
| 8 | Convolutions | COVERED (CNN demo, conv2d/pool2d, diagram) |
| 9 | Transformers | COVERED (extensively) |
| 10 | Diffusion models | COVERED (demo + diagram) |
| 11-13 | The cancer-detection project (datasets, classifier) | OUT OF SCOPE as a project; its concepts (dataset assembly, classification) are covered generically |
| 14 | Metrics + augmentation | PARTIAL -- Data Augmentation glossary entry exists; no augmentation demo or helper idioms (shift/flip/noise on arrays are expressible) |
| 15 | Segmentation | GAP -- U-Net entry + diagram exist; no "Segmentation" glossary entry, no demo (a toy 2D mask task is expressible) |
| 16 | Multi-GPU training | CONCEPT ONLY (deliberate) -- data/tensor/pipeline-parallel entries + 3 diagrams; hands-on multi-GPU is not an MLPL capability |
| 17 | Deployment | PARTIAL -- mlpl-serve + connect mode ARE the deployment story; no glossary entry naming the concept (inference serving) |

### Math and Architectures of Deep Learning

| Ch | Topic | Status |
|----|-------|--------|
| 1-2 | Overview; vectors/matrices/tensors | COVERED |
| 3 | Vector calculus | PARTIAL -- Gradient covered deeply; no Jacobian or Hessian glossary entries |
| 4 | Linear-algebra tools | GAP -- PCA is covered, but no Eigenvalue/Eigenvector or SVD entries; no demo (power iteration for the top eigenvector is expressible today, no new builtins) |
| 5 | Probability distributions | GAP -- no Gaussian/Normal, Bernoulli, or Uniform entries; a distributions demo (randn / rand_ints + hist) is expressible today |
| 6 | Bayesian tools | GAP -- only Naive Bayes exists; no Bayes' theorem, prior/posterior/likelihood, MLE-vs-MAP entries; a coin-flip grid-posterior demo is expressible today |
| 7 | Function approximation | COVERED (Universal Approximation entry, MLP demos) |
| 8 | Forward/backprop | COVERED (Gradient Flow demo, diagram, lessons) |
| 9 | Loss, optimization, regularization | MOSTLY -- optimizers/schedules/weight-decay covered; Dropout is glossary-only (no training-time dropout in the runtime); no L1-vs-L2 comparison |
| 10 | Convolutions | COVERED |
| 11 | Image classification + object detection | PARTIAL -- classification covered; no Object Detection / IoU entries, no demo |
| 12 | Manifolds + homeomorphism | MOSTLY -- Manifold Hypothesis + dim-reduction zoo; homeomorphism unmentioned (one-line extension of the existing entry) |
| 13 | Fully-Bayes parameter estimation | GAP -- no entry; a 1-parameter grid-posterior demo is expressible today |
| 14 | Latent space, AE, VAE | MOSTLY -- Latent Space + Autoencoder + VAE entries, AE demo; no VAE demo (reparameterization trick is expressible with noise hoisted outside grad()) |

## The gap list, by cheapest sufficient fix

Glossary entries (~16 small additions): Accuracy, Kernel method,
Ensemble learning (bagging/voting/stacking), Feature engineering,
Bayes' theorem, Prior / Posterior / Likelihood, MLE vs MAP,
Gaussian (Normal), Bernoulli, Uniform, Eigenvalues/Eigenvectors,
SVD, Jacobian, Hessian, Segmentation, Object detection (IoU),
Inference serving. Plus one-line extensions: homeomorphism (in
Manifold Hypothesis), L1 vs L2 (in the weight-decay entry).

Demos expressible with TODAY's builtins (no runtime work):
metrics playground (accuracy/precision/recall vs threshold),
linear SVM via hinge loss (+ explicit feature-map kernel trick),
voting ensemble of small MLPs, power iteration (top eigenvector),
distributions + histograms, Bayes grid posterior (coin flip),
data augmentation (shift/flip/noise), toy 2D segmentation mask,
VAE (noise hoisted), decision stump.

Diagrams (~4): naive Bayes, kernel trick, Bayes' theorem,
eigenvectors/SVD picture.

Findability fixes (text-only): the gradient-descent demo intro
should say "linear regression"; the logical-gates demo should say
"perceptron".

Runtime work (only if demos want it later): training-time
dropout; beyond that, NO new builtins are required for any gap
above.

Explicit non-goals (state plainly, do not chase): hands-on
multi-GPU training, the PyTorch medical-imaging project as a
project, production deployment beyond mlpl-serve, pretrained
model import (already queued as interchange work).
