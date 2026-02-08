# Adversarial Attacks in End to End Neural OCR Systems

A repository for a short technical project on adversarial attacks in OCR systems.

A case study on **Poisoning Language-Model Training Data at Collection Time via Adversarial Perturbations in DeepSeek OCR**.

## Motivation

Language models have achieved strong performance across a wide range of natural language tasks. Much of this progress is attributable to the scale and diversity of the training data used across pretraining and post-training phases. To further improve performance in specific domains or low-resource languages, auxiliary data-collection pipelines are increasingly used, including Optical Character Recognition (OCR) systems that extract textual content from rich visual sources such as scanned documents, books, and images.

Recent vision–language models (VLMs) have demonstrated OCR capabilities that rival or exceed those of traditional OCR systems, motivating their use in large-scale training data generation. One such model is [DeepSeek-OCR](https://deepseek-ocr.io/), which is positioned as a production-grade tool for high-throughput text extraction and data compression. Beyond accuracy, DeepSeek-OCR emphasizes aggressive visual-to-textual compression, enabling efficient storage and downstream language-model training.

However, as an end-to-end neural vision system, OCR models inherit well-known vulnerabilities from the vision domain. In particular, adversarial perturbations to input images can induce systematic and targeted errors in extracted text. When such OCR outputs are ingested directly into language-model training corpora, these perturbations introduce a new and underexplored attack surface: poisoning training data at collection time, prior to any textual filtering or deduplication.

## Threat Model

An adversary is considered to operate at the data collection stage of a language-model training pipeline that relies on OCR-based text extraction from visual sources. The adversary is characterized by the following capabilities and limitations:

- **Partial control over training data sources.**  
  It is assumed that poisoned content can be injected into a subset of training data collected from uncontrolled or weakly curated sources, such as public document repositories or web-scale image corpora. This reflects realistic conditions in modern large-scale data collection pipelines, where provenance and ownership of individual samples are often unclear. Compromise of the entire dataset is not required; passive data injection without breaching infrastructure is sufficient.

- **Knowledge of the OCR model.**  
  Knowledge of the OCR system used to extract textual content from images is assumed. This is reasonable given the small number of end-to-end, production-grade OCR models currently deployed at scale and the common practice of publicly documenting such pipelines. No knowledge is assumed regarding downstream language models, filtering heuristics, or post-processing steps applied after OCR, nor of the specific models ultimately trained on the extracted text.

- **Ability to craft stealthy adversarial images.**  
  Visually plausible images can be generated that evade standard sanity checks, including human inspection and automated filtering based on vision–language models or heuristic quality controls. Malicious or backdoor-triggering textual content may be embedded into full documents in a manner that survives OCR extraction and cannot be reliably removed through downstream textual filtering or post-training data curation.

Under this threat model, the primary attack surface consists of injecting adversarially perturbed images into uncontrolled data sources that are later ingested by OCR-based data collection pipelines. A secondary attack vector involves publishing seemingly benign public datasets containing carefully crafted visual backdoors that activate only upon OCR processing.

Not all agents operating under this model must be malicious. A non-adversarial but strategic actor could inject robust visual watermarks into document images that reliably elicit identifiable textual patterns under OCR. Such mechanisms could serve defensive purposes, including intellectual property attribution or dataset provenance tracking.

## Adversarial Attacks

### Simple Attack Based on Patch-Based Methods

The most straightforward method to force an OCR model to produce attacker-intended text is to modify the original image using simple patch-based techniques. This approach functions as expected, as illustrated in Fig. 1. Further discussion of this attack class is limited in value unless stealth can be achieved against neural or human inspection (for example, `rn` appearing visually similar to `m`).

### Attack Success Rates with PGD-Based Methods

Attack success rates were evaluated using a projected gradient descent (PGD)–based optimization procedure with teacher forcing, designed to induce specific target outputs from the OCR model. The optimization objective directly maximizes the likelihood of attacker-specified textual content under the OCR decoder.

Two classes of adversarial image perturbations were considered:  
(i) **Full-image perturbation**, and  
(ii) **Targeted image perturbation**.  

In the former, adversarial noise is applied to all pixels in the image. In the latter, perturbations are restricted to specific spatial regions of the document image, with the goal of injecting targeted text into localized portions of the OCR output, often interleaved with the document’s original textual content.

For all experiments in this section, the DeepSeek-OCR paper itself was used as the evaluation corpus. The document consists of 22 pages containing a diverse mix of tables, plots, diagrams, and dense text, making it a representative and challenging target for OCR-based attacks.

#### Full-Image Perturbation

Baseline OCR outputs were first extracted for the full document corpus. Two adversarial objectives were then constructed using a language model:

1. Injection of a specific backdoor trigger at an arbitrary location in the extracted text.  
2. Induction of OCR outputs containing multiple instances of predefined **CONTENTHARM** targets.  

An illustration of these objectives is provided in Figure 1.

| Objective              | OR     | ASR    |
|------------------------|--------|--------|
| ContentHarm Objective  | 0.9545 | 0.2273 |
| Backdoor Objective     | 1.0000 | 0.1364 |

**Table 1.** Optimization Rate (OR) indicates the proportion of images for which adversarial perturbation reduced the overall loss by more than 10% of the original value. Attack Success Rate (ASR) indicates the fraction of images (out of 22) for which the attacker achieved the objective. All experiments were capped at 100 iterations and run with an ℓ∞ constraint of ϵ = 0.06. Higher success rates are expected with additional iterations, though this requires further investigation.

Using PGD with an ℓ∞ constraint of ϵ = 0.06, all pages could be perturbed to reduce loss substantially within 100 optimization steps. Table 1 summarizes the attack success rates. While loss exhibited a consistent downward trend, the most effective attacks remained closely aligned with prior work (Carlini et al., 2023), where the prefix of the output is directly optimized.

#### Targeted Image Perturbation

Targeted perturbations applied to localized regions of the document were next evaluated. Three target types were considered, illustrated in Figure 2:

- Tabular regions  
- Graphical elements  
- Whitespace areas  

The underlying hypothesis was that spatially localized perturbations, combined with an output objective inserting adversarial text between expected tokens, would enable controlled injection without globally corrupting OCR output. Contrary to this expectation, targeted attacks proved substantially less effective.

Across all three target types and for ϵ ∈ {0.01, 0.03, 0.06}, meaningful reductions in optimization loss were not achieved within 100 PGD iterations. In most cases, loss values exhibited minimal or no downward trend. This outcome is notable given that perturbations were applied in the native input resolution mode (the base setting), rather than the *Gundam* mode, which involves tiling.

These results suggest that DeepSeek-OCR relies on globally contextual visual representations rather than strictly localized feature extraction, rendering naive targeted perturbations ineffective.

**Figure 1.** Illustration of a native attack. The change made to the original image in the top row and the corresponding outputs in the bottom; arranged as left (before change) and right (after change).
![naive-attack-objective](figs/naive-attack-objectives.png)

**Figure 2.** Illustration of full-image attack objectives. On the left, a backdoor trigger is injected; on the right, the image is optimized to induce OCR output containing CONTENTHARM statements.
![simple-attack-objective](figs/simple-attack-objectives.png)

**Figure 3.** Illustration of targeted attack objectives exploiting non-textual regions. Top-left: perturbing graphical elements. Top-right: perturbing whitespace regions between text. Bottom-middle: perturbing tabular data to alter structured numerical outputs.
![targetted-attack-objective](figs/targetted-attack-objectives.png)

## Discussion

*In progress.*

## Citation

If this work is used in any other project, please cite:

```
@misc{mishra_adversarial_ocr_2026,
    title = {Adversarial Attacks in OCR Systems: Poisoning Language-Model Training Data at Collection Time},
    author = {Abhishek Mishra},
    year = {2026},
    howpublished = {\url{https://github.com/abhishek9909/adversarial-attacks-ocr}},
    note = {Technical project and case study on adversarial perturbations in DeepSeek-OCR}
}
```

## Colabs

The entirety of the work was done on GoogleColab in A100-40GB.
- [Naive Attacks](https://colab.research.google.com/drive/1Cx1AM7859AlGshkA4nq1jaLsMsRBZhBe?usp=sharing)
- [Simple Attacks](https://colab.research.google.com/drive/1SBveH5gCYGwS0IUbGC5XN_VdNNx2O1lP?usp=sharing)