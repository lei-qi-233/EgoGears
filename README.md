# EgoGears

**EgoGears** is a first-person outdoor video-understanding benchmark built from human-collected recordings across multiple routes, covering three movement states—walking, jogging, and running—and two lighting conditions—daytime and nighttime. Its question-answer pairs evaluate both single-video understanding and cross-video reasoning, measuring local visual, spatial, motion, and temporal evidence as well as whether observations can be aligned and composed across independent encounters. The benchmark diagnoses how changes in viewpoint, movement speed, illumination, and route phase affect the transfer of scene and route understanding between observations.

> **Repository status:** This repository is still being organized. File names and contents will continue to be refined, and more detailed documentation will be added in future updates.

> **Repository contents:** This GitHub repository contains the code and intermediate files used to build and evaluate the EgoGears dataset. For the final released dataset, evaluation results, and statistical information, please visit the EgoGears Hugging Face page.

---

### 🚀 Quick Links

<p align="center">
	<a href="YOUR_ARXIV_LINK">
		<img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv Paper"/>
	</a>
	<a href="https://huggingface.co/datasets/lei-qi-233/EgoGears">
		<img src="https://img.shields.io/badge/Hugging%20Face-Dataset-orange?logo=huggingface" alt="Hugging Face Dataset"/>
	</a>
</p>

---

Unlike evaluations that focus only on aggregate video-level accuracy, EgoGears distinguishes among the following capabilities:

- extracting reliable visual, spatial, and motion evidence from a single video;
- binding observations to the correct video, place, time, and route phase;
- establishing correspondence across encounters and composing evidence into a consistent understanding of routes.

## Dataset Overview

EgoGears is built from human-collected first-person outdoor videos covering repeated traversals of real routes, enabling models to compare observations within shared physical environments.

| Item | Count |
| --- | ---: |
| Human-collected first-person recordings | 126 |
| Outdoor routes | 39 |
| Total video duration | 22.6 hours |
| Single-video questions | 567 |
| Multi-video questions | 1,487 |
| Questions requiring alignment across independent recordings | 531 |

The recordings cover three movement states—walking, jogging, and running—and repeated route traversals under two lighting conditions: daytime and nighttime. Models must handle changes in viewpoint, movement speed, and illumination, as well as reversals in landmark order between outbound and return journeys.

## Key Features

### 1. Joint Single- and Multi-Video Evaluation

The single-video split measures foundational capabilities available from local observations, including:

- recognizing objects and their attributes;
- understanding egocentric spatial relations;
- understanding self-motion and motion trajectories;
- identifying temporal events and order.

The multi-video split further requires models to compare independent recordings, determine which observations correspond to the same place or route phase, and compose evidence distributed across videos.

This design distinguishes between failures to understand local content and failures to use understood content correctly across encounters.

### 2. Cross-Video Correspondence in Real Outdoor Experiences

EgoGears does not merely compare visually similar video segments. It also requires models to understand their relationships within a route, for example:

- identifying corresponding places across recordings;
- determining travel direction and route phase;
- aligning shared paths and localizing route divergence;
- inferring changes in landmark order between outbound and return journeys;
- correctly associating current observations with historical observations.

Models must therefore preserve observation identity, spatial context, and temporal order rather than relying only on isolated object recognition.

### 3. Diagnosing Evidence Binding

The question design makes it central to determine which video, place, and moment an observation belongs to. Each question typically combines:

- an observation target;
- a temporal or route-phase condition;
- a spatial, motion, or route relation to infer.

Distractors introduce structured perturbations to visually supported evidence, including reversed directions, altered temporal order, route-phase mismatches, and incorrect attribution of landmarks or events. Models must determine not only whether a fact appears, but also where and when it appears and which observation it belongs to.

### 4. Evaluating Complex Route Reasoning

Multi-video questions cover input structures ranging from local evidence integration to cross-recording route correspondence. They require models to handle non-contiguous observations, route-state changes, and ordered landmark relations. Some multiple-choice questions have a variable number of correct answers and are scored against the complete answer set, preventing partial guesses from receiving disproportionately high scores.

### 5. An Evaluation Protocol Grounded in Visual Evidence

To reduce dataset shortcuts and evaluation noise, EgoGears uses several design measures:

- deterministic option shuffling;
- metadata anonymization;
- screening for text-only shortcuts;
- video-grounded answer verification;
- human review;
- exact-set scoring for multiple-choice questions.

These measures help ensure that benchmark results more directly reflect models' ability to understand and compose video evidence.

## Evaluation Targets

EgoGears organizes its evaluation around three related targets:

1. **Local Understanding**: Can a model correctly recover visual, spatial, motion, and temporal information from a single video?
2. **Evidence Integration**: Can a model combine non-contiguous and distributed video evidence into a consistent conclusion?
3. **Cross-Recording Correspondence**: Can a model identify corresponding places, route phases, and travel relationships across recordings?

## Main Findings

On the main leaderboard, we evaluated 29 single-video and 31 multi-video configurations from six model families. The results show that:

- across the 20 configurations evaluated comparably, every model performed worse on multi-video questions than on single-video questions;
- the mean decrease on multi-video evaluation was **22.5 percentage points**;
- the gap persisted when answer formats and scoring methods were held fixed;
- the performance decrease cannot be explained simply by the increased number of videos or recording boundaries;
- the main bottlenecks were **observation-evidence binding** and **ordered route-state tracking**;
- across eight matched single-video comparisons, explicit reasoning improved every configuration, with a median gain of **7.8 percentage points**;
- explicit reasoning helped egocentric spatial relations (**14.0 percentage points**) substantially more than trajectory understanding (**3.7 percentage points**).

These results show that verifying local evidence is not equivalent to understanding movement and route structure. EgoGears therefore serves as a diagnostic tool for studying why local video understanding does not reliably transfer to reasoning across encounters.

## Applications

EgoGears can be used to study and evaluate:

- cross-video understanding in multimodal large language models (MLLMs);
- embodied intelligence and wearable assistants for outdoor environments;
- understanding and reuse of human route demonstrations;
- recognition of previously visited places and route memory;
- visual evidence binding, spatiotemporal correspondence, and trajectory reasoning;
- transfer from single-video capabilities to multi-encounter world modeling.

## Contributions

- formulate cross-video outdoor understanding around evidence binding, cross-recording correspondence, and spatiotemporal composition;
- introduce the EgoGears benchmark with 567 single-video questions and 1,487 multi-video questions;
- connect observations under different movement speeds and lighting conditions through repeated outdoor routes;
- diagnose specific model failures using structured distractors and exact scoring;
- systematically report performance differences between single-video and multi-video evaluation across multiple MLLM configurations.

## Contributors

The EgoGears benchmark dataset and this GitHub repository are established and maintained by:

* Yuedong Tan: Co-first author
* [Lei Qi](https://github.com/LEI-QI-233): Co-first author
* [Kunyu Peng](https://cvhci.iar.kit.edu/people_2123.php): Research supervisor


## License

This project is released under the [MIT License](LICENSE).

## Citation

If you find this dataset useful in your research, please use the following BibTeX entry for citation: