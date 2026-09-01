# Annotated Bibliography 
**35 references, all metadata fetched and verified programmatically** — arXiv API for the 23 arXiv papers, Crossref/Semantic Scholar for the 12 non-arXiv classics (verification audit: `lit_report.md`, `lit_report2.md`, `lit_report3.md`; BibTeX source of truth: `../references.bib`). Each entry lists its bib key and the role it plays in the paper.

---

## 1. Hierarchical and recurrent reasoning models

The architectural thread the paper interrogates.

- **Hierarchical Reasoning Model** — Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, Yasin Abbasi Yadkori (2025). arXiv: [2506.21734](https://arxiv.org/abs/2506.21734). `wang2025hrm`
  *Role:* the hierarchy hypothesis under test — fast/slow recurrent modules with adaptive computation credited with latent multi-step planning; motivates the entire audit and the HRM/HRM-v2 arms.
- **Less is More: Recursive Reasoning with Tiny Networks** — Alexia Jolicoeur-Martineau (2025). arXiv: [2510.04871](https://arxiv.org/abs/2510.04871). `jolicoeur2025trm`
  *Role:* independent evidence that much of HRM's benchmark performance survives drastic architectural simplification — the paper's planning-side negatives corroborate from a different angle.

## 2. Classical heuristic search foundations

The integration vocabulary: admissibility, bounded suboptimality, focal bands, real-time/state-local search.

- **A Formal Basis for the Heuristic Determination of Minimum Cost Paths** — Peter Hart, Nils Nilsson, Bertram Raphael (1968). *IEEE Transactions on Systems Science and Cybernetics*. DOI: [10.1109/tssc.1968.300136](https://doi.org/10.1109/tssc.1968.300136). `hart1968astar`
  *Role:* A\* itself — the planner every learned provider plugs into.
- **Heuristic Search Viewed as Path Finding in a Graph** — Ira Pohl (1970). *Artificial Intelligence*. DOI: [10.1016/0004-3702(70)90007-X](https://doi.org/10.1016/0004-3702(70)90007-X). `pohl1970`
  *Role:* weighted A\* and the w-bound that focal variants preserve.
- **Studies in Semi-Admissible Heuristics** — Judea Pearl, Jin H. Kim (1982). *IEEE Transactions on Pattern Analysis and Machine Intelligence*. DOI: [10.1109/tpami.1982.4767270](https://doi.org/10.1109/tpami.1982.4767270). `pearl1982semiadmissible`
  *Role:* A\*ε / focal search — the admissible band inside which the discrete program's learned ranker recovers its 6–15% win.
- **Real-Time Heuristic Search** — Richard Korf (1990). *Artificial Intelligence*. DOI: [10.1016/0004-3702(90)90054-4](https://doi.org/10.1016/0004-3702(90)90054-4). `korf1990rta`
  *Role:* the state-local search tradition that C13's bounded-observation restriction descends from.
- **Real-Time Adaptive A\*** — Sven Koenig, Maxim Likhachev (2006). *Proceedings of AAMAS 2006*. DOI: [10.1145/1160633.1160682](https://doi.org/10.1145/1160633.1160682). `koenig2006rtaa`
  *Role:* RTAA\* — updating heuristics from current-state experience; the closest classical kin to C13's inference-time local Bellman backup.
- **Multi-Heuristic A\*** — Sandip Aine, Siddharth Swaminathan, Venkatraman Narayanan, Victor Hwang, Maxim Likhachev (2014). *Robotics: Science and Systems X*. DOI: [10.15607/rss.2014.x.056](https://doi.org/10.15607/rss.2014.x.056). `aine2016mha`
  *Role:* multi-heuristic scheduling — flagged as the natural home for unreliable learned signals the program did not explore.

## 3. Learned heuristics and guidance for search

The literature whose gains the paper's harness analysis sits underneath.

- **Learning Heuristic Functions for Large State Spaces** — Shahab Jabbari Arfaee, Sandra Zilles, Robert C. Holte (2011). *Artificial Intelligence*. DOI: [10.1016/j.artint.2011.08.001](https://doi.org/10.1016/j.artint.2011.08.001). `arfaee2011bootstrap`
  *Role:* bootstrapped heuristic learning — the classic supervised-heuristic lineage.
- **Learning Heuristic Search via Imitation (SaIL)** — Mohak Bhardwaj, Sanjiban Choudhury, Sebastian Scherer (2017). arXiv: [1707.03034](https://arxiv.org/abs/1707.03034). `bhardwaj2017sail`
  *Role:* imitation of clairvoyant oracles for search guidance.
- **Neural Network Heuristics for Classical Planning: A Study of Hyperparameter Space** — Patrick Ferber, Malte Helmert, Jörg Hoffmann (2020). *Frontiers in Artificial Intelligence and Applications* (ECAI 2020). DOI: [10.3233/faia200364](https://doi.org/10.3233/faia200364). `ferber2020neural`
  *Role:* supervised neural heuristics in classical planning — evidence that training configuration dominates outcomes, consonant with the harness thesis.
- **Path Planning using Neural A\* Search** — Ryo Yonetani, Tatsunori Taniai, Mohammadamin Barekatain, Mai Nishimura, Asako Kanezaki (2020). arXiv: [2009.07476](https://arxiv.org/abs/2009.07476). `yonetani2021neuralastar`
  *Role:* differentiable A\* — the integration-by-construction alternative to post-hoc provider insertion.
- **TransPath: Learning Heuristics for Grid-Based Pathfinding via Transformers** — Daniil Kirilenko, Anton Andreychuk, Aleksandr Panov, Konstantin Yakovlev (2022). arXiv: [2212.11730](https://arxiv.org/abs/2212.11730). `kirilenko2023transpath`
  *Role:* modern learned grid heuristics — the discrete program's nearest published relatives.
- **Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal** — Leah Chrestien, Tomáš Pevný, Stefan Edelkamp, Antonín Komenda (2023). arXiv: [2310.19463](https://arxiv.org/abs/2310.19463). `chrestien2023rank`
  *Role:* the training-side argument for ranking over magnitude; the paper's focal result is its integration-side mirror (a near-perfect ranker with collapsed magnitude, rescued by consuming it as a ranking).
- **Subgoal Search for Complex Reasoning Tasks** — Konrad Czechowski, Tomasz Odrzygóźdź, Marek Zbysiński, Michał Zawalski, Krzysztof Olejnik, Yuhuai Wu, Łukasz Kuciński, Piotr Miłoś (2021). arXiv: [2108.11204](https://arxiv.org/abs/2108.11204). `czechowski2021subgoal`
  *Role:* learned *planner-side* hierarchy (subgoals) — explicitly scoped out of the paper's formulation-level negative.

## 4. Differentiable planning and neural algorithmic structure

Kin to the constructive C13 mechanism (one Bellman backup at the planner interface).

- **Value Iteration Networks** — Aviv Tamar, Yi Wu, Garrett Thomas, Sergey Levine, Pieter Abbeel (2016). arXiv: [1602.02867](https://arxiv.org/abs/1602.02867). `tamar2016vin`
  *Role:* embedding value-iteration steps in a network; C13-K/M's inference-time radius-bounded backup is a single such step applied at the planner interface, and C12-B's "efficiency-not-value-iteration" reading is framed against it.
- **Gated Path Planning Networks** — Lisa Lee, Emilio Parisotto, Devendra Singh Chaplot, Eric Xing, Ruslan Salakhutdinov (2018). arXiv: [1806.06408](https://arxiv.org/abs/1806.06408). `lee2018gppn`
  *Role:* the gated-recurrence refinement of VIN; second anchor for the iterative-refiner discussion.
- **Neural Algorithmic Reasoning** — Petar Veličković, Charles Blundell (2021). arXiv: [2105.02761](https://arxiv.org/abs/2105.02761). `velickovic2021nar`
  *Role:* the broader program of executing the right algorithmic step inside learned systems — the frame for "less information + correct local computation."

## 5. Motion planning: substrate and learning

- **Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces** — Lydia Kavraki, Petr Švestka, Jean-Claude Latombe, Mark Overmars (1996). *IEEE Transactions on Robotics and Automation*. DOI: [10.1109/70.508439](https://doi.org/10.1109/70.508439). `kavraki1996prm`
  *Role:* the PRM — the substrate of the entire continuous program.
- **Cooperative Pathfinding** — David Silver (2005). *Proceedings of AIIDE*. DOI: [10.1609/aiide.v1i1.18726](https://doi.org/10.1609/aiide.v1i1.18726). `silver2005cooperative`
  *Role:* space–time A\* over (state, time) — the C8 dynamic substrate's ancestry.
- **SIPP: Safe Interval Path Planning for Dynamic Environments** — Mike Phillips, Maxim Likhachev (2011). *IEEE ICRA 2011*. DOI: [10.1109/icra.2011.5980306](https://doi.org/10.1109/icra.2011.5980306). `phillips2011sipp`
  *Role:* the other canonical dynamic-obstacle search formulation cited alongside Silver for C8.
- **Learning Sampling Distributions for Robot Motion Planning** — Brian Ichter, James Harrison, Marco Pavone (2017). arXiv: [1709.05448](https://arxiv.org/abs/1709.05448). `ichter2018sampling`
  *Role:* learning inside sampling-based planners — the continuous learned-planning tradition.
- **Motion Planning Networks (MPNet)** — Ahmed H. Qureshi, Anthony Simeonov, Mayur J. Bency, Michael C. Yip (2018). arXiv: [1806.05767](https://arxiv.org/abs/1806.05767). `qureshi2019mpnet`
  *Role:* end-to-end learned planners as the contrast class to guidance-inside-A\*.

## 6. Architectures used as experimental arms

- **Long Short-Term Memory** — Sepp Hochreiter, Jürgen Schmidhuber (1997). *Neural Computation*. DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735). `hochreiter1997lstm`
  *Role:* the LSTM baselines of the discrete forecasting program.
- **Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks** — Yikang Shen, Shawn Tan, Alessandro Sordoni, Aaron Courville (2018). arXiv: [1810.09536](https://arxiv.org/abs/1810.09536). `shen2019onlstm`
  *Role:* ON-LSTM — the structured-recurrence arm that wins C5 and the structured-dynamics forecasting protocol.
- **Adaptive Computation Time for Recurrent Neural Networks** — Alex Graves (2016). arXiv: [1603.08983](https://arxiv.org/abs/1603.08983). `graves2016act`
  *Role:* ACT — the halting mechanism whose learned behavior C11 tests (and finds anti-correlated with depth).
- **PonderNet: Learning to Ponder** — Andrea Banino, Jan Balaguer, Charles Blundell (2021). arXiv: [2107.05407](https://arxiv.org/abs/2107.05407). `banino2021pondernet`
  *Role:* the modern halting formulation cited alongside ACT for the HRM-v2 arm.
- **U-Net: Convolutional Networks for Biomedical Image Segmentation** — Olaf Ronneberger, Philipp Fischer, Thomas Brox (2015). arXiv: [1505.04597](https://arxiv.org/abs/1505.04597). `ronneberger2015unet`
  *Role:* the global-view field architecture — repeatedly the strongest learned arm (C6–C8, C11 shallow-K).
- **FiLM: Visual Reasoning with a General Conditioning Layer** — Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville (2017). arXiv: [1709.07871](https://arxiv.org/abs/1709.07871). `perez2018film`
  *Role:* the conditioning mechanism of C11's `unet_film` arm.

## 7. Adaptation and weight-space composition

- **LoRA: Low-Rank Adaptation of Large Language Models** — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021). arXiv: [2106.09685](https://arxiv.org/abs/2106.09685). `hu2021lora`
  *Role:* the low-rank adapter whose plateau C9h attributes to capacity rather than the output bound.
- **Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time** — Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, Ludwig Schmidt (2022). arXiv: [2203.05482](https://arxiv.org/abs/2203.05482). `wortsman2022soups`
  *Role:* weight averaging — the optimistic prior that C10's interpolation null qualifies.
- **Editing Models with Task Arithmetic** — Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi (2022). arXiv: [2212.04089](https://arxiv.org/abs/2212.04089). `ilharco2022task`
  *Role:* task-vector composition — the second weight-space frame for C10.

## 8. Evaluation rigor and reproducibility

- **Deep Reinforcement Learning that Matters** — Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger (2017). arXiv: [1709.06560](https://arxiv.org/abs/1709.06560). `henderson2018matters`
  *Role:* the canonical warning about seed/protocol sensitivity that the matched-solved paired methodology answers.
- **Deep Reinforcement Learning at the Edge of the Statistical Precipice** — Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare (2021). arXiv: [2108.13264](https://arxiv.org/abs/2108.13264). `agarwal2021precipice`
  *Role:* the case for interval-based aggregate reporting — the statistics framing for bootstrap CIs and the world-clustered reanalysis.
- **Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)** — Joelle Pineau, Philippe Vincent-Lamarre, Koustuv Sinha, Vincent Larivière, Alina Beygelzimer, Florence d'Alché-Buc, Emily Fox, Hugo Larochelle (2020). arXiv: [2003.12206](https://arxiv.org/abs/2003.12206). `pineau2021reproducibility`
  *Role:* the reproducibility-program precedent for the paper's preregistration/hard-stop discipline.

---

*Count: 23 arXiv + 12 Crossref/Semantic Scholar = 35 entries, matching `references.bib` one-to-one. One deliberate omission: the ARC Prize team's HRM analysis ("hidden drivers" of ARC performance) could not be verified on arXiv, so the simplification finding is cited via Jolicoeur-Martineau (2025) instead.*
