# GPT

## GPT-1, GPT-2, and GPT-3 Comparison

| Feature                           | GPT-1                      | GPT-2              | GPT-3                       |
| --------------------------------- | -------------------------- | ------------------ | --------------------------- |
| Release Year                      | 2018                       | 2019               | 2020                        |
| Maximum Parameters                | 117M                       | 1.5B               | 175B                        |
| Maximum Number of Layers          | 12                         | 48                 | 96                          |
| Maximum Hidden Dimension          | 768                        | 1600               | 12288                       |
| Maximum Number of Attention Heads | 12                         | 25                 | 96                          |
| Context Length                    | 512                        | 1024               | 2048                        |
| Vocabulary Size                   | ~40,000                    | 50,257             | 50,257                      |
| Model Architecture                | Decoder-only               | Decoder-only       | Decoder-only                |
| Layer Normalization               | Post-LN                    | Pre-LN             | Pre-LN                      |
| Final Layer Normalization         | No                         | Yes                | Yes                         |
| Position Embedding                | Learnable                  | Learnable          | Learnable                   |
| Attention Mechanism               | Dense Causal               | Dense Causal       | Dense + Local Sparse Causal |
| Activation Function               | GELU                       | GELU               | GELU                        |
| Learning Paradigm                 | Pre-training + Fine-tuning | Zero-shot Learning | In-context Learning         |

## References

- [GPT-1: Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [Dive into Deep Learning/Decoder-Only Language Models (GPT)](https://d2l.smola.org/chapter_transformers/gpt.html)
