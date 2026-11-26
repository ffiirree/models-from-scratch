# Models from Scratch

Architecture notes and hands-on reimplementations of Transformer, GPT, ViT, CLIP, robot policies, vision-language-action models, world models, and agentic systems.

## Models

### Fundamental Architectures

- [x] [Transformer](/01_transformer), 2017.06
  
### LLMs

- [x] [GPT-1/2/3](/03_gpt), 2018.06
- [ ] PaLM
- [ ] LLaMA
- [ ] Gemma

### Vision Models

- [x] [ViT](/02_vit), 2020.10
- [ ] [DINO](/07_dino), 2021.04

### VLMs

- [x] [CLIP](/06_clip), 2021.03
- [ ] SigLIP

### Generative Models

- [ ] DDPM
- [ ] [Flow Matching](/04_flow_matching), 2022.10

### Robot Policies

- [ ] RT-1
- [ ] [Diffusion Policy:Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), 2023.03
- [ ] ACT
- [ ] RT-2
- [ ] [OpenVLA:An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246), 2024.06
- [ ] [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164v1), 2024.10
- [ ] [π0.5: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054), 2025.04

## Architecture

| Component                           | Transformer (2017)                            | GPT-2 (2019)                | PaLM (2022)                           | LLaMA 1 (2023)        | LLaMA 2 (2023)        | Gemma 1 (2024)                | Gemma 2 (2024)                                         |
| ----------------------------------- | --------------------------------------------- | --------------------------- | ------------------------------------- | --------------------- | --------------------- | ----------------------------- | ------------------------------------------------------ |
| Architecture                        | Encoder-decoder                               | Decoder-only                | Decoder-only                          | Decoder-only          | Decoder-only          | Decoder-only                  | Decoder-only                                           |
| Self-attention mask                 | Bidirectional encoder; causal decoder         | Causal                      | Causal                                | Causal                | Causal                | Causal                        | Causal                                                 |
| Cross-attention                     | Yes, in decoder                               | No                          | No                                    | No                    | No                    | No                            | No                                                     |
| Attention mechanism                 | MHA                                           | MHA                         | MQA                                   | MHA                   | MHA / GQA             | MQA / MHA                     | GQA                                                    |
| Attention scope                     | Global                                        | Global                      | Global                                | Global                | Global                | Global                        | Alternating local and global                           |
| Local window                        | None                                          | None                        | None                                  | None                  | None                  | None                          | 4,096 tokens                                           |
| Positional encoding                 | Fixed sinusoidal (absolute)                   | Learned absolute embeddings | RoPE                                  | RoPE                  | RoPE                  | RoPE                          | RoPE                                                   |
| Position injection                  | Added to token embeddings                     | Added to token embeddings   | Applied to Q and K                    | Applied to Q and K    | Applied to Q and K    | Applied to Q and K            | Applied to Q and K                                     |
| Normalization                       | LayerNorm                                     | LayerNorm                   | LayerNorm (no additive bias)          | RMSNorm               | RMSNorm               | RMSNorm                       | RMSNorm                                                |
| Normalization placement             | Post-norm                                     | Pre-norm                    | Shared pre-norm for parallel branches | Pre-norm              | Pre-norm              | Pre-norm                      | Pre- and post-sublayer norm                            |
| Final normalization                 | No separate final norm                        | LayerNorm                   | LayerNorm                             | RMSNorm               | RMSNorm               | RMSNorm                       | RMSNorm                                                |
| FFN design                          | Two-projection MLP                            | Two-projection MLP          | Gated FFN                             | Gated FFN             | Gated FFN             | Gated FFN                     | Gated FFN                                              |
| FFN activation                      | ReLU                                          | GELU                        | SwiGLU (SiLU gate)                    | SwiGLU                | SwiGLU                | GeGLU (approximate GELU gate) | GeGLU                                                  |
| FFN projections                     | 2                                             | 2                           | 3                                     | 3                     | 3                     | 3                             | 3                                                      |
| Attention / FFN topology            | Sequential                                    | Sequential                  | Parallel                              | Sequential            | Sequential            | Sequential                    | Sequential                                             |
| Attention and FFN projection biases | Yes                                           | Yes                         | No                                    | No                    | No                    | No                            | No                                                     |
| Residual connections                | Yes                                           | Yes                         | Yes                                   | Yes                   | Yes                   | Yes                           | Yes                                                    |
| Input embedding scaling             | sqrt(d_model)                                 | None                        | No explicit scaling specified here    | None                  | None                  | sqrt(d_model)                 | sqrt(d_model)                                          |
| Input/output weight tying           | Yes (shared vocabulary in original setup)     | Yes                         | Yes                                   | No                    | No                    | Yes                           | Yes                                                    |
| Attention logit soft-capping        | No                                            | No                          | No                                    | No                    | No                    | No                            | Yes (50)                                               |
| Output logit soft-capping           | No                                            | No                          | No                                    | No                    | No                    | No                            | Yes (30)                                               |
| Primary training objective          | Conditional next-token prediction (seq2seq)   | Next-token prediction       | Next-token prediction                 | Next-token prediction | Next-token prediction | Next-token prediction         | Next-token prediction; distillation for smaller models |
| Inference generation                | Autoregressive decoder conditioned on encoder | Autoregressive              | Autoregressive                        | Autoregressive        | Autoregressive        | Autoregressive                | Autoregressive                                         |
