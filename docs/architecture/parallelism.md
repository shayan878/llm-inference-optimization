# Parallelism strategies for LLM inference

Parallelism is a placement and workload-distribution strategy. The right choice depends on model size, GPU count, interconnect topology, memory capacity, and whether the objective is to fit a model or serve more independent traffic.

## Tensor parallelism (TP)

Tensor parallelism splits computations or weight matrices within a transformer layer across multiple GPUs. It is commonly used when a model does not fit on one GPU but can fit across GPUs in one node. It introduces communication between participating devices, so adding GPUs is not guaranteed to improve latency for small models.

In vLLM, configure the number of GPUs with `--tensor-parallel-size` (or `-tp`). Example for four GPUs:

```bash
vllm serve /model --tensor-parallel-size 4
```

## Pipeline parallelism (PP)

Pipeline parallelism distributes consecutive groups of layers across devices. Activations move between stages; pipeline schedules may introduce bubbles when stages are idle. It is useful when a model must span devices/nodes or when tensor parallelism is a poor fit for the hardware topology.

Current vLLM documentation supports pipeline parallel inference via `--pipeline-parallel-size` (or `-pp`) and describes combining TP and PP for multi-node deployments. The old internship note stating that vLLM did not support pipeline parallelism is outdated.

Example shape (check syntax and compatibility for the selected vLLM release):

```bash
vllm serve /model \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2
```

## Data parallelism (DP)

Data parallel inference runs multiple model replicas. Each replica handles a separate share of requests. This can increase aggregate serving capacity when the model fits per replica; it does not reduce the memory required for each complete replica. In training, data parallelism also requires gradient synchronisation, which is a different workload from inference.

## Expert parallelism (EP)

Mixture-of-Experts (MoE) models contain multiple expert feed-forward networks and a router. Expert parallelism distributes experts across devices so that selected experts process routed tokens. Communication patterns, routing imbalance, all-to-all traffic, and expert load balance affect performance. Support and best practices depend on the specific model and vLLM version.

## Interconnects: InfiniBand and Ethernet

InfiniBand is common in large HPC clusters because of its low-latency, high-bandwidth fabric and RDMA support. Ethernet is broadly deployed, flexible, and can also support RDMA using RoCE in suitable configurations. The comparison is not simply “InfiniBand is always faster”: NICs, switch topology, congestion control, RDMA configuration, software stack, and collective communication patterns all matter.

For multi-node LLM inference, measure the actual communication workload and validate the network fabric rather than relying on a generic bandwidth headline.

## Ray

Ray is a distributed execution framework with APIs for tasks, actors, data, training, tuning, and serving. It can coordinate larger application workflows and clusters. It is not a replacement for understanding vLLM's own model-parallel and request-scheduling configuration; choose it when the overall application needs distributed orchestration.

## Decision guide

| Problem | Candidate strategy | Main trade-off |
|---|---|---|
| Model fits on one GPU | Single-GPU serving | Simplest topology; limited per-replica capacity |
| Model too large for one GPU, fits within one node | Tensor parallelism | More memory per model replica; collective communication overhead |
| Model must span nodes or uneven layer placement helps | Pipeline parallelism, often combined with TP | Stage scheduling and activation-transfer overhead |
| Model fits on each GPU and more request capacity is needed | Data parallel replicas | Duplicated model memory; aggregate throughput may improve |
| MoE experts need distributed placement | Expert parallelism | Routing/all-to-all complexity and load balance |

References: [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) and [vLLM optimisation guide](https://docs.vllm.ai/en/latest/configuration/optimization/).
