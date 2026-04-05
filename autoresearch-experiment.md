# autoresearch — experimento M5

**Origen:** Karpathy publicó el 8 marzo 2026 [`autoresearch`](https://github.com/karpathy/autoresearch), un framework de 630 líneas que deja a un agente iterar experimentos de entrenamiento ML en bucle sobre un modelo pequeño, sin intervención humana. Single GPU, PyTorch/CUDA.

## Idea del experimento

Replicar el concepto en el MacBook M5, sin CUDA, con lo que ya tenemos:

```
Pi (orquestador ligero — scheduler, métricas, bucle)
    ↓
M5 → Qwen3.5:9b (agente evaluador — ya corriendo para DevonThink)
         ↓ propone modificaciones al config/LoRA
     Qwen3.5-2B-Base (modelo objetivo — se re-fine-tunea)
         ↓ mlx-lm LoRA ~5 min por iteración
     métricas BPB → el 9B decide si guarda o descarta
```

## Stack

| Componente | Herramienta |
|---|---|
| Fine-tuning | `mlx-lm` (Apple Silicon nativo, sin CUDA) |
| Modelo agente | Qwen3.5:9b vía Ollama (ya instalado) |
| Modelo objetivo | Qwen3.5-2B-Base |
| Orquestación | Script Python ligero en Pi |
| Métrica | bits-per-byte (BPB) como en Karpathy |

## Por qué funciona

- El 9B ya está caliente en el M5 → no es peso extra
- MLX está optimizado para Apple Silicon → iteraciones rápidas
- Pi hace de scheduler sin consumir GPU del M5
- Los modelos Qwen3.5 mini caben con margen

## Diferencia vs Karpathy original

Karpathy usa PyTorch + CUDA. Aquí: MLX + MPS. El repo original no corre out-of-the-box en M5 — necesita adaptación o reescritura del loop de entrenamiento con `mlx-lm`.

## Próximos pasos

- [ ] Clonar autoresearch y hacer fork
- [ ] Portar el training loop a `mlx-lm` (LoRA fine-tune en lugar de full training)
- [ ] Definir dataset objetivo pequeño y métrica BPB
- [ ] Script de orquestación minimalista para Pi
- [ ] Primera ejecución nocturna de prueba

## Referencias

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [mlx-lm fine-tuning](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [Qwen3 model family](https://huggingface.co/Qwen)
