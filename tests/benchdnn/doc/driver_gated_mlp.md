# GatedMLP Driver

## Usage
``` sh
    ./benchdnn --gated_mlp [benchdnn-knobs] [gated_mlp-knobs] [gated_mlp-desc] ...
```

where *gated_mlp-knobs* are:

 - `--dt={f32 [default], ...}` -- source, weight, and destination data types.
            Interface supports broadcasting, when a single input is provided,
            e.g., `--dt=f32`, the value is applied for all tensors. Five
            individual values can be specified in SRC, W_GATE, W_UP, W_DOWN,
            DST order.
            Refer to [data types](knobs_dt.md) for details.
 - `--stag={abx [default], ...}` -- memory format of the source tensor.
            Refer to [tags](knobs_tag.md) for details.
 - `--wtag={abx [default], ...}` -- memory format of the weight tensors
            (applied to all three: gate, up, down).
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={abx [default], ...}` -- memory format of the destination tensor.
            Refer to [tags](knobs_tag.md) for details.
 - `--activation={swish [default], gelu_erf, gelu_tanh}` -- specifies the
            gated activation function applied after the gate matmul.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.
 - Any attributes options. Refer to [attributes](knobs_attr.md) for details.

and *gated_mlp-desc* is a problem descriptor. The canonical form is:
```
    MBxICxOC
```
Here `x` is the delimiter for the three dimensions: `MB` (batch size), `IC`
(input channels / model dimension), and `OC` (intermediate dimension). All
tensor shapes are derived from these three values:
- Source: `[MB, IC]`
- Gate weights: `[IC, OC]`
- Up weights: `[IC, OC]`
- Down weights: `[OC, IC]`
- Destination: `[MB, IC]`

## Essence of Testing

The GatedMLP operation computes
`DST = (activation(SRC * W_gate) * (SRC * W_up)) * W_down`.
The reference executes each step independently using f32 matmul and eltwise
primitives on the CPU. The driver compares the fused primitive output against
this stepwise reference.

## Examples

Run the default validation set of GatedMLP using `inputs/gated_mlp/shapes_basic`
file:
``` sh
    ./benchdnn --gated_mlp --batch=inputs/gated_mlp/shapes_basic
```

Run f16 GatedMLP with gelu_erf activation on a small shape:
``` sh
    ./benchdnn --gated_mlp --dt=f16 --activation=gelu_erf 64x128x256
```

Run GatedMLP performance benchmark on GPU with an LLM shape:
``` sh
    ./benchdnn --mode=f --gated_mlp --engine=gpu --dt=f16 \
               --activation=swish 1024x896x4864
```

More examples with different driver options can be found at
inputs/gated_mlp/test_\*.
