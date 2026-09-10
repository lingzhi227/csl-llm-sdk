# WP11 report — accepted candidate001

Scope is one original BF16 Q/K pair, ordinary RMS256 with offset weights and
partial RoPE64 for integer text positions0–7. Projection, GQA, attention composition,
large-position range reduction and full-model inference are outside this result.

CPU reference001 failed during class loading before arithmetic because the framework
class integration decorator was still present. It ran2.10031014seconds with
206983168peak bytes,13unchanged source files. CPU reference002 added only the
three-line class-decorator extraction hook; the original math and fixtures did not
change. Static default-frequency method relocation preserves explicit config binding.

CPU002 passed10calls/2heads/5120BF16 outputs with zero nominal official differences.
Every tail is bit-exact; position0 identity holds numerically; separate rotary
products and addition are each BF16 in the pinned source. Trig source28027point
qualification passed the fixed2^-20gate. Runtime4.16152761seconds/224657408peak bytes;
13source and4output hashes preserved. Both CPU owned units are absent and quiet.
The device bundle adds only actual-stage validation to the unchanged source oracle.

One-PE candidate001 compiled with actual ordinary section end20080. Including4096
stack allowance gives24176, below49152. Simulation completed in55.51667428seconds with192573440peak bytes. Compilation
used4.15130861seconds and419364864peak bytes. All ten calls passed full conditional
stage/source output/immutable input-weight-frequency-probe/guard/state/event/handle
checks, including all5120official/nominal BF16 outputs with zero differences.
Tail bit equality, position0 numeric identity, actual product/add casts and zero
signs, zero-angle exact sin0/cos1, and reset were separately validated.

Worst actual-angle sine/cosine errors were4.58872e-8/3.56175e-8; fixed device probes
were4.50138e-8/2.08105e-8. The original2^-20gate was unchanged. Exactly185physical
copies,49054hostu32slots,134604native bytes and12launches completed with370durable
entry/exit records. Normal stop and postrun identity checks passed:28source and
11compiled files unchanged. Both SDK owned units are absent/inactive, MainPID0
and empty cgroups. No second candidate or additional SDK replay was needed.
The controller independently audited and accepted this bounded result. The public
snapshot passes54host tests; it preserves the initial CPU loading failure and its
arithmetic-preserving extraction repair.
