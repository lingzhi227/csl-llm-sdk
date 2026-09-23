# Original sequential-stage preparation and static compilation

[Milestone report](../../docs/SEQUENTIAL-PREPARATION-STAGE0-COMPILE.md) ·
[Evidence summary](../../evidence/sequential-stages/qualification.json) ·
[Exact source identities](source-map.json).

The source directory contains the actual original-row reader, direct ROI
packing, independent halfword decoder, complete head geometry/preparation,
sequential context contract and bounded program-index writer used by the
accepted preparations and first-stage compilation. Every source file is
byte-identical to its indicated frozen candidate component.

These are reusable library components, not a portable launch bundle. Original
model data, accepted family plans, actual device bindings and deployment-specific
resource admissions are supplied separately. The original reader also checks
its recorded filesystem and immutable shard identity. No weights, generated
large layouts, raw neural captures, complete program index or SDK binaries are
included. The context contract describes intended full-model execution; only
the preparation and stage0 static compilation scope is accepted here.
