# Kernel provenance

The WP01 local GEMV kernel is copied byte-for-byte from the prior first-party
CSL-LLM `local_gemv_bf16_f32_colmajor.csl`, also preserved as CSL handbook
reference C01. No historical result is relabeled as new-model acceptance.

Source SHA-256: `3e92175efe12b28065bb77934c54136db3c9287c68c81877781a0e527354304c`.
The module uses synchronous DSR dest/src0 bank 3 for BF16 halfword expansion,
and dest/src0/src1 bank 4 for FP32 FMA. The source weight address advances within
a call; all descriptors, expanded zeros and result zeros are reloaded each call.
WP01 reserves these leases and invokes no overlapping kernel while they are active.
One-PE CSL wrapper/host validation and model-tile range loading are new work.
