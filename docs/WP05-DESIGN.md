# WP05 — Full-width ordinary RMS and device BF16 conversion

The operator is `BF16_RNE((x / sqrt(mean(x²)+epsilon)) * (1+w))`, for5120 elements.
It uses the ordinary zero-centered gain from the [audited semantics](WP04-SEMANTICS.md),
not DeltaNet's direct-gain convention. Epsilon is1e-6 rounded to FP32. The fixture
inputs and gains are synthetic finite exactly representable BF16 values or zero;
FP32 staging is a lossless expansion of those inputs. No checkpoint download is
required. Overflow, unrestricted NaN payloads and subnormal support are not claimed.

## Storage lifecycle

One PE has a5120-element FP32 staging vector and a5120-element BF16 gain/output
vector, each with endpoint guards. The payload is30 KiB. Scalar diagnostics,
state, timestamp words and a16-value conversion probe are additional small buffers.

1. The host uploads all original input values and all BF16 gain bits for each call.
2. The PE computes the complete squared sum before modifying staging.
3. For each element, the PE loads its gain, computes the FP32 normalized value and
   offset-gain product, overwrites staging with that pre-cast result, and replaces
   the consumed gain slot with BF16 output.
4. The host reads both complete result vectors. No later consumer may use the
   overwritten gain as gain; the next call uploads the entire gain vector again.

Thus both device buffers have declared destructive lifecycles. The frozen host
input/gain fixtures remain immutable. This design preserves simultaneous
observation of every FP32 pre-cast value and every BF16 output without allocating
a third full vector. It uses the existing native16 transport; no packed ABI or
host interstage scheduling is introduced.

## Independent acceptance gates

Let u=2^-24, `gamma(n)=n*u/(1-n*u)`, and m=2^-126. A direct FP64 sum of input
squares gives s. The declared sum bound is `gamma(10240)*s+10240*m`, allowing
separate FP32 multiply/add or an equivalent fused reduction. For q=s/5120+eps,
propagate that bound through division and addition with a two-operation gamma
term and an absolute floor. These are fixed before execution.

The actual device sqrt is independently compared with sqrt(actual q), and its
actual inverse with1/(actual sqrt); each relative gate is2^-20. These are required
acceptance tests, not assumed vendor accuracy guarantees. The pre-cast vector
bound propagates the q interval, both approximation budgets, FP32 offset-gain
addition and two FP32 multiplies. An additional gain-application gate uses the
actual inverse to distinguish downstream multiplication from upstream root error.
Every zero input or w=-1 element must produce exact numerical zero.

BF16 conversion is checked against each **actual pre-cast FP32 bit pattern**.
The device uses the integer bias/retained-bit algorithm; the independent host
oracle classifies the discarded halfword as below/above/exactly at midpoint,
selecting an even retained mantissa on ties. Signed zero is preserved. A separate
nearest-BF16 calculation directly compares distances from the FP64 mathematical
oracle, avoiding an intermediate FP32 double-rounding artifact. Its parity count
is reported separately: allowed upstream FP32 error can cross a BF16 midpoint
without making the device conversion incorrect.

The same device conversion function also processes16 frozen raw FP32 probes:
positive/negative ties with even and odd retained mantissas, adjacent values on
both sides, exponent carries, exact values and signed zero. Probe scope is stated
separately from normal RMS outputs. The fourth full-vector fixture must contain
outputs that distinguish gain-before-cast from casting normalized values first.
This catches applying the correct gain to the wrong precision boundary.

Four calls share one runtime: zero, varying unit gain, signed nonuniform offset
gain including-1, and rounding-order adversaries. All5120 values, stage scalars,
guards, generations, call/probe counts and stable handles are checked every call.
Timestamp words surround the RMS kernel only; they do not measure host transfers
or physical-wafer performance.

## SRAM and host resource admission

The application SRAM ceiling is49152 bytes (48 KiB). Before simulation, the
compiler's actual application-section highest ending address plus a4096-byte
stack allowance must fit that ceiling. High device-configuration addresses do
not increase application SRAM. The allowance is a conservative declared budget,
not a measured dynamic stack peak. Tests reject one-byte overflow, a larger
ceiling, a smaller stack allowance and high-address application text disguised
by an address-only classifier.

The first candidate used an incorrect61440-byte admission ceiling and a separate
output buffer. It was controller-interrupted after compilation and simulation
startup, and cannot be accepted as a pass. The approved repair shares gain/output
storage and corrects the SRAM gate; the original failure and precise source diff
remain preserved. Arithmetic, fixtures, numerical thresholds and stack allowance
are unchanged.

The repaired candidate retains a300-second compile and180-second simulator
deadline, one heavy job, MemoryMax20 GiB, zero task swap,8 GiB available RAM
reserve,20 GiB cache cap and32 GiB disk reserve. Compile failure or SRAM refusal
prevents simulation. No unchanged retry or automatic deadline increase is allowed.
