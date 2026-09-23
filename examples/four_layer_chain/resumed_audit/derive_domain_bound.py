"""Conservative scalar error accounting for the unchanged frozen exp kernel.

No model execution or data arrays. Decimal bounds are rounded upward to coarser
published bounds; actual captured-stage gates keep the original 2^-20 budget.
"""
from pathlib import Path
from decimal import Decimal,localcontext
from fractions import Fraction
import hashlib,json,math,struct,re,sys
L=Path(__file__).resolve().parent
kernel=Path(sys.argv[1])
source=kernel.read_bytes()
coefficient_literals=re.findall(r'(?:var p:f32|p)=([0-9.]+)',source.decode())
assert len(coefficient_literals)==9
f32=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]
with localcontext() as context:
 context.prec=80;D=Decimal;u=D(2)**-24;tiny=D(2)**-126;r=D('.347');ln2=D(2).ln()
 hi=D(f32(.693145751953125));lo=D(f32(.000001428606765330187));c=D(f32(math.log(2)));threshold=c/2
 assert Fraction(float(hi))==Fraction(22713,32768) and 70*22713<2**24
 selection=(70*u/(1-70*u))*(D(48)+70*c)+140*tiny
 # After at most69 negative additions, -48+69*ln2 > -threshold even
 # under this error bound. Positive inputs<=5 require at most7 steps.
 assert D(-48)+69*c-selection>-threshold and D(5)-7*c+selection<threshold
 argument=70*abs(ln2-hi-lo)+u*70*abs(lo)+u*(r+u*70*abs(lo))+4*tiny
 assert threshold+selection+70*abs(c-ln2)+argument<r
 # n=+/-1 requires abs(x)>threshold; generic upper enclosure still holds.
 assert threshold>hi/2 and c+threshold+selection<2*hi
 # For abs(n)>=2 both Sterbenz inequalities have increasing positive margins.
 assert 2*c-threshold-selection>hi and 2*c+threshold+selection<4*hi
 assert c-hi/2>0 and 2*hi-c>0
 for degree,literal in zip(range(8,-1,-1),coefficient_literals):
  actual=f32(float(literal));bits=struct.unpack('<I',struct.pack('<f',actual))[0]
  prev=struct.unpack('<f',struct.pack('<I',bits-1))[0];nxt=struct.unpack('<f',struct.pack('<I',bits+1))[0]
  assert abs(D(literal)-D(actual))<min(abs(D(literal)-D(prev)),abs(D(literal)-D(nxt)))
  assert actual==f32(1/math.factorial(degree))
 magnitude=D(1)/D(math.factorial(8));err=abs(D(f32(float(magnitude)))-magnitude)
 coefficients=[dict(degree=8,f32=float(f32(float(magnitude))))]
 for k in range(7,-1,-1):
  coefficient=D(1)/D(math.factorial(k));stored=D(f32(float(coefficient)))
  product_error=r*err+u*r*(magnitude+err)+tiny
  addition_input_error=product_error+abs(stored-coefficient)
  err=addition_input_error+u*(abs(stored)+r*magnitude+product_error)+tiny
  magnitude=coefficient+r*magnitude;coefficients.append(dict(degree=k,f32=float(stored)))
 polynomial=err/(-r).exp();remainder=(2*r).exp()*r**9/D(math.factorial(9))
 relative=polynomial+remainder+(argument.exp()-1)*(1+polynomial+remainder)
 assert selection<D('.000403') and argument<D('2.070e-8')
 assert polynomial<D('2.025e-7') and remainder<D('4.024e-10') and relative<D('2.24e-7')<D(2)**-20
 result=dict(version='chain-exp-domain-v2',old_domain=[-48,1],new_domain=[-48,5],
  kernel=dict(bytes=len(source),sha256=hashlib.sha256(source).hexdigest()),kernel_changed=False,
  actual_layer12_precheck=dict(cases=288,only_old_domain_failures=177,arithmetic_failures=0,observed_range=[-10.2734375,4.09375],report_sha256='4ca40258b54917407bcda0602902328c6b55e9eac7138dac2914d7243e8f0f8f'),
  assumptions=['Same finite FP32 arithmetic model as the accepted kernel; each multiply/add may round separately. Fused multiply-add has no larger bound.',
   'A min-normal absolute allowance is included per operation, covering any tiny intermediate flush; exponential output remains normal.',
   'Finite input within the explicit domain is required; no NaN, infinity, arbitrary overflow or arbitrary negative-exponent claim.'],
  selection=dict(maximum_updates=70,negative_iterations_at_most=69,positive_iterations_at_most=7,published_absolute_error_bound='.000403',recomputed_remainder_absolute_bound='.347'),
  reduction=dict(high_exact_ratio='22713/32768',integer_product_exact='abs(n)*22713 < 2^24 for abs(n)<=70',
   first_subtraction='For n=0 identity. For abs(n)=1, abs(x)>FP32(ln2/2)>hi/2 and abs(x)<C+B+selection_error<2hi. For abs(n)>=2, abs(x) lies between abs(n)C-B-E and abs(n)C+B+E; both factor2 margins are positive at n=2 and increase thereafter. Same sign then gives exact Sterbenz subtraction.',
   argument_absolute_error_bound='2.070e-8'),
  polynomial=dict(degree=8,coefficient_rounding='Each stored FP32 coefficient is compared with exact1/k!.',
   coefficients=coefficients,rounding_recurrence='E_product=r*E+u*r*(M+E)+tiny; E_next=E_product+coefficient_error+u*(abs(stored_coefficient)+r*M+E_product)+tiny; M_next=1/k!+r*M.',
   relative_rounding_bound='2.025e-7',relative_taylor_remainder_bound='4.024e-10'),
  scaling=dict(conservative_n=[-70,7],biased_exponent=[57,134],exact_power_of_two_scaling=True,output_normal=True,overflow=False),
  conservative_combined_relative_bound='2.24e-7',unchanged_actual_stage_relative_gate='2^-20',
  quantitative_tolerances_changed=False,device_reexecution=False,full_model_domain_claim=False,
  calculation=dict(selection=str(selection),argument=str(argument),polynomial=str(polynomial),taylor=str(remainder),combined=str(relative)))
# This reproduction prints only and does not mutate evidence.
print(json.dumps({k:result[k] for k in ['version','kernel','new_domain','conservative_combined_relative_bound','unchanged_actual_stage_relative_gate']}))
