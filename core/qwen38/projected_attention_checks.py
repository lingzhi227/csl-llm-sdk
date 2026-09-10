"""Small observation-conditional sensitivity witnesses; never source intervals."""
import math
from .rms_numerics import gamma,TINY


def ignored_qk_witness(query,cache_keys):
    """The frozen ignore-QK counterexample sets the query to zero.

    Check its reported zero dots against the actual original query and cache.
    BF16 products are exactly representable in binary64; fsum rounding is
    covered by an additional binary64 term. The accepted FP32 gamma512 bound
    covers separate product/reduction operations. No source radius is changed.
    """
    rows=[]
    for key in cache_keys:
        if len(query)!=256 or len(key)!=256:raise ValueError('D256 operands required')
        products=[a*b for a,b in zip(query,key)]
        if any(not math.isfinite(x) for x in products):raise ValueError('Finite products required')
        center=math.fsum(products);magnitude=math.fsum(abs(x) for x in products)
        bound=math.nextafter((gamma(512)+2**-52)*magnitude+512*TINY,math.inf)
        rows.append(dict(expected_dot=center,allowed_absolute_error=bound,counterexample_dot=0.,
                         rejected=abs(center)>bound))
    if not rows:raise ValueError('Nonempty valid cache required')
    return dict(rejected=any(r['rejected'] for r in rows),dot_checks=rows,
                scope='Conditional on actual observed Q and cached K; complements conservative original-hidden source gate')
