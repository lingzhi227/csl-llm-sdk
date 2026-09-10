"""Coverage and conservative arithmetic-error accounting, not a device proof."""
import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.rms_numerics import f32,U


def emulate(x):
    n=0;select=x
    while select<f32(-math.log(2)/2):select=f32(select+f32(math.log(2)));n-=1
    while select>f32(math.log(2)/2):select=f32(select-f32(math.log(2)));n+=1
    r=f32(f32(x-f32(n*0.693145751953125))-f32(n*f32(0.000001428606765330187)))
    p=f32(1/math.factorial(8))
    for k in reversed(range(8)):p=f32(f32(1/math.factorial(k))+f32(r*p))
    return math.ldexp(p,n),r,n


def arithmetic_bound():
    # Selection loop error allows R=.3467. Split-ln2 high product is exact for
    # every possible n; bound remainder arithmetic and low-part approximation.
    R=0.3467;low=math.log(2)-0.693145751953125
    reduced_error=U*R+35*(abs(f32(low)-low)+U*abs(f32(low)))+1e-12
    magnitude=1/math.factorial(8);error=abs(f32(magnitude)-magnitude)
    for k in reversed(range(8)):
        coefficient=1/math.factorial(k);ce=abs(f32(coefficient)-coefficient)
        product=(R+reduced_error)*(magnitude+error)
        error=ce+R*error+reduced_error*(magnitude+error)+U*product+U*(coefficient+ce+(1+U)*product)
        magnitude=coefficient+R*magnitude
    truncation=math.exp(R)*R**9/math.factorial(9)
    return {'remainder_error_bound':reduced_error,'polynomial_rounding_bound':error,
            'truncation_bound':truncation,'relative_bound':(error+truncation)/math.exp(-R)}


class ExpBoundedTests(unittest.TestCase):
    def test_domain_grid_extremes_and_reduction_boundaries(self):
        points=[f32(-24+25*i/1024) for i in range(1025)]
        for n in range(-35,2):
            mid=(n+0.5)*math.log(2)
            for d in (-1e-5,-1e-6,0,1e-6,1e-5):
                if -24<=mid+d<=1:points.append(f32(mid+d))
        for x in points:
            value,r,n=emulate(x)
            self.assertLessEqual(abs(r),0.3467)
            self.assertLessEqual(abs(value-math.exp(x))/math.exp(x),2**-20)
            self.assertEqual(f32(n*0.693145751953125),n*0.693145751953125)

    def test_rounding_and_truncation_budget(self):
        self.assertLess(arithmetic_bound()['relative_bound'],2**-20)


if __name__=='__main__':unittest.main()
