"""Runtime coordinates let identical CSL programs serve multiple original layers."""
import numpy as np
import resident_geometry as geo

def iter_identity():
    for start in range(0,geo.HEIGHT,32):
        height=min(32,geo.HEIGHT-start);data=np.zeros((height,geo.WIDTH,4),np.uint32)
        for dy in range(height):
            y=start+dy
            for x in range(geo.WIDTH):
                role=geo.role_at(x,y)
                data[dy,x]=[role.layer if role.layer>=0 else 65535,y,
                            role.row if role.row>=0 else 65535,role.expert if role.expert>=0 else 65535]
        assert data.nbytes<=1<<20
        yield start,height,data
