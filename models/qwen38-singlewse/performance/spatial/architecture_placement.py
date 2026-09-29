"""Native-bank-aware capacity floor for front-loaded architecture candidates.

The complete lowering and its original weight/auxiliary extents are checked at
each candidate area. A successful area is only a packing floor; the final map is
checked again because native contraction rounding is not assumed monotonic.
This deliberately does not reuse the legacy coordinate/route owner set.
"""
from copy import deepcopy
from spatial.layer_schedule import lower_region,select_native_shape
from spatial.pipeline_lowering import boundary_ports


def capacity_resolver(calibration):
    cache={}
    def minimum(region,config):
        key=(region['role'],tuple((m['dtype'],tuple(m['shape']))for m in region['matrices']),
             region['auxiliary_pages'],config.payload_per_pe)
        if key not in cache:
            start=region['minimum_pes']
            for area in range(start,start+1024):
                candidate=deepcopy(region);candidate['rect']=[0,0,area,1]
                try:
                    lowered=lower_region(candidate,config.payload_per_pe)
                    if region['role']in ('gate_up','down'):
                        lowered=select_native_shape(lowered,calibration,config.payload_per_pe)
                    if lowered['banks']['admitted']:
                        cache[key]=area;break
                except ValueError:
                    continue
            else:raise ValueError('No native capacity floor within the bounded 1024-PE expansion')
        return cache[key]
    return minimum


def align_stage_flow(plan):
    """Select each T partition reflection using actual entry/exit faces.

    This cheap geometric ordering is a candidate generator, not a service-time
    objective. Exact native owner routes are screened after this choice.
    """
    for index,stage in enumerate(plan['stages'][1:-1],1):
        sx,sy,sw,sh=stage['rect']
        incoming=boundary_ports(plan['stages'][index-1]['rect'],stage['rect'])
        outgoing=boundary_ports(stage['rect'],plan['stages'][index+1]['rect'])
        def center(rows,key):
            return [sum(r[key][i]for r in rows)/len(rows)for i in (0,1)]
        entry=center(incoming,'destination');exit=center(outgoing,'source')
        def transformed(rect,xflip,yflip):
            x,y,w,h=rect
            return [sx+sw-(x-sx)-w if xflip else x,sy+sh-(y-sy)-h if yflip else y,w,h]
        def distance(rect,point):
            x,y,w,h=rect;return abs(x+(w-1)/2-point[0])+abs(y+(h-1)/2-point[1])
        mix=next(r for r in stage['regions']if r['role']=='mix')
        down=next(r for r in stage['regions']if r['role']=='down')
        choices=[(distance(transformed(mix['rect'],x,y),entry)+distance(transformed(down['rect'],x,y),exit),x,y)
                 for x in (False,True)for y in (False,True)]
        _,xflip,yflip=min(choices)
        for r in stage['regions']:r['rect']=transformed(r['rect'],xflip,yflip)
        stage['flow_reflection']=[xflip,yflip]
    return plan
