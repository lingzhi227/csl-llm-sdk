"""Native-bank-aware capacity floor for front-loaded architecture candidates.

The complete lowering and its original weight/auxiliary extents are checked at
each candidate area. A successful area is only a packing floor; the final map is
checked again because native contraction rounding is not assumed monotonic.
This deliberately does not reuse the legacy coordinate/route owner set.
"""
from copy import deepcopy
from spatial.layer_schedule import lower_region,select_native_shape


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
