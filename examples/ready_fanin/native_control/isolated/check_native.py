"""Additional actual native-consumption evidence; original protocol checker is unchanged."""
def check_native(case,counts):
    if not 0<=case<8:
        raise ValueError('Unknown native consumption case')
    expected=1 if case<4 or case==6 else 0
    if counts!=[[expected,0],[0,0]]:
        raise ValueError('Exact native control consumption count and no ordinary guard entry required')
