"""Offline actual-SIM007 regression and bounded settlement tests; no SDK imports."""
from pathlib import Path
from copy import deepcopy
import ast,json
INITIAL=[2, 0, 2, 0, 8, 1, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4194600, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 1, 2, 1, 0, 4194600, 0, 0, 0]
FINAL=[2, 0, 2, 0, 8, 1, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4194600, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 1, 2, 1, 0, 4194600, 0, 0, 0]

def main():
    source=(Path(__file__).with_name('driver.py')).read_text();tree=ast.parse(source)
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='settled' for t in n.targets)]
    assert len(assignments)==1
    expression=compile(ast.Expression(assignments[0].value),'<actual host settled predicate>','eval')
    def gate(raw):
        rows=[raw[:33],raw[33:]];return eval(expression,{'__builtins__':{}},dict(initial_snapshot=rows,rx=rows[0][:30],tx=rows[1][:30]))
    assert INITIAL[2]==INITIAL[34]==2 and INITIAL[31]==1 and FINAL[31]==2
    assert gate(INITIAL) is False and gate(FINAL) is True
    rejected=1
    for index,values in [(31,[0,1,3]),(30,[1]),(32,[1]),(63,[1]),(64,[1]),(65,[1]),(2,[0,1,3]),(34,[0,1,3])]:
        for value in values:
            bad=deepcopy(FINAL);bad[index]=value;assert gate(bad) is False;rejected+=1
    assert 'for i in range(8):' in source and 'assert copies<17' in source
    assert source.count('time.sleep(.025)')==1
    assert "if not settled:raise ValueError('No complete settled snapshot baseline within eight polls')" in source
    assert 'assert stable_snapshots(initial_snapshot,final_snapshot)==(state,tail_normal_events,consumption_events)' in source
    print(json.dumps(dict(passed=True,actual_final_positive=1,actual_initial_rejected=True,negative_rejected=rejected,SDK_invocations=0)))

if __name__=='__main__':main()
