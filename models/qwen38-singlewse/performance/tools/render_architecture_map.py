"""Draw the selected PE-region map from exact JSON coordinates, without jobs."""
import argparse,hashlib,json
from pathlib import Path
from xml.sax.saxutils import escape


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh map path required')
    raw=a.plan.read_bytes();plan=json.loads(raw)
    colors=dict(embedding='#c4b5fd',head='#c4b5fd',mix='#93c5fd',gate_up='#fcd34d',down='#86efac')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 790 1280" width="790" height="1280">',
         '<rect width="790" height="1280" fill="#f8fafc"/>',
         '<style>text{font-family:system-ui,sans-serif;fill:#0f172a} .layer{font-size:11px;font-weight:700;paint-order:stroke;stroke:#fff;stroke-width:2px;stroke-linejoin:round}</style>',
         '<text x="20" y="28" font-size="20" font-weight="700">Selected 64-layer dialogue layout</text>',
         '<text x="20" y="48" font-size="12">One conversation · 512-position state budget · 750 × 1160 application PEs</text>',
         '<text x="20" y="66" font-size="12">Planning only: complete routes, compiled SRAM, runtime and TPS are unqualified.</text>',
         '<g transform="translate(20,88)">']
    for s in plan['stages']:
        for r in s['regions']:
            x,y,w,h=r['rect'];label=escape(f"{r['id']} [{x},{y},{w},{h}]")
            svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{colors[r["role"]]}" stroke="#64748b" stroke-width=".35"><title>{label}</title></rect>')
        x,y,w,h=s['rect'];svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#0f172a" stroke-width="1.2"/>')
        if s['layer']is None:
            svg.append(f'<text x="{x+w/2}" y="{y+h/2}" text-anchor="middle" font-size="15" font-weight="700" transform="rotate(-90 {x+w/2} {y+h/2})">{escape(s["id"])} · resident weights</text>')
        else:
            kind='ATTN'if s['layer']%4==3 else'GDN';c=s['planned_controller']
            svg.append(f'<text class="layer" x="{x+4}" y="{y+15}">L{s["layer"]} {kind}</text>')
            cx,cy=c['pe'];svg.append(f'<circle cx="{cx+.5}" cy="{cy+.5}" r="1.5" fill="#0f172a"><title>Controller {c["mode"]}: {c["pe"]}</title></circle>')
    svg+=['<rect x="0" y="1158" width="750" height="2" fill="#ef4444"/>','</g>',
          '<text x="20" y="1267" font-size="11">Blue: mixer/state · Gold: gate/up/fusion · Green: down/residual · Red: feedback · Dots: planned controllers</text>',
          '<metadata>source-plan-sha256:'+hashlib.sha256(raw).hexdigest()+'</metadata>','</svg>']
    a.output.open('x').write('\n'.join(svg)+'\n')


if __name__=='__main__':main()
