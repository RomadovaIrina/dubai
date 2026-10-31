import json,glob,sys,os
d=sys.argv[1]
for f in sorted(glob.glob(d+"/*.json")):
    try: r=json.load(open(f))
    except Exception: continue
    fl=r.get("flicker") or {}; cs=r.get("chin_shade") or {}; cf=r.get("chin_flow") or {}
    g=lambda x,k,fmt="%.2f": (fmt % x[k]) if x.get(k) is not None else "-"
    print(os.path.basename(f)[:-5].ljust(14), "lipHF",g(fl,"ratio"),"lipBr",g(fl,"bright_ratio"),"ring",g(fl,"ring_bright_ratio"),"ringS",fl.get("ring_bright_strong","-"),"| chinBr",g(cs,"bright_ratio"),"chinTex",g(cs,"tex_ratio"),"chinBrS",cs.get("bright_strong","-"),"shadeHP",g(cs,"shade_hp_rms"),"step",g(cs,"shade_step_rms"),"| vel",g(cf,"vel_rms","%.4f"),"lat",g(cf,"vel_lat_rms","%.4f"),"swim",g(cf,"swim_rms","%.4f"),"ev",cf.get("events","-"),"S",cf.get("strong","-"))
