p = "/workspace/dub/keysync/ks_fast.py"; s = open(p).read()
old = '''            if ig.startswith("trunc:"): return "full" if gs["n"] < int(ig.split(":")[1]) else "cond"
'''
new = '''            if ig.startswith("trunc:"): return "full" if gs["n"] < int(ig.split(":")[1]) else "cond"
            if ig.startswith("mix:"): return "full" if gs["n"] < int(ig.split(":")[1]) else "2way"   # 3-way first K, then ref + audio rows
            if ig == "ucc": return "full" if gs["n"] % 2 == 0 else "ucc"   # the uncond row every other step, its (e_ref - e_uc) reused
'''
assert old in s; s = s.replace(old, new, 1)
old = '''        def call(x, sig):
            if gs["mode"] == "full": return call0(x, sig)
            if gs["mode"] == "cond": return x
'''
new = '''        def call(x, sig):
            if gs["mode"] == "full":
                if ig == "ucc": e_uc, e_ref, _ = x.chunk(3); gs["d"] = e_ref - e_uc
                return call0(x, sig)
            if gs["mode"] == "cond": return x
            if gs["mode"] == "ucc":
                e_ref, c_ar = x.chunk(2); return call0(torch.cat([e_ref - gs["d"], e_ref, c_ar]), sig)
'''
assert old in s; s = s.replace(old, new, 1)
s = s.replace('''            def keep(v, n):   # rows: [uncond | ref | audio+ref], each third n long
                return v[2 * n:3 * n] if gs["mode"] == "cond" else v[n:3 * n]''', '''            def keep(v, n):   # rows: [uncond | ref | audio+ref], each third n long; 2way / ucc keep ref + audio
                return v[2 * n:3 * n] if gs["mode"] == "cond" else v[n:3 * n]''', 1)
open(p, "w").write(s); print("ok")
