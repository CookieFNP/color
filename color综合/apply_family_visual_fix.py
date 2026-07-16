from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
XYZ_TO_SRGB = np.array([[3.2404542,-1.5371385,-0.4985314],[-0.9692660,1.8760108,0.0415560],[0.0556434,-0.2040259,1.0572252]], dtype=np.float64)

def lab_to_srgb(lab):
    L,a,b = [float(x) for x in lab]
    fy=(L+16)/116; fx=fy+a/500; fz=fy-b/200
    eps=216/24389; k=24389/27
    def finv(t):
        t3=t**3
        return t3 if t3>eps else (116*t-16)/k
    xyz=np.array([finv(fx)*D65[0], finv(fy)*D65[1], finv(fz)*D65[2]])
    rgb_lin=XYZ_TO_SRGB @ xyz
    rgb_lin=np.clip(rgb_lin,0,1)
    rgb=np.where(rgb_lin<=0.0031308,12.92*rgb_lin,1.055*np.power(rgb_lin,1/2.4)-0.055)
    return tuple(np.clip(rgb*255,0,255).astype(np.uint8).tolist())

def de00(lab1, lab2):
    L1,a1,b1=[float(x) for x in lab1]; L2,a2,b2=[float(x) for x in lab2]
    C1=math.hypot(a1,b1); C2=math.hypot(a2,b2); avgC=(C1+C2)/2
    G=0.5*(1-math.sqrt(avgC**7/(avgC**7+25**7))) if avgC else 0
    a1p=(1+G)*a1; a2p=(1+G)*a2
    C1p=math.hypot(a1p,b1); C2p=math.hypot(a2p,b2)
    def hp(ap,bb):
        if ap==0 and bb==0: return 0.0
        h=math.degrees(math.atan2(bb,ap))
        return h+360 if h<0 else h
    h1p=hp(a1p,b1); h2p=hp(a2p,b2)
    dLp=L2-L1; dCp=C2p-C1p
    if C1p*C2p==0: dhp=0
    else:
        dh=h2p-h1p
        if dh>180: dh-=360
        elif dh<-180: dh+=360
        dhp=dh
    dHp=2*math.sqrt(C1p*C2p)*math.sin(math.radians(dhp/2))
    avgLp=(L1+L2)/2; avgCp=(C1p+C2p)/2
    if C1p*C2p==0: avghp=h1p+h2p
    else:
        if abs(h1p-h2p)<=180: avghp=(h1p+h2p)/2
        elif h1p+h2p<360: avghp=(h1p+h2p+360)/2
        else: avghp=(h1p+h2p-360)/2
    T=1-0.17*math.cos(math.radians(avghp-30))+0.24*math.cos(math.radians(2*avghp))+0.32*math.cos(math.radians(3*avghp+6))-0.20*math.cos(math.radians(4*avghp-63))
    dt=30*math.exp(-(((avghp-275)/25)**2))
    Rc=2*math.sqrt(avgCp**7/(avgCp**7+25**7)) if avgCp else 0
    Sl=1+(0.015*((avgLp-50)**2))/math.sqrt(20+(avgLp-50)**2)
    Sc=1+0.045*avgCp
    Sh=1+0.015*avgCp*T
    Rt=-math.sin(math.radians(2*dt))*Rc
    return float(math.sqrt((dLp/Sl)**2+(dCp/Sc)**2+(dHp/Sh)**2+Rt*(dCp/Sc)*(dHp/Sh)))

def pick_lab_cols(df, prefix):
    choices=[(f"{prefix}_L",f"{prefix}_a",f"{prefix}_b"),("corrected_L","corrected_a","corrected_b"),("final_L","final_a","final_b"),("visual_L","visual_a","visual_b"),("L","a","b")]
    for c in choices:
        if all(x in df.columns for x in c): return c
    raise RuntimeError("找不到Lab列。请确认有 corrected_L/corrected_a/corrected_b 或 final_L/final_a/final_b，或用 --lab-prefix 指定。")

def family(code, name, L, a, b):
    name=str(name)
    chroma=math.hypot(a,b)
    if b>=18 and L>=55: return "yellow"
    if any(k in name for k in ["黄","香槟","象牙"]) and b>=10 and L>=55: return "yellow"
    if chroma<=9: return "gray"
    if any(k in name for k in ["灰","银","白","黑","乌托","爵士","哑白","星光"]) and chroma<=16: return "gray"
    return "normal"

def fix_lab(L,a,b,fam,args):
    L2,a2,b2=float(L),float(a),float(b)
    if fam=="yellow":
        L2 += args.yellow_l_add
        b2 = b2*args.yellow_b_gain + args.yellow_b_add
    elif fam=="gray":
        a2 *= args.gray_a_gain
        if b2 < 1.0:
            b2 = b2*args.gray_b_gain_cold + args.gray_b_add_cold
        else:
            b2 = b2*args.gray_b_gain_warm + args.gray_b_add_warm
    return float(np.clip(L2,0,100)), float(np.clip(a2,-128,127)), float(np.clip(b2,-128,127))

def draw_board(df, out, cols_lab, title):
    Lc,ac,bc=cols_lab
    cols=8; pw=104; ph=72; lh=38; gap=8
    rows=math.ceil(len(df)/cols)
    img=Image.new("RGB",(cols*pw+(cols+1)*gap, rows*(ph+lh)+(rows+1)*gap+36),(245,245,245))
    d=ImageDraw.Draw(img)
    try:
        font=ImageFont.truetype("simhei.ttf",13); tfont=ImageFont.truetype("simhei.ttf",18)
    except Exception:
        font=ImageFont.load_default(); tfont=ImageFont.load_default()
    d.text((gap,8),title,fill=(0,0,0),font=tfont)
    for i,(_,r) in enumerate(df.iterrows()):
        rr=i//cols; cc=i%cols
        x=gap+cc*(pw+gap); y=36+gap+rr*(ph+lh+gap)
        rgb=lab_to_srgb((r[Lc],r[ac],r[bc]))
        d.rectangle([x,y,x+pw,y+ph], fill=rgb, outline=(60,60,60))
        code=str(r.get("code",r.get("编号","")))
        name=str(r.get("name",r.get("名称","")))
        fam=str(r.get("family_fix",""))
        d.text((x,y+ph+2),(code+" "+name)[:14],fill=(0,0,0),font=font)
        d.text((x,y+ph+18),fam,fill=(80,80,80),font=font)
    out.parent.mkdir(parents=True,exist_ok=True)
    img.save(out)

def main():
    ap=argparse.ArgumentParser(description="v09基础上做黄色补黄、灰色去蓝的分色系视觉微调。")
    ap.add_argument("--in-csv",required=True)
    ap.add_argument("--out-dir",default="v10_family_fix_out")
    ap.add_argument("--lab-prefix",default="corrected")
    ap.add_argument("--std-prefix",default="std")
    ap.add_argument("--code-col",default="code")
    ap.add_argument("--name-col",default="name")
    ap.add_argument("--yellow-l-add",type=float,default=-0.3)
    ap.add_argument("--yellow-b-add",type=float,default=3.0)
    ap.add_argument("--yellow-b-gain",type=float,default=1.05)
    ap.add_argument("--gray-a-gain",type=float,default=0.75)
    ap.add_argument("--gray-b-gain-cold",type=float,default=0.55)
    ap.add_argument("--gray-b-add-cold",type=float,default=2.0)
    ap.add_argument("--gray-b-gain-warm",type=float,default=0.85)
    ap.add_argument("--gray-b-add-warm",type=float,default=0.3)
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.in_csv,encoding="utf-8-sig")
    Lc,ac,bc=pick_lab_cols(df,args.lab_prefix)
    std_cols=(f"{args.std_prefix}_L",f"{args.std_prefix}_a",f"{args.std_prefix}_b")
    has_std=all(c in df.columns for c in std_cols)

    rows=[]
    for _,r in df.iterrows():
        row=dict(r)
        code=r.get(args.code_col,r.get("编号",""))
        name=r.get(args.name_col,r.get("名称",""))
        L=float(pd.to_numeric(r[Lc],errors="coerce")); a=float(pd.to_numeric(r[ac],errors="coerce")); b=float(pd.to_numeric(r[bc],errors="coerce"))
        fam=family(code,name,L,a,b)
        L2,a2,b2=fix_lab(L,a,b,fam,args)
        row.update({"family_fix":fam,"fix_L":L2,"fix_a":a2,"fix_b":b2,"fix_delta_L":L2-L,"fix_delta_a":a2-a,"fix_delta_b":b2-b})
        if has_std:
            std=(float(r[std_cols[0]]),float(r[std_cols[1]]),float(r[std_cols[2]]))
            row["before_self_deltaE"]=de00((L,a,b),std)
            row["after_self_deltaE"]=de00((L2,a2,b2),std)
            row["self_deltaE_improvement"]=row["before_self_deltaE"]-row["after_self_deltaE"]
        rows.append(row)

    outdf=pd.DataFrame(rows)
    out_csv=out/"glue_128_v10_family_fix.csv"
    outdf.to_csv(out_csv,index=False,encoding="utf-8-sig")
    draw_board(outdf,out/"preview_before.png",(Lc,ac,bc),"before family fix")
    draw_board(outdf,out/"preview_after.png",("fix_L","fix_a","fix_b"),"after family fix")
    summary={"in_csv":args.in_csv,"lab_cols":[Lc,ac,bc],"family_counts":outdf["family_fix"].value_counts().to_dict(),"out_csv":str(out_csv),"params":vars(args)}
    if has_std:
        summary.update({
            "before_mean_deltaE":float(outdf["before_self_deltaE"].mean()),
            "after_mean_deltaE":float(outdf["after_self_deltaE"].mean()),
            "before_p95_deltaE":float(outdf["before_self_deltaE"].quantile(0.95)),
            "after_p95_deltaE":float(outdf["after_self_deltaE"].quantile(0.95)),
        })
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("=== Done ===")
    print("lab cols:",Lc,ac,bc)
    print("family counts:",summary["family_counts"])
    if has_std:
        print("before mean ΔE:",summary["before_mean_deltaE"])
        print("after  mean ΔE:",summary["after_mean_deltaE"])
        print("before p95  ΔE:",summary["before_p95_deltaE"])
        print("after  p95  ΔE:",summary["after_p95_deltaE"])
    print("out csv:",out_csv)
    print("preview before:",out/"preview_before.png")
    print("preview after :",out/"preview_after.png")
    print("summary:",out/"summary.json")

if __name__=="__main__":
    main()
