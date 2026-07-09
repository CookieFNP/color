# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np

D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
SRGB_TO_XYZ = np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]], dtype=np.float64)
XYZ_TO_SRGB = np.array([[3.2404542,-1.5371385,-0.4985314],[-0.9692660,1.8760108,0.0415560],[0.0556434,-0.2040259,1.0572252]], dtype=np.float64)

@dataclass
class Std:
    code: str
    name: str
    lab: np.ndarray

def imread(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)

def imwrite(path, img):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or '.png', img)
    if not ok:
        raise RuntimeError(f'写图失败: {path}')
    buf.tofile(str(path))

def srgb_to_linear(rgb):
    x = np.asarray(rgb, dtype=np.float64)
    if x.size and x.max() > 1.0:
        x = x / 255.0
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0, 1)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1 / 2.4)) - 0.055)
    return np.clip(np.round(y * 255), 0, 255).astype(np.uint8)

def rgb_to_lab(rgb):
    lin = srgb_to_linear(rgb)
    xyz = lin @ SRGB_TO_XYZ.T
    t = xyz / D65
    e = 216 / 24389; k = 24389 / 27
    f = np.where(t > e, np.cbrt(t), (k * t + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)

def lab_to_rgb(lab):
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[...,0], lab[...,1], lab[...,2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    e = 216 / 24389; k = 24389 / 27
    def inv(t):
        t3 = t ** 3
        return np.where(t3 > e, t3, (116 * t - 16) / k)
    xyz = np.stack([inv(fx), inv(fy), inv(fz)], axis=-1) * D65
    lin = xyz @ XYZ_TO_SRGB.T
    return linear_to_srgb(lin)

def de2000(lab1, lab2):
    lab1 = np.asarray(lab1, dtype=np.float64); lab2 = np.asarray(lab2, dtype=np.float64)
    L1,a1,b1 = lab1[...,0],lab1[...,1],lab1[...,2]
    L2,a2,b2 = lab2[...,0],lab2[...,1],lab2[...,2]
    C1 = np.sqrt(a1*a1+b1*b1); C2 = np.sqrt(a2*a2+b2*b2)
    Cb = (C1+C2)/2; G = 0.5*(1-np.sqrt((Cb**7)/(Cb**7+25**7+1e-30)))
    a1p=(1+G)*a1; a2p=(1+G)*a2
    C1p=np.sqrt(a1p*a1p+b1*b1); C2p=np.sqrt(a2p*a2p+b2*b2)
    h1p=(np.degrees(np.arctan2(b1,a1p))+360)%360; h2p=(np.degrees(np.arctan2(b2,a2p))+360)%360
    dLp=L2-L1; dCp=C2p-C1p
    dh=h2p-h1p; dh=np.where(C1p*C2p==0,0,dh); dh=np.where(dh>180,dh-360,dh); dh=np.where(dh<-180,dh+360,dh)
    dHp=2*np.sqrt(C1p*C2p)*np.sin(np.radians(dh)/2)
    Lbp=(L1+L2)/2; Cbp=(C1p+C2p)/2
    hsum=h1p+h2p; hdiff=np.abs(h1p-h2p)
    hbp=np.where(C1p*C2p==0, hsum, np.where(hdiff<=180, hsum/2, np.where(hsum<360,(hsum+360)/2,(hsum-360)/2)))
    T=1-0.17*np.cos(np.radians(hbp-30))+0.24*np.cos(np.radians(2*hbp))+0.32*np.cos(np.radians(3*hbp+6))-0.20*np.cos(np.radians(4*hbp-63))
    dt=30*np.exp(-(((hbp-275)/25)**2)); Rc=2*np.sqrt((Cbp**7)/(Cbp**7+25**7+1e-30))
    Sl=1+(0.015*((Lbp-50)**2))/np.sqrt(20+(Lbp-50)**2); Sc=1+0.045*Cbp; Sh=1+0.015*Cbp*T
    Rt=-np.sin(np.radians(2*dt))*Rc
    return np.sqrt((dLp/Sl)**2+(dCp/Sc)**2+(dHp/Sh)**2+Rt*(dCp/Sc)*(dHp/Sh))

def parse_lab_cell(s):
    s = str(s).strip().strip('"').strip("'").replace('，', ',')
    p = [x.strip() for x in s.split(',') if x.strip()]
    if len(p) != 3:
        raise ValueError(f'Lab格式错误: {s}')
    return np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float64)

def find_csv():
    names = []
    for base in [Path.cwd(), Path(__file__).resolve().parent]:
        for p in base.glob('*.csv'):
            low = p.name.lower()
            if any(k in low for k in ['standard','standards','lab','标准','色']):
                names.append(p)
    if names:
        return sorted(set(names), key=lambda x: len(x.name))[0]
    allcsv = list(Path.cwd().glob('*.csv'))
    if len(allcsv) == 1:
        return allcsv[0]
    raise FileNotFoundError('未找到标准Lab CSV，请用 --standards-csv 指定')

def load_stds(path):
    d = {}
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.reader(f):
            if not row or len(row) < 3: continue
            if row[0].strip().lower() in {'code','id','编号'}: continue
            code = row[0].strip().upper(); name = row[1].strip()
            try:
                lab = parse_lab_cell(row[2])
            except Exception:
                if len(row) >= 5:
                    lab = np.array([float(row[2]), float(row[3]), float(row[4])], dtype=np.float64)
                else:
                    raise
            d[code] = Std(code, name, lab)
    if not d: raise RuntimeError('标准CSV没有有效行')
    return d

def nearest(lab, stds, topk=3):
    codes = list(stds.keys()); labs = np.array([stds[c].lab for c in codes], dtype=np.float64)
    de = de2000(np.repeat(np.asarray(lab).reshape(1,3), len(codes), axis=0), labs)
    idx = np.argsort(de)[:topk]
    return [{'code':stds[codes[i]].code,'name':stds[codes[i]].name,'lab':stds[codes[i]].lab.round(4).tolist(),'delta_e_2000':float(de[i])} for i in idx]

def resize_display(img, max_side=1200):
    h,w = img.shape[:2]; s = min(1.0, max_side / max(h,w))
    if s >= 0.999: return img.copy(), 1.0
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA), s

def select_four_points(img):
    disp, scale = resize_display(img); pts=[]; view=disp.copy()
    def redraw():
        nonlocal view
        view = disp.copy()
        cv2.putText(view, 'Click chart corners: TL,TR,BR,BL | R reset | Enter OK', (20,40), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,255,255), 2)
        for i,(x,y) in enumerate(pts):
            cv2.circle(view,(x,y),6,(0,0,255),-1); cv2.putText(view,str(i+1),(x+8,y-8),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
    def cb(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x,y)); redraw()
    redraw(); win='select ColorChecker corners'; cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.setMouseCallback(win, cb)
    while True:
        cv2.imshow(win, view); k=cv2.waitKey(20)&0xff
        if k in [13,10] and len(pts)==4: break
        if k in [ord('r'),ord('R')]: pts.clear(); redraw()
        if k == 27: cv2.destroyWindow(win); raise RuntimeError('取消选点')
    cv2.destroyWindow(win)
    return (np.array(pts, dtype=np.float32) / scale).astype(np.float32)

def select_roi(img, label):
    disp, scale = resize_display(img); win = f'ROI {label}'
    print(f'请框选 {label}，Enter/Space确认')
    x,y,w,h = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False); cv2.destroyWindow(win)
    if w <= 0 or h <= 0: raise RuntimeError(f'未选择ROI: {label}')
    H,W = img.shape[:2]
    x1=int(round(x/scale)); y1=int(round(y/scale)); x2=int(round((x+w)/scale)); y2=int(round((y+h)/scale))
    x1=max(0,min(W-1,x1)); x2=max(x1+1,min(W,x2)); y1=max(0,min(H-1,y1)); y2=max(y1+1,min(H,y2))
    return (x1,y1,x2,y2)

def warp_chart(photo, corners, size):
    w,h = size
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.asarray(corners,dtype=np.float32), dst)
    return cv2.warpPerspective(photo, M, (w,h)), M

def trimmed_mean_rgb(pix, trim=10):
    pix = np.asarray(pix, dtype=np.float64).reshape(-1,3)
    if pix.shape[0] == 0: return np.zeros(3)
    if trim <= 0 or pix.shape[0] < 10: return pix.mean(axis=0)
    L = rgb_to_lab(pix)[:,0]
    lo,hi = np.percentile(L, trim), np.percentile(L, 100-trim)
    m = (L >= lo) & (L <= hi)
    if m.sum() < 5: m = np.ones(len(pix), dtype=bool)
    return pix[m].mean(axis=0)

def extract_chart(img_bgr, rows=4, cols=6, center=.5, trim=10):
    h,w = img_bgr.shape[:2]; rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); out=[]
    pw,ph = w/cols, h/rows
    for r in range(rows):
        for c in range(cols):
            cx,cy=(c+.5)*pw,(r+.5)*ph; sw,sh=pw*center,ph*center
            x1,x2=int(round(cx-sw/2)),int(round(cx+sw/2)); y1,y2=int(round(cy-sh/2)),int(round(cy+sh/2))
            x1=max(0,min(w-1,x1)); x2=max(x1+1,min(w,x2)); y1=max(0,min(h-1,y1)); y2=max(y1+1,min(h,y2))
            out.append(trimmed_mean_rgb(rgb[y1:y2,x1:x2].reshape(-1,3), trim))
    return np.array(out, dtype=np.float64)

def features(linear_rgb, model='root_poly2'):
    flat=np.asarray(linear_rgb,dtype=np.float64).reshape(-1,3); R=flat[:,0:1]; G=flat[:,1:2]; B=flat[:,2:3]; one=np.ones_like(R); eps=1e-12
    if model=='linear_bias': return np.c_[R,G,B,one]
    if model=='poly2': return np.c_[R,G,B,R*R,G*G,B*B,R*G,R*B,G*B,one]
    if model=='root_poly2': return np.c_[R,G,B,np.sqrt(np.maximum(R*G,eps)),np.sqrt(np.maximum(R*B,eps)),np.sqrt(np.maximum(G*B,eps)),one]
    raise ValueError(model)

def chart_weights(ref_rgb, mode='none', gray_weight=4, light_weight=2.5, light_L=70):
    n=len(ref_rgb); w=np.ones(n,dtype=np.float64); mode=mode.lower()
    if mode=='none': return w
    if mode in ['gray','gray_light'] and n>=24: w[18:24]=np.maximum(w[18:24], gray_weight)
    if mode in ['light','gray_light']:
        L = rgb_to_lab(ref_rgb)[:,0]; w[L>=light_L]=np.maximum(w[L>=light_L], light_weight)
    return w/w.mean()

def fit_model(cap_rgb, ref_rgb, model='root_poly2', ridge=1e-6, sample_w=None):
    X=features(srgb_to_linear(cap_rgb), model); Y=srgb_to_linear(ref_rgb).reshape(-1,3)
    if sample_w is not None:
        sw=np.asarray(sample_w,dtype=np.float64).reshape(-1); sw=np.maximum(sw,1e-8); sw=sw/sw.mean(); s=np.sqrt(sw)[:,None]; X=X*s; Y=Y*s
    if ridge>0:
        reg=ridge*np.eye(X.shape[1]); reg[-1,-1]=0
        return np.linalg.solve(X.T@X+reg, X.T@Y)
    W,_,_,_=np.linalg.lstsq(X,Y,rcond=None); return W

def apply_model(img_bgr, W, model='root_poly2'):
    rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); lin=srgb_to_linear(rgb); h,w=lin.shape[:2]
    out=(features(lin,model)@W).reshape(h,w,3)
    return cv2.cvtColor(linear_to_srgb(out), cv2.COLOR_RGB2BGR)

def apply_lab_shift(img_bgr, shift):
    rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); lab=rgb_to_lab(rgb)+np.asarray(shift).reshape(1,1,3)
    lab[...,0]=np.clip(lab[...,0],0,100); lab[...,1]=np.clip(lab[...,1],-128,127); lab[...,2]=np.clip(lab[...,2],-128,127)
    return cv2.cvtColor(lab_to_rgb(lab), cv2.COLOR_RGB2BGR)

def roi_rgb(img_bgr, roi, trim=10, debug=None):
    x1,y1,x2,y2=roi; crop=img_bgr[y1:y2,x1:x2]; rgb=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB); h,w=crop.shape[:2]
    m=np.zeros((h,w),np.uint8); mx=int(w*.07); my=int(h*.07); m[my:h-my if h-my>my else h, mx:w-mx if w-mx>mx else w]=255
    pix=rgb[m>0].reshape(-1,3)
    if len(pix)<20: pix=rgb.reshape(-1,3); m[:]=255
    lab=rgb_to_lab(pix); L=lab[:,0]; lo,hi=np.percentile(L,trim),np.percentile(L,100-trim); keep=(L>=lo)&(L<=hi)
    if keep.sum()>=30:
        med=np.median(lab[keep],axis=0); dist=np.linalg.norm(lab[keep]-med,axis=1); thr=np.percentile(dist,90); idx=np.where(keep)[0]; keep2=np.zeros_like(keep); keep2[idx[dist<=thr]]=True
        if keep2.sum()>=20: keep=keep2
    if keep.sum()<20: keep=np.ones(len(pix),dtype=bool)
    if debug:
        vis=crop.copy(); ys,xs=np.where(m>0); dbg=np.zeros_like(m)
        if len(xs)==len(keep): dbg[ys[keep],xs[keep]]=255
        overlay=vis.copy(); overlay[dbg>0]=(0,255,0); imwrite(debug, cv2.addWeighted(vis,.65,overlay,.35,0))
    return pix[keep].mean(axis=0)

def mask_for_mode(mode):
    mode=mode.lower()
    return {'none':np.array([0,0,0.]),'l':np.array([1,0,0.]),'b':np.array([0,0,1.]),'lb':np.array([1,0,1.]),'lab':np.array([1,1,1.])}[mode]

def parse_list(s, cast=str):
    return [cast(x.strip()) for x in str(s).replace('，',',').split(',') if x.strip()]

def stats(vals):
    a=np.asarray(vals,dtype=np.float64)
    return dict(mean=float(a.mean()), median=float(np.median(a)), max=float(a.max()), p95=float(np.percentile(a,95)), std=float(a.std()))

def write_csv(path, rows):
    if not rows: return
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore',restval=''); wr.writeheader(); wr.writerows(rows)

def flat(row):
    out={}
    for k,v in row.items():
        if isinstance(v,(list,dict)): out[k]=json.dumps(v,ensure_ascii=False)
        elif isinstance(v,np.ndarray): out[k]=json.dumps(v.tolist(),ensure_ascii=False)
        else: out[k]=v
    return out

def draw_rois(img, specs, title):
    out=img.copy()
    for s in specs:
        x1,y1,x2,y2=s['roi']; cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,255),3); cv2.putText(out,f"{s['idx']}:{s['code']}",(x1,max(30,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,255,255),2)
    cv2.putText(out,title,(30,55),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
    return out

def side(left,right):
    h=min(left.shape[0],right.shape[0])
    if left.shape[0]!=h: left=cv2.resize(left,(int(left.shape[1]*h/left.shape[0]),h))
    if right.shape[0]!=h: right=cv2.resize(right,(int(right.shape[1]*h/right.shape[0]),h))
    return np.hstack([left,right])

def get_codes(args, stds):
    if args.target_codes:
        codes=[x.upper() for x in parse_list(args.target_codes,str)]
    else:
        n=int(input('请输入要框选的胶块数量：').strip()); codes=[]
        for i in range(n): codes.append(input(f'请输入第{i+1}/{n}个胶块编号，例如W015：').strip().upper())
    miss=[c for c in codes if c not in stds]
    if miss: raise KeyError(f'标准CSV中找不到: {miss}')
    print('目标顺序:', ' -> '.join([f'{c}({stds[c].name})' for c in codes]))
    return codes

def save_roi_json(path, specs):
    rows=[{'idx':s['idx'],'code':s['code'],'roi':list(map(int,s['roi']))} for s in specs]
    Path(path).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')

def load_roi_json(path, stds):
    rows=json.loads(Path(path).read_text(encoding='utf-8-sig')); specs=[]
    for r in rows:
        c=r['code'].upper(); specs.append({'idx':int(r['idx']),'code':c,'name':stds[c].name,'std_lab':stds[c].lab,'roi':tuple(map(int,r['roi']))})
    return specs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--photo',default='IMG_0800.jpg')
    ap.add_argument('--standard-chart',default='standard_chart.png')
    ap.add_argument('--standards-csv',default=None)
    ap.add_argument('--out',default='output_final_optimizer')
    ap.add_argument('--target-codes',default=None,help='例如 W015,W016,W031；为空则手动输入数量和序号')
    ap.add_argument('--roi-json',default=None,help='复用已保存ROI，不重新框选')
    ap.add_argument('--rows',type=int,default=4); ap.add_argument('--cols',type=int,default=6)
    ap.add_argument('--center-ratio',type=float,default=.5); ap.add_argument('--chart-trim',type=float,default=10); ap.add_argument('--target-trim',type=float,default=10)
    ap.add_argument('--model',default='root_poly2',choices=['linear_bias','poly2','root_poly2'])
    ap.add_argument('--ridge-list',default='1e-6')
    ap.add_argument('--chart-weight-modes',default='none,gray_light',help='none,gray,light,gray_light')
    ap.add_argument('--gray-weight',type=float,default=4); ap.add_argument('--light-weight',type=float,default=2.5); ap.add_argument('--light-l-threshold',type=float,default=70)
    ap.add_argument('--lab-shift-modes',default='none,Lb',help='none,L,b,Lb,Lab')
    ap.add_argument('--k-list',default='0,0.5,0.75,1.0,1.25')
    ap.add_argument('--select-metric',default='mean',choices=['mean','max','p95'])
    ap.add_argument('--debug',action='store_true')
    args=ap.parse_args()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    csv_path=Path(args.standards_csv) if args.standards_csv else find_csv(); stds=load_stds(csv_path)
    print(f'读取标准CSV: {csv_path}，共{len(stds)}个')
    photo=imread(args.photo); stdchart=imread(args.standard_chart)
    if photo is None: raise FileNotFoundError(args.photo)
    if stdchart is None: raise FileNotFoundError(args.standard_chart)

    print('点击实拍图中色卡四角：左上、右上、右下、左下')
    corners=select_four_points(photo); sh,sw=stdchart.shape[:2]; capchart,M=warp_chart(photo,corners,(sw,sh))
    imwrite(out/'01_captured_chart_warped.png',capchart); imwrite(out/'02_standard_chart.png',stdchart)
    cap_rgb=extract_chart(capchart,args.rows,args.cols,args.center_ratio,args.chart_trim); ref_rgb=extract_chart(stdchart,args.rows,args.cols,args.center_ratio,args.chart_trim)
    ref_lab_chart=rgb_to_lab(ref_rgb); cap_lab_chart=rgb_to_lab(cap_rgb); chart_de_before=de2000(cap_lab_chart,ref_lab_chart)

    if args.roi_json:
        specs=load_roi_json(args.roi_json,stds)
    else:
        codes=get_codes(args,stds); specs=[]
        for i,c in enumerate(codes,1): specs.append({'idx':i,'code':c,'name':stds[c].name,'std_lab':stds[c].lab,'roi':select_roi(photo,f'{i:02d}_{c}_{stds[c].name}')})
        save_roi_json(out/'selected_rois.json',specs)
    print(f'目标ROI数量: {len(specs)}')

    before_labs=[]
    maskdir=out/'target_masks'; maskdir.mkdir(exist_ok=True)
    for s in specs:
        rgb=roi_rgb(photo,s['roi'],args.target_trim,maskdir/f"target_{s['idx']:02d}_{s['code']}_before_mask.png" if args.debug else None)
        before_labs.append(rgb_to_lab(rgb[None,:])[0])

    ridge_list=parse_list(args.ridge_list,float); modes=parse_list(args.chart_weight_modes,str); shift_modes=parse_list(args.lab_shift_modes,str); k_list=parse_list(args.k_list,float)
    cands=[]; details={}; cache={}; cid=0
    for ridge in ridge_list:
        for mode in modes:
            cw=chart_weights(ref_rgb,mode,args.gray_weight,args.light_weight,args.light_l_threshold)
            W=fit_model(cap_rgb,ref_rgb,args.model,ridge,cw)
            base_photo=apply_model(photo,W,args.model); base_chart=apply_model(capchart,W,args.model)
            base_labs=[]
            for s in specs:
                rgb=roi_rgb(base_photo,s['roi'],args.target_trim); base_labs.append(rgb_to_lab(rgb[None,:])[0])
            residual=np.array([lab-s['std_lab'] for lab,s in zip(base_labs,specs)]); mean_res=residual.mean(axis=0)
            base_chart_lab=rgb_to_lab(extract_chart(base_chart,args.rows,args.cols,args.center_ratio,args.chart_trim))
            for sm in shift_modes:
                ks=[0.0] if sm.lower()=='none' else k_list
                for k in ks:
                    shift=-mean_res*mask_for_mode(sm)*float(k)
                    rows=[]; des=[]; correct=0
                    for i,(s,blab,alab) in enumerate(zip(specs,before_labs,base_labs)):
                        lab=alab+shift; std=s['std_lab']; de=float(de2000(lab[None,:],std[None,:])[0]); nde=nearest(lab,stds,3); ok=nde[0]['code']==s['code']; correct+=ok; des.append(de)
                        bde=float(de2000(blab[None,:],std[None,:])[0])
                        rows.append({'idx':s['idx'],'code':s['code'],'name':s['name'],'roi':list(s['roi']),'standard_lab':std.round(4).tolist(),'before_lab':blab.round(4).tolist(),'after_lab':lab.round(4).tolist(),'before_deltaE':bde,'after_deltaE':de,'improvement':bde-de,'delta_lab_after':(lab-std).round(4).tolist(),'pred_code':nde[0]['code'],'pred_name':nde[0]['name'],'pred_deltaE':nde[0]['delta_e_2000'],'correct':bool(ok),'nearest_after':nde})
                    st=stats(des); chart_de=de2000(base_chart_lab+shift.reshape(1,3),ref_lab_chart)
                    cand={'candidate_id':cid,'model':args.model,'ridge_alpha':ridge,'chart_weight_mode':mode,'lab_shift_mode':sm,'k':float(k),'shift_L':float(shift[0]),'shift_a':float(shift[1]),'shift_b':float(shift[2]),'target_mean_deltaE':st['mean'],'target_median_deltaE':st['median'],'target_max_deltaE':st['max'],'target_p95_deltaE':st['p95'],'target_std_deltaE':st['std'],'classification_acc':correct/len(specs),'chart_before_mean_deltaE':float(chart_de_before.mean()),'chart_after_mean_deltaE':float(chart_de.mean()),'chart_after_max_deltaE':float(chart_de.max()),'mean_residual_L_before_shift':float(mean_res[0]),'mean_residual_a_before_shift':float(mean_res[1]),'mean_residual_b_before_shift':float(mean_res[2])}
                    cands.append(cand); details[cid]=rows; cache[cid]={'photo':base_photo,'chart':base_chart,'shift':shift}; cid+=1
    metric={'mean':'target_mean_deltaE','max':'target_max_deltaE','p95':'target_p95_deltaE'}[args.select_metric]
    best=sorted(cands,key=lambda r:(r[metric],-r['classification_acc'],r['target_max_deltaE']))[0]; bid=best['candidate_id']; shift=cache[bid]['shift']
    best_photo=apply_lab_shift(cache[bid]['photo'],shift) if np.linalg.norm(shift)>1e-10 else cache[bid]['photo']
    best_chart=apply_lab_shift(cache[bid]['chart'],shift) if np.linalg.norm(shift)>1e-10 else cache[bid]['chart']
    imwrite(out/'03_best_corrected_photo.png',best_photo); imwrite(out/'04_best_corrected_chart.png',best_chart)
    imwrite(out/'05_before_after_best.png',side(draw_rois(photo,specs,'Before'),draw_rois(best_photo,specs,'Best corrected')))
    write_csv(out/'candidate_summary.csv',[flat(x) for x in cands]); write_csv(out/'top20_candidates.csv',[flat(x) for x in sorted(cands,key=lambda r:(r[metric],-r['classification_acc'],r['target_max_deltaE']))[:20]]); write_csv(out/'best_target_results.csv',[flat(x) for x in details[bid]])
    report={'input':{'photo':args.photo,'standard_chart':args.standard_chart,'standards_csv':str(csv_path),'standards_count':len(stds)},'note':'best 是按本批次已输入胶块标准编号优化出来的最小E效果；正式上线时 Lab shift 建议改为灰阶色卡/历史样本估计。','chart':{'corners':corners.round(3).tolist(),'before_mean_deltaE':float(chart_de_before.mean()),'before_max_deltaE':float(chart_de_before.max())},'best_candidate':best,'best_target_results':details[bid],'outputs':{'best_corrected_photo':str(out/'03_best_corrected_photo.png'),'best_corrected_chart':str(out/'04_best_corrected_chart.png'),'before_after_best':str(out/'05_before_after_best.png'),'candidate_summary_csv':str(out/'candidate_summary.csv'),'top20_candidates_csv':str(out/'top20_candidates.csv'),'best_target_results_csv':str(out/'best_target_results.csv'),'report':str(out/'report.json')}}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n========== 最佳结果 ==========')
    print(f"candidate_id: {bid}")
    print(f"model={best['model']} ridge={best['ridge_alpha']} weight={best['chart_weight_mode']} shift_mode={best['lab_shift_mode']} k={best['k']}")
    print(f"shift: L={best['shift_L']:.4f}, a={best['shift_a']:.4f}, b={best['shift_b']:.4f}")
    print(f"target mean/median/max/p95 ΔE00: {best['target_mean_deltaE']:.4f} / {best['target_median_deltaE']:.4f} / {best['target_max_deltaE']:.4f} / {best['target_p95_deltaE']:.4f}")
    print(f"classification_acc: {best['classification_acc']:.4f}")
    print(f"chart after mean/max ΔE00: {best['chart_after_mean_deltaE']:.4f} / {best['chart_after_max_deltaE']:.4f}")
    for r in details[bid]: print(f"[{r['idx']}] {r['code']} {r['name']}: {r['before_deltaE']:.3f}->{r['after_deltaE']:.3f}, pred={r['pred_code']} {r['pred_name']}, correct={r['correct']}")
    print('\n输出目录:', out)

if __name__ == '__main__':
    main()
