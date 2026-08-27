from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.ndimage import sobel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.color_spaces import rgb_to_lab
from src.data_loader import load_rgb

NAVY, BLUE, ORANGE = "#17365D", "#4F81BD", "#F79646"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, required=True); parser.add_argument("--data-project-root", type=Path, required=True)
    args = parser.parse_args(); root=args.project_root.resolve(); data_root=args.data_project_root.resolve()
    inventory=pd.read_csv(root/'data/inventory.csv')
    sample=pd.concat(
        [group.sample(min(90, len(group)), random_state=42) for _, group in inventory.groupby('label')],
        ignore_index=True,
    )
    records=[]; rgb_values=[]; hsv_values=[]; lab_values=[]
    for row in sample.itertuples():
        image=load_rgb(data_root/row.relative_path,96)
        gray=.299*image[...,0]+.587*image[...,1]+.114*image[...,2]
        edge=np.hypot(sobel(gray,0),sobel(gray,1))
        # Chuẩn hóa theo biên Sobel lý thuyết để dùng cùng một ngưỡng cho mọi ảnh.
        edge_density = np.mean((edge / (4.0 * 255.0)) > 0.08)
        records.append({'label':row.label,'brightness':gray.mean(),'contrast':gray.std(),'edge_density':edge_density,'width':row.width,'height':row.height,'aspect_ratio':row.aspect_ratio})
        rgb_values.append(image.reshape(-1,3)[::64]); hsv_values.append(np.asarray(Image.fromarray(image).convert('HSV')).reshape(-1,3)[::64]); lab_values.append(rgb_to_lab(image).reshape(-1,3)[::64])
    stats=pd.DataFrame(records); stats.to_csv(root/'results/eda_sample_metrics.csv',index=False)
    sns.set_theme(style='whitegrid')
    for x,title,file in [('width','Phân bố chiều rộng ảnh','image_width_distribution.png'),('height','Phân bố chiều cao ảnh','image_height_distribution.png'),('aspect_ratio','Phân bố tỷ lệ khung hình','aspect_ratio_distribution.png'),('brightness','Phân bố độ sáng','brightness_distribution.png'),('contrast','Phân bố tương phản RMS','contrast_distribution.png'),('edge_density','Phân bố mật độ biên','edge_density_distribution.png')]:
        fig,ax=plt.subplots(figsize=(9,5)); sns.histplot(stats,x=x,hue='label',element='step',stat='density',common_norm=False,ax=ax); ax.set_title(title,color=NAVY,weight='bold'); fig.tight_layout(); fig.savefig(root/'figures/eda'/file,dpi=200); plt.close(fig)
    for values,names,title,file in [(np.vstack(rgb_values),['R','G','B'],'Histogram RGB','rgb_histogram.png'),(np.vstack(hsv_values),['H','S','V'],'Phân bố HSV','hsv_distribution.png'),(np.vstack(lab_values),['L*','a*','b*'],'Phân bố CIELAB','lab_distribution.png')]:
        fig,axs=plt.subplots(1,3,figsize=(12,3.8))
        for i,(ax,name) in enumerate(zip(axs,names)): sns.histplot(values[:,i],bins=40,ax=ax,color=[ORANGE,BLUE,NAVY][i]); ax.set_title(name)
        fig.suptitle(title,color=NAVY,weight='bold'); fig.tight_layout(); fig.savefig(root/'figures/eda'/file,dpi=200); plt.close(fig)
    print(stats.groupby('label')[['brightness','contrast','edge_density']].mean())


if __name__=='__main__': main()
