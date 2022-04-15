import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

from src.base.gt_loaders.gt_names import GTName


def __calc_row_idx(orig_idx):
    if orig_idx < 10:
        return 0
    elif orig_idx%10 >= 0:
        return orig_idx//10


def __calc_col_idx(orig_idx):
    if orig_idx < 10:
        return orig_idx
    elif orig_idx>=10:
        return orig_idx%10


def __calc_empty_slots_pos(nrows, ncols, n_imgs):
    pos_list = []
    n_empy_slots = None
    if n_imgs%ncols == 0:
        n_empy_slots = 0
    else:
        n_empy_slots = ncols - n_imgs%ncols 

    last_row_idx = nrows-1
    for i in range(n_empy_slots):
        pos_list.append((last_row_idx,ncols-i-1))
    return pos_list
    

def __calc_nrows(n_imgs, ncols):
    nrows = None
    if n_imgs < ncols:
        nrows = 1
    elif n_imgs%ncols == 0:
        nrows = int(n_imgs/10)
    elif n_imgs%ncols > 0:
        nrows = n_imgs//10 + 1
    return nrows


def __get_colors_seq(labels, predictions):
    color_seq = []
    for l,p in zip(labels, predictions):
        if l == p:
            color_seq.append('green')
        else:
            color_seq.append('red')
    return color_seq


def draw_imgs(imgs_list, labels=None, predictions=None, heatmaps=None, title="", figsize=None, from_nparray=False):
    if len(imgs_list) == 0:
        return
    
    NIMGS = len(imgs_list)
    NCOLS = 10
    NROWS = __calc_nrows(NIMGS, NCOLS)
    
    height = 2*NROWS+(NIMGS%NCOLS)-2 if NIMGS > 30 else NROWS+1
    fsize = figsize if figsize != None else (2*NCOLS, height)
    f, axarr = plt.subplots(NROWS, NCOLS, figsize=fsize)
    f.suptitle(title, fontsize=16, y=1.2 if NIMGS < 20 else 0.99)
    
    imgs_list = np.array(imgs_list)
    W,H,C = imgs_list.shape[1:4]
    
    new_imgs_list = np.ones((NROWS,NCOLS,W,H,C), dtype='uint8')*255
    
    for orig_im_idx in range(NIMGS):
        r_idx = __calc_row_idx(orig_im_idx)
        c_idx = __calc_col_idx(orig_im_idx)
        new_imgs_list[r_idx,c_idx,:,:,:] = imgs_list[orig_im_idx,:,:,:]
    
    new_imgs_list = new_imgs_list.reshape(NROWS, NCOLS, W, H, C)

    empty_slots_pos = __calc_empty_slots_pos(NROWS, NCOLS, NIMGS)
    
    for i,j in empty_slots_pos:
        new_imgs_list[i,j,:,:,:] = np.ones(imgs_list[0].shape, dtype='uint8')*255
    
    color_seq = None
    if labels is not None and predictions is not None:
        color_seq = __get_colors_seq(labels, predictions)
    
    idx = 0
    hm_idx = 0
    for r in range(NROWS):
        for c in range(NCOLS):
            image = new_imgs_list[r,c,:,:,:]
            
            if heatmaps is not None:
                image = cv2.resize(image, heatmaps[0].shape)
            
            if NIMGS <= NCOLS:
                if not from_nparray:
                    axarr[c].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    axarr[c].imshow(image)
                    
                if (0,c) not in empty_slots_pos:
                    if labels is not None and predictions is not None:
                        axarr[c].set_title(f"{c} - {labels[idx]}", color=color_seq[idx])
                        idx += 1
                    else:
                        axarr[c].set_title(f"{c}")
                
                if heatmaps is not None:
                    if hm_idx < len(heatmaps):
                        axarr[c].imshow(heatmaps[hm_idx], alpha=0.5)
                    hm_idx += 1
                
                axarr[c].axis('off')
                
            elif NIMGS > NCOLS:
                if not from_nparray:
                    axarr[r,c].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    axarr[r,c].imshow(image)
                
                if (r,c) not in empty_slots_pos:
                    if labels is not None and predictions is not None:
                        axarr[r,c].set_title(f"{r*NCOLS+c} - {labels[idx]}", color=color_seq[idx])
                        idx += 1
                    else:
                        axarr[r,c].set_title(f"{r*NCOLS+c}")
                
                    if heatmaps is not None:
                        axarr[r,c].imshow(heatmaps[hm_idx], alpha=0.5)
                        hm_idx += 1
                
                axarr[r,c].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return f


def show_log_img(log_imgs_list, img_idx):
    img1 = np.array(np.array(log_imgs_list))[img_idx,:,:,:]
    plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
    plt.show()

    
def draw_img(img, fig_size=None, img_size=None, from_nparray=False):
    new_img = img
    if img_size != None:
        new_img = cv2.resize(img, img_size)
    
    if not from_nparray:
        plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)


# len(imgs)%7 == 0 must be TRUE
def draw_gallery(imgs, ncols=7, show_gallery=True):
    array = np.array([np.asarray(Image.open(imgs[i]).convert('RGB')) for i in range(len(imgs))])
    
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))

    fig = plt.figure(figsize=(14,14),facecolor='w') 
    ax = fig.add_subplot(111)
    ax = ax.imshow(result, cmap=cm.jet)
    plt.axis('off')
    
    if show_gallery:
        plt.show()
    
    return fig
        
       
def __set_axe(ax, thresh_list, evals, title):
    ax.plot(thresh_list, evals.prec, color="red",    linestyle="--", marker="^", label="Precision")
    ax.plot(thresh_list, evals.rec,  color="green",  linestyle="--", marker="o", label="Recall")
    ax.plot(thresh_list, evals.npv,  color="blue",   linestyle='-',  marker="^", label='NPV')
    ax.plot(thresh_list, evals.acc,  color="orange", linestyle='-',  marker="o", label='Accuracy')
    ax.plot(thresh_list, evals.fpr,  color="purple", linestyle='-',  marker="^", label='FPR')
    ax.plot(thresh_list, evals.fnr,  color="black",  linestyle='-',  marker="o", label='FNR')
    ax.set_title(title)
    ax.set_ylim(0,100)
    ax.set_ylabel('percent')
    ax.set_xlabel('threshold')
    ax.grid(True)
    ax.set_yticks(range(-10, 100, 10))
    ax.legend()
    return ax


def draw_evaluations(eval_df, mult_thresh=False):
    fvc_evals = eval_df[eval_df.gtruth == GTName.FVC.value]
    pybossa_evals = eval_df[eval_df.gtruth == GTName.PYBOSSA.value]
    genki_evals = eval_df[eval_df.gtruth == GTName.GENKI.value]
    
    print(genki_evals.shape)

    size = (10,8) if not mult_thresh else (16,5)

    if mult_thresh:
        thresh_list_1_fvc = [t[0] for t in fvc_evals.thresh]
        thresh_list_2_fvc = [t[1] for t in fvc_evals.thresh]
        thresh_list_1_pybossa = [t[0] for t in pybossa_evals.thresh]
        thresh_list_2_pybossa = [t[1] for t in pybossa_evals.thresh]

        fig1,ax1 = plt.subplots(1, 2, figsize=size, sharey=True)
        __set_axe(ax1[0], thresh_list_1_fvc, fvc_evals, GTName.FVC.value.upper())
        __set_axe(ax1[1], thresh_list_1_pybossa, pybossa_evals, GTName.PYBOSSA.value.upper())

        fig2,ax2 = plt.subplots(1, 2, figsize=size, sharey=True)
        __set_axe(ax2[0], thresh_list_2_fvc, fvc_evals, GTName.FVC.value.upper())
        __set_axe(ax2[1], thresh_list_2_pybossa, pybossa_evals, GTName.PYBOSSA.value.upper())

        return fig1,fig2
    else:
        fig = None
        if genki_evals.shape[0] == 0:
            fig,ax = plt.subplots(2, 1, figsize=size)
            __set_axe(ax[0], fvc_evals.thresh, fvc_evals, GTName.FVC.value.upper())
            __set_axe(ax[1], pybossa_evals.thresh, pybossa_evals, GTName.PYBOSSA.value.upper())
        
        else:
            fig,ax = plt.subplots(3, 1, figsize=(16,10))
            __set_axe(ax[0], fvc_evals.thresh, fvc_evals, GTName.FVC.value.upper())
            __set_axe(ax[1], pybossa_evals.thresh, pybossa_evals, GTName.PYBOSSA.value.upper())
            __set_axe(ax[2], genki_evals.thresh, genki_evals, GTName.GENKI.value.upper())

        fig.tight_layout(pad=3.0)

        return fig    
        