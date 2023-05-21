import os
import glob
import matplotlib.pyplot as plt
import sure as sure
from skimage import io, exposure, img_as_float

def process_image(image_path, output_dir):  #definirea functiei process_image cu argumente:
                                            # image_path (calea către fișierul imagine) și output_dir (directorul de ieșire).
    img = io.imread(image_path)             #citirea imaginii din calea specificata

    min_val = img.min()
    max_val = img.max()
    contrast_diff = max_val - min_val          #stabilim contrastul minim pt citire
    threshold = 257                            #valoarea de prag a contrastrastului

    print(f"Contrast diferit pentru {image_path}: {contrast_diff}")

    if contrast_diff < threshold:            # verificam daca diferenta de contrast este mai mica decat pragul.
        img_rescale = exposure.rescale_intensity(img)    #editare imagine prntru a indeplini cerinta(procesare imaginii)

        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_processed{ext}") #ne arata in fisierul selectat
                                                                        # ca imaginile au fost procesate
        io.imsave(output_path, img_rescale)
        print(f"Proceseaza imaginile salvate: {output_path}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))   #creearea axelor
        plot_img_and_hist(img_rescale, axes)
        plt.show()
    else:
        print(f"Treci peste imagini, contrast suficient: {image_path}")

def process_images_in_folder(folder_path, output_dir, extension=".jpg"): #analizarea imaginilor cu extensia respectiva
    image_files = glob.glob(os.path.join(folder_path, f"*{extension}"))

    for image_file in image_files:
        process_image(image_file, output_dir)

def plot_img_and_hist(image, axes, bins=256):   #creeaza grafice pe baza imaginilor analizate

    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

folder_path = r"C:\Users\Daiana\Desktop\lp\poze contrast scazut"       #extractie date
output_dir = r"C:\Users\Daiana\Desktop\lp\poze contrast mare"          #date procesate

process_images_in_folder(folder_path, output_dir)

#bibliografie
#https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
#https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.is_low_contrast
#https://www.analyticsvidhya.com/blog/2022/08/image-contrast-enhancement-using-clahe/
#https://bigmms.github.io/chen_tcsvt19_enhancement/