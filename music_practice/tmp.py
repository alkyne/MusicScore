import numpy as np
from scipy import ndimage, misc
from glob import glob
import os
import matplotlib.pyplot as plt
import shutil

import pdf2image

def read_pdf(filepath):
    pages = pdf2image.convert_from_path(filepath)
    return pages

def checkpath(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

def clear_dataset():
    shutil.rmtree('Dataset/')

def save_pages(pages, composer):
    path = 'Converted_Data/%s/' %composer
    checkpath(path)
    for page in pages:
        file_number = len(glob(path + '*'))
        page.save(path + str(composer + str(file_number) + '.jpg'))

def save_rows(sliced_rows, composer):
    path = 'Dataset/%s/' %composer
    checkpath(path)
    for row in sliced_rows:
        file_number = len(glob(path + '*'))

        #Resize image for smaller set
        row = misc.imresize(row, (35,200))

        misc.imsave(str(path) + '/' + str(file_number) + '.jpg', row)

def process_composer(composer):
    print('Processing: ' + str(composer))
    files = glob('Source_Data/%s/*' %(composer))
    file_num = 0
    for f in files:
        print('Working on %s, PDF file %s of %s' %(composer, file_num, len(files)))
        pages = read_pdf(f)
        save_pages(pages, composer)
        file_num += 1

def open_image(path):
    image = ndimage.imread(path, mode='L')
    #check image size, restrict to 2000 wide
    if image.shape[1] > 2000:
        image = image[0:, 0:2000]

    return image

def line_contrast(page_image):
    line_contr =[]
    for line in page_image:
        line_contr.append(max(line) - min(line))
    return line_contr

def detect_bars(page_image):
    avg_contrast = []
    for line in page_image:
        avg_contrast.append(np.sum(line) / len(line))
    #avg_contrast = np.power(avg_contrast, 3)
    return avg_contrast

def find_rows(line_contr):
    detected_rows = []
    row_start = 0
    row_end = 0
    detect_state = 0 #0 if previous line was not part of a row
    cur_row = 0
    for contrast in line_contr:
        if contrast < 50 and detect_state == 0:
            row_start = cur_row
        elif contrast >= 50 and detect_state == 0:
            row_start = cur_row
            detect_state = 1
        elif contrast < 50 and detect_state == 1: #if end of row, evaluate AOI height
            row_end = cur_row
            rowheight = row_start - row_end
            #print("height of row was: " + str(rowheight))
            if abs(rowheight) >= 150:
                detected_rows.append((row_start, row_end))
            detect_state = 0
        elif contrast >= 50 and detect_state == 1:
            pass
        else:
            print("unknown situation, help!, state: " + str(detect_state))
        cur_row += 1
    return detected_rows

def slice_rows(page_image, detected_rows, composer):
    sliced_rows = []
    max_height= 350
    max_width = 2000
    for x,y in detected_rows:
        im_sliced = np.copy(page_image[x:y])
        new_im = np.empty((max_height, max_width))
        new_im.fill(255)
        if im_sliced.shape[0] <= max_height:
            new_im[0:im_sliced.shape[0], 0:im_sliced.shape[1]] = im_sliced
            sliced_rows.append(new_im)
        elif max_height < im_sliced.shape[0] < 1.25 * max_height:
            im_sliced = im_sliced[0:max_height, 0:im_sliced.shape[1]]
            new_im[0:im_sliced.shape[0], 0:im_sliced.shape[1]] = im_sliced
            sliced_rows.append(new_im)
        else:
            print("Skipping block of height: %s px" %im_sliced.shape[0])
            checkpath('Dataset/%s/Errors/' %composer)
            file_number = len(glob('Dataset/%s/Errors/*' %composer))
            #save to error dir for manual inspection
            misc.imsave('Dataset/%s/Errors/%s_%s.jpg' %(composer, file_number, composer), im_sliced)
    return sliced_rows

if __name__ == '__main__':
    try: 
        clear_dataset()
    except:
        pass
    composers = [x.split('/')[-1] for x in glob('Source_Data/*')]


    for composer in composers:
        composer = composer.split('\\')[-1]
        #process_composer(composer)
        images = glob('Converted_Data/' + str(composer) + '/*')
        imagefile_num = 0
        for imagefile in images:
            print('Working on %s, image file %s of %s' %(composer, imagefile_num, len(images)))
            image = open_image(imagefile)
            line_contr = line_contrast(image)
            #line_contr = detect_bars(image)
            detected_rows = find_rows(line_contr)
            sliced_rows = slice_rows(image, detected_rows, composer)
            save_rows(sliced_rows, composer)
            imagefile_num += 1

    '''image = open_image('Beethoven68.jpg')
    line_contr = line_contrast(image)
    plt.plot(line_contr)
    plt.title('Row-wise pixel value range')
    plt.xlabel('Horizontal row #')
    plt.ylabel('Pixel value range on row')
    plt.show()
        #detected_rows = find_rows(line_contr)'''