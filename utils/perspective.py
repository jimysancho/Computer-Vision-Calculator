import cv2 as cv
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def pad_image(img, N=50, M=50):
    rows, cols = img.shape
    mask = np.zeros((rows + N, cols + N), dtype=np.uint8)
    mask[N//2: -N//2,M//2: -M//2] = img.copy()
    return mask

def imshow(img, plot=True):
    if not plot:
        cv.imshow('c', img)
        cv.waitKey(0)
    else:
        plt.imshow(img, cmap='viridis')
        plt.colorbar()
        plt.show()
    
def order_vertices(vertices):
    
    # we order by x coordinate
    x_ordered = sorted(vertices, key=lambda x: x[0])
    
    # we order by y coordinate
    y_ordered = sorted(vertices, key=lambda x: x[1])
    
    up = np.array(y_ordered[:2])
    down = np.array(y_ordered[2:])
    left = np.array(x_ordered[:2])
    
    for point in up:
        if point in left:
            up_left = point
        else:
            up_right = point
            
    for point in down:
        if point in left:
            down_left = point
        else:
            down_right = point
            
    return np.array([up_left, up_right, down_left, down_right])

def get_new_vertices(vertices):

    x, y = vertices.transpose(1, 0)
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    x_max = x_sorted[-1] - x_sorted[0]
    y_max = y_sorted[-1] - y_sorted[0]
    
    new_vertices = np.array([[0, 0], [x_max, 0], [0, y_max], [x_max, y_max]])
    return new_vertices
        
def identify_equations(block, kernel=np.ones((1, 12)), iterations=6, 
                       show=False):
    
    block_copy = block.copy()
    c = cv.Canny(block, 0, 255)
    dil = cv.dilate(c, kernel, iterations=iterations)
    
    contours, _ = cv.findContours(dil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    equations = []
       
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(block_copy, (x, y), (x + w, y + h), (0, 0, 0), 2)
        equations.append([x, y, w, h])
    
    equations.sort(key=lambda x: x[1])     
    
    if show:   
        imshow(block_copy)
        
    return equations

def identify_characters(img, equation, kernel=np.ones((12, 2)), percentege=0.1):
    
    x, y, w, h = equation
    region = img[y: y + h, x: x + w]
    region_copy = region.copy()
    region = cv.GaussianBlur(region, (7, 7), 1, 1)
    c = cv.Canny(region, 0, 255)
    dil = cv.dilate(c, kernel=kernel, iterations=3)
    
    contours, _ = cv.findContours(dil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    symbols = []
    
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        margin = int((w * percentege))
        cv.rectangle(region_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
        symbols.append([x, y, w, h, margin])
        
    symbols.sort(key=lambda x: x[0])
    imshow(region_copy)
    return symbols, region
    
def draw_symbols(symbols, img):
    
    regions = []
    for symbol in symbols:
        x, y, w, h = symbol
        region = img[y: y + h, x: x + w]
        imshow(region)
        regions.append(region)
            
    return regions        

def final_prep(symbols, block, 
               shape=(20, 20), show=True, 
               pad_shape=(8, 8)):
    
    final_symbols = []
    for (x, y, w, h, _) in np.array(symbols):
        
        i = block[y: y + h, x: x + w]
        i = 255 - i
        _, binary = cv.threshold(i, 0, 255, cv.THRESH_OTSU)
        x, y, w, h = cv.boundingRect(binary)
        binary = binary[y: y + h, x: x + w]
        
        binary = cv.resize(binary, shape)
        binary = binary / binary.max()
        binary = cv.dilate(binary, np.ones((2, 2)))
        l = pad_image(binary, N=pad_shape[0], M=pad_shape[1])
        
        final_symbols.append(l)
        if show:
            imshow(l)
            
    return final_symbols

def equation_block(img, percentege=0.1, show=False, length=50):
    
    img_copy = img.copy()
    img_copy = cv.GaussianBlur(img_copy, (7, 7), 3, 3)
    canny = cv.Canny(img, 0, 255)
    dil = cv.dilate(canny, np.ones((3, 3)), iterations=3)
    ero = cv.erode(dil, np.ones((5, 5)))
    filled_mask = np.uint8(255 * ndimage.binary_fill_holes(ero, np.ones((3, 3))))
    filled_mask = cv.dilate(filled_mask, np.ones((2, 2)), iterations=10)
        
    canny_filled = cv.Canny(filled_mask, 0, 255)    
    dilate = cv.dilate(canny_filled, kernel=np.ones((3, 3)), iterations=5)
    dilate = np.uint8(255 * ndimage.binary_fill_holes(dilate))
           
    dilate = cv.GaussianBlur(dilate, (7, 7), 3, 3)
    vertices = cv.goodFeaturesToTrack(dilate, 4, 0.001, length).squeeze()
    new_vertices = get_new_vertices(vertices)
    vertices = order_vertices(vertices)
    
    for point in vertices:
        cv.circle(img_copy, point.astype(int), 10, (0, 0, 255), cv.FILLED)
        
    if show: 
        imshow(dilate)
        imshow(img_copy)
        
    homography = cv.getPerspectiveTransform(np.float32(vertices), np.float32(new_vertices))
    block = cv.warpPerspective(img, homography, dsize=(int(np.max(new_vertices[:, 0])), 
                                                                int(np.max(new_vertices[:, 1]))))
    rows, cols = block.shape
    r = rows * percentege
    c = cols * percentege

    r, c = int(r // 2), int(c // 2)

    return block[r: rows - r, :cols - c]    