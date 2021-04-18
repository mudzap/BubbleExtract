import cv2
import numpy as np
import glob
import os

# Helper functions for preprocessing
# Typically, hard-coded values are utilized
# These are not extensively tested, they work
# howver, for this case

# Preprocessing functions by different methods
#    Laplacian: Uses laplacian filter and a "soft-binarization"
#        for border identification
#    Canny: Uses Canny's border detection algorithm
#    Threshold: Simply thresholds the image
#
#    A preprocessed image that makes use of the complete dynamic
#    range is recommended (0-255)
def filter_method_laplacian(np_image_set, morph_sz):
    for index, image in enumerate(np_image_set):
        #Laplacian
        np_image_set[index] = cv2.Laplacian(image, cv2.CV_64F)
        np_image_set[index] = np.maximum(0, np_image_set[index])

        #Soft-binarize
        np_image_set[index] =  np.divide(np_image_set[index], np.max(np_image_set[index])) #Normaliza
        np_image_set[index] = np.asarray(np_image_set[index]*255, dtype=np.uint8)

        #Simple threshold
        im_max = np.amax(np_image_set[index])
        im_min = np.amin(np_image_set[index])
        threshold_simple = (im_max - im_min)*0.1
        ret, np_image_set[index] = cv2.threshold(
            np_image_set[index], threshold_simple,
            im_max, cv2.THRESH_BINARY)

        #Dilates in order to prevent "leaks"
        h_d = np.ones((morph_sz, morph_sz), np.double)
        np_image_set[index] = cv2.dilate(np_image_set[index], h_d)
        
        
def filter_method_thresh(np_image_set, morph_sz):
    for index, image in enumerate(np_image_set):
        #Simple threshold
        im_max = np.amax(np_image_set[index])
        im_min = np.amin(np_image_set[index])
        threshold_simple = (im_max - im_min)*0.7
        ret, np_image_set[index] = cv2.threshold(
            np_image_set[index], threshold_simple,
            im_max, cv2.THRESH_BINARY_INV)

        #Dilates in order to prevent leaks
        h_d = np.ones((morph_sz, morph_sz), np.uint8)
        np_image_set[index] = cv2.dilate(np_image_set[index], h_d)


def filter_method_canny(np_image_set, morph_sz):
    for index, image in enumerate(np_image_set):
        im_max = np.amax(np_image_set[index])
        im_min = np.amin(np_image_set[index])
        threshold_simple = (im_max - im_min)*0.7        
        np_image_set[index] = cv2.Canny(np_image_set[index], 100, 127, 3)
        
        h_d = np.ones((morph_sz, morph_sz), np.double)
        np_image_set[index] = cv2.dilate(np_image_set[index], h_d)
        
     
# Sampling
# Of course you can determine the number of sampling points from the
# image aspect ratio, currently I find no need to do so.

# Gets pixels from which a speech bubble candidate will be proposed
def get_sampling_points(sz, points):
    step = sz/points
    offset = step/2
    point_array = []

    for i in range(0, points[0]-1):
        for j in range(0, points[1]-1):
            new_point = offset + (step[1]*j, step[0]*i)
            point_array.append(new_point)

    return np.array(point_array, dtype=np.uint32)

# Gets candidates and stores them in out_set, these are extracted from the
# original_set, but masking is done depending on the filtered_set
# res_xy specifies the grid resolution from which sampling will be done
# thresh is mostly redundant, but I recommend keeping it at 127
def get_speech_bubble_candidates(original_set, filtered_set, out_set, res_xy, thresh=127):
    point_res = np.array(res_xy)    

    for index, image in enumerate(filtered_set):
        points = get_sampling_points(image.shape, point_res)

        ## Required for floodFill
        mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
        
        current_fill = 1
        for p in points:
            if(image[p[1], p[0]] < thresh):
                old_mask = np.copy(mask)

                ret, filled_area, mask, rect = cv2.floodFill(
                    image,
                    mask,
                    (p[0], p[1]),
                    255,
                    127, 20,
                    cv2.FLOODFILL_MASK_ONLY# + cv2.FLOODFILL_FIXED_RANGE
                )

                #If any filling is done, tracks it
                if(ret > 0):
                    #print(ret)
                    current_fill += 1

                # Excludes fillings when either too few pixels or too much
                # are filled (should just use a tuple really)
                if(ret > 1000 and ret < 200000):
                    x, y, w, h = rect[0], rect[1], rect[2], rect[3]

                    # Draw rectangles in original image
                    # Useful for illustrative purposes
                    #out_image[index] = cv2.rectangle(out_image[index], (x, y), (x + w, y + h), (127, 0, 0), 2)

                    # Makes output mask only from changes in the mask
                    out_mask = cv2.bitwise_xor(mask, old_mask)

                    # Dilates to return to original area (from preprocessing)
                    h_d = np.ones((5, 5), np.uint8)
                    out_mask = cv2.dilate(out_mask, h_d)

                    # Fill holes in mask
                    out_mask_copy = np.copy(out_mask)
                    mask_sz = out_mask.shape
                    out_mask = out_mask[1:mask_sz[0]-1, 1:mask_sz[1]-1]
                    _ret, out_mask, out_mask_copy, _rect = cv2.floodFill(out_mask, out_mask_copy, (0,0), 255)
                    out_mask = cv2.bitwise_not(out_mask)

                    # Extracts elements in a rectangle with the minimum size required
                    out_region = cv2.bitwise_and(original_set[index], original_set[index], mask=out_mask)
                    out_region = out_region[y:y+h, x:x+w]

                    # Extracts "empty" region
                    out_mask = out_mask[y:y+h, x:x+w]
                    out_mask = cv2.bitwise_not(out_mask)

                    # Outputs as a 3 channel image with the following:
                    #   Original image data, masked
                    #   Ditto
                    #   Masked out area
                    out_region_2ch = np.dstack((out_region, out_region, out_mask))
                    out_set.append(out_region_2ch)

# This function is only meant to extract everything
# for its further manual classification, since I
# lacked a data set.
def get_data_main():
    # Cargar imagenes
    image_set_gs = []
    os.chdir("examples")
    for file in glob.glob("*.jpg"):
        image = cv2.imread(file)
        image_set_gs.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        print("Reading: " + file)

    #Almacenar original para mostrarlo despues
    np_original = np.array(image_set_gs)
    np_image_set = np.copy(np_original)

    filter_method_canny(np_image_set, 3)

    bubble_set = []
    res_xy = (32, 64)
    BUBBLE_SAMPLE_THRESH = 127 

    get_speech_bubble_candidates(np_original, np_image_set, bubble_set, res_xy, BUBBLE_SAMPLE_THRESH)
    bubble_set = np.array(bubble_set)

    i = 0
    for image in bubble_set:
        cv2.imwrite("output/a" + str(i) + ".png", image)
        i += 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("EOP")
