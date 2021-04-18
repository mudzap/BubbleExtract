import cv2
import numpy as np
import glob
import os


# Filtros!
def filter_method_laplacian(np_image_set, morph_sz):
    for index, image in enumerate(np_image_set):
        #Laplaciano
        np_image_set[index] = cv2.Laplacian(image, cv2.CV_64F)
        np_image_set[index] = np.maximum(0, np_image_set[index])

        #Soft-binarize (para establecer regiones)
        np_image_set[index] =  np.divide(np_image_set[index], np.max(np_image_set[index])) #Normaliza
        np_image_set[index] = np.asarray(np_image_set[index]*255, dtype=np.uint8)

        #Simple threshold, se adquiere manualmente los maximos y minimos en caso de no haber
        #preprocesamiento previo
        im_max = np.amax(np_image_set[index])
        im_min = np.amin(np_image_set[index])
        threshold_simple = (im_max - im_min)*0.1
        ret, np_image_set[index] = cv2.threshold(np_image_set[index], threshold_simple, im_max, cv2.THRESH_BINARY)

        #Dilatar para prevener "escapes"
        h_d = np.ones((morph_sz, morph_sz), np.double)
        np_image_set[index] = cv2.dilate(np_image_set[index], h_d)
        
        
def filter_method_thresh(np_image_set, morph_sz):
    for index, image in enumerate(np_image_set):
        #Simple threshold, se adquiere manualmente los maximos y minimos en caso de no haber
        #preprocesamiento previo
        im_max = np.amax(np_image_set[index])
        im_min = np.amin(np_image_set[index])
        threshold_simple = (im_max - im_min)*0.7
        ret, np_image_set[index] = cv2.threshold(np_image_set[index], threshold_simple, im_max, cv2.THRESH_BINARY_INV)

        #Dilatar para prevener "escapes"
        h_d = np.ones((morph_sz, morph_sz), np.uint8)
        np_image_set[index] = cv2.dilate(np_image_set[index], h_d)


def filter_method_canny(np_image_set, morph_sz):
    for index, image in enumerate(np_image_set):
        im_max = np.amax(np_image_set[index])
        im_min = np.amin(np_image_set[index])
        threshold_simple = (im_max - im_min)*0.7        
        np_image_set[index] = cv2.Canny(np_image_set[index], 100, 127, 3)
        
        #Dilatar para prevener "escapes"
        h_d = np.ones((morph_sz, morph_sz), np.double)
        np_image_set[index] = cv2.dilate(np_image_set[index], h_d)
        
     
# Muestreo
# Es posible determinar el numero de puntos con el radio aspecto original de la imagen!
# Optar por esto eventualmente
def get_sampling_points(sz, points):
    step = sz/points
    offset = step/2
    point_array = []

    # EL -1 SI ES ESPERADO, DEBIDO AL OFFSET
    for i in range(0, points[0]-1):
        for j in range(0, points[1]-1):
            new_point = offset + (step[1]*j, step[0]*i)
            point_array.append(new_point)

    return np.array(point_array, dtype=np.uint32)


def get_speech_bubble_candidates(original_set, filtered_set, out_set, res_xy, thresh):
    point_res = np.array(res_xy)    

    for index, image in enumerate(filtered_set):
        points = get_sampling_points(image.shape, point_res)

        ## La mascarilla debe de tener una 'pestana' de 1 pixel
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
                    cv2.FLOODFILL_MASK_ONLY# | cv2.FLOODFILL_FIXED_RANGE
                )

                #Si rellena algun pixel, cuenta como relleno
                if(ret > 0):
                    #print(ret)
                    current_fill += 1

                #Si rellena mas de cierto threshold, procesa y almacena la region
                if(ret > 1000 and ret < 200000):
                    x, y, w, h = rect[0], rect[1], rect[2], rect[3]

                    #Dibuja rectangulos en la imagen original
                    #out_image[index] = cv2.rectangle(out_image[index], (x, y), (x + w, y + h), (127, 0, 0), 2)

                    #Identifica los cambios en la mascara (nueva mascara solo de nuevos cambios)
                    out_mask = cv2.bitwise_xor(mask, old_mask)

                    #Dilata para intentar mantener el area original a detectar
                    h_d = np.ones((5, 5), np.uint8)
                    out_mask = cv2.dilate(out_mask, h_d)

                    #Copia la mascarilla para el relleno de hoyos
                    out_mask_copy = np.copy(out_mask)

                    #Ajusta tamano de la mascarilla a aquel de la imagen
                    mask_sz = out_mask.shape
                    out_mask = out_mask[1:mask_sz[0]-1, 1:mask_sz[1]-1]

                    #Rellena hoyos en la mascara
                    _ret, out_mask, out_mask_copy, _rect = cv2.floodFill(out_mask, out_mask_copy, (0,0), 255)
                    #cv2.imshow('mask'+str(current_fill)+str(index), out_mask)
                    #out_mask_inv = cv2.bitwise_not(out_mask)
                    out_mask = cv2.bitwise_not(out_mask)
                    #out_mask = cv2.bitwise_or(out_mask, out_mask_inv)
                    #cv2.imshow('mask_inv'+str(current_fill)+str(index), out_mask_inv)                

                    #Extrae solo nuevos elementos de la mascarilla de la imagen y los almacena
                    #en la region de tamano minimo necesario
                    out_region = cv2.bitwise_and(original_set[index], original_set[index], mask=out_mask)
                    out_region = out_region[y:y+h, x:x+w]

                    #Extrae region donde no hay informacion util
                    out_mask = out_mask[y:y+h, x:x+w]
                    out_mask = cv2.bitwise_not(out_mask)

                    #out_region = cv2.cvtColor(out_region, cv2.COLOR_BGR2GRAY)
                    out_region_2ch = np.dstack((out_region, out_region, out_mask))
                    out_set.append(out_region_2ch)

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
