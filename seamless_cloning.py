# Importaciones necesarias
import cv2
import numpy as np 
 
# Leer imágenes
src = cv2.imread("person/pull-person/0000063_06000_d_0000007_person_01.jpg")
dst = cv2.imread("input/test-a.jpg")
 
# Crea una máscara aproximada alrededor del objeto.
src_mask = np.zeros(src.shape, src.dtype)
#poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
poly = np.array([
	[0, 0],
    [85, 0],
    [160, 85],
    [0, 160]
], np.int32)
cv2.fillPoly(src_mask, [poly], (255, 255, 255))
 
# Aquí es donde se colocará el CENTRO del avión.
center = (460,340)
 
# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
 
# Guardar resultados
cv2.imwrite("output/opencv-seamless-cloning-example.jpg", output)