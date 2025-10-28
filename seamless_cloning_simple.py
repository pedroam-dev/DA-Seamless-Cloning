# Importaciones necesarias
import cv2
import numpy as np
import os

def resize_image_to_target(image, target_width=960, target_height=540):
    """
    Redimensionar imagen a resolución objetivo manteniendo proporción
    """
    h, w = image.shape[:2]
    
    # Calcular escala para mantener proporción
    scale = min(target_width/w, target_height/h)
    
    # Nuevas dimensiones
    new_width = int(w * scale)
    new_height = int(h * scale)
    
    # Redimensionar
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Crear canvas del tamaño objetivo
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Centrar imagen en canvas
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas

# CONFIGURACIÓN - Modifica estas rutas según tus archivos
src_image_path = "person/pull-person/0000063_06000_d_0000007_person_01.jpg"      
dst_image_path = "input/test-a.jpg"          
output_path = "output/test-a1.jpg" 

# Resolución objetivo
target_width = 960
target_height = 540

print("Seamless Cloning con Redimensionamiento")
print("="*50)
print(f"Imagen fuente: {src_image_path}")
print(f"Imagen destino: {dst_image_path}")
print(f"Resolución objetivo: {target_width}x{target_height}")

# Leer imágenes
src = cv2.imread(src_image_path)
dst = cv2.imread(dst_image_path)

if src is None:
    print(f"Error: No se pudo cargar {src_image_path}")
    exit(1)
    
if dst is None:
    print(f"Error: No se pudo cargar {dst_image_path}")
    exit(1)

print(f"Imagen fuente original: {src.shape[1]}x{src.shape[0]}")
print(f"Imagen destino original: {dst.shape[1]}x{dst.shape[0]}")

# Redimensionar imagen destino a 960x540
dst_resized = resize_image_to_target(dst, target_width, target_height)
print(f"Imagen destino redimensionada: {dst_resized.shape[1]}x{dst_resized.shape[0]}")

# Crear máscara alrededor del objeto a clonar
src_mask = np.zeros(src.shape, src.dtype)

# Coordenadas del polígono que define el objeto (ajusta según tu imagen)
poly = np.array([
    [0, 0],
    [85, 0],
    [160, 85],
    [0, 160]
], np.int32)

# Rellenar polígono en la máscara
cv2.fillPoly(src_mask, [poly], (255, 255, 255))

# Definir centro donde se colocará el objeto (centro de la imagen redimensionada)
center = (target_width // 2, target_height // 2)
print(f"Centro de clonado: {center}")

# Realizar seamless cloning
print("Realizando seamless cloning...")
output = cv2.seamlessClone(src, dst_resized, src_mask, center, cv2.NORMAL_CLONE)

# Crear directorio de salida si no existe
os.makedirs("output", exist_ok=True)

# Guardar resultado
if cv2.imwrite(output_path, output):
    print(f"Resultado guardado en: {output_path}")
    print("Proceso completado exitosamente")
else:
    print(f"Error al guardar: {output_path}")

# Opcional: También guardar el fondo redimensionado para referencia
background_path = "output/background_resized.jpg"
cv2.imwrite(background_path, dst_resized)
print(f"Fondo redimensionado guardado en: {background_path}")