# Importación de librerías necesarias
import cv2           
import numpy as np   
import os            
import glob          

class PersonDetector:
    """
    Clase para detectar y recortar personas en imágenes usando OpenCV
    """
    
    def __init__(self):
        """
        Inicializar el detector de personas con HOG y validadores adicionales
        """
        # Inicializar el detector HOG de OpenCV
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Intentar cargar detectores DNN más precisos como respaldo
        self.dnn_net = None
        try:
            # Cargar modelo YOLO pre-entrenado si está disponible
            # Este es más preciso pero requiere archivos adicionales
            print("🔍 Buscando modelos DNN adicionales...")
            
        except Exception as e:
            print("ℹ️ Usando solo detector HOG (recomendado para precisión)")
        
        # Parámetros de validación estricta para personas
        self.min_person_area = 1500      # Área mínima en píxeles
        self.max_person_area = 200000    # Área máxima en píxeles  
        self.min_aspect_ratio = 1.2      # Relación altura/ancho mínima (personas son más altas)
        self.max_aspect_ratio = 4.0      # Relación altura/ancho máxima
        
        print("Detector de personas con validación estricta inicializado")
    
    def validate_person_detection(self, image, bbox):
        """
        Validar si una detección corresponde realmente a una persona
        
        Args:
            image: imagen completa
            bbox: (x, y, w, h) de la detección
            
        Returns:
            (is_valid, confidence_score, reasons)
        """
        x, y, w, h = bbox
        reasons = []
        confidence_score = 0.0
        
        # 1. Validar dimensiones básicas
        area = w * h
        aspect_ratio = h / w if w > 0 else 0
        
        if area < self.min_person_area:
            reasons.append(f"Área muy pequeña ({area} < {self.min_person_area})")
            return False, 0.0, reasons
            
        if area > self.max_person_area:
            reasons.append(f"Área muy grande ({area} > {self.max_person_area})")
            return False, 0.0, reasons
            
        if aspect_ratio < self.min_aspect_ratio:
            reasons.append(f"Muy ancho para ser persona ({aspect_ratio:.2f} < {self.min_aspect_ratio})")
            return False, 0.0, reasons
            
        if aspect_ratio > self.max_aspect_ratio:
            reasons.append(f"Muy alto para ser persona ({aspect_ratio:.2f} > {self.max_aspect_ratio})")
            return False, 0.0, reasons
            
        confidence_score += 0.3  # Pasó validaciones básicas
        
        # 2. Extraer región de interés
        roi = image[max(0, y):min(image.shape[0], y+h), 
                   max(0, x):min(image.shape[1], x+w)]
        
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            reasons.append("ROI inválida")
            return False, 0.0, reasons
        
        # 3. Análisis de características humanas
        human_features_score = self.analyze_human_features(roi)
        confidence_score += human_features_score * 0.4
        
        # 4. Análisis de gradientes (características HOG)
        gradient_score = self.analyze_gradients(roi)
        confidence_score += gradient_score * 0.3
        
        # 5. Umbral final
        is_valid = confidence_score > 0.6
        
        if is_valid:
            reasons.append(f"Persona válida (score: {confidence_score:.2f})")
        else:
            reasons.append(f"No es persona (score: {confidence_score:.2f})")
            
        return is_valid, confidence_score, reasons
    
    def analyze_human_features(self, roi):
        """
        Analizar características específicas humanas en la región
        """
        score = 0.0
        
        # Convertir a escala de grises para análisis
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Análisis de variación vertical (personas tienen variación vertical)
        vertical_profile = np.mean(gray, axis=1)
        vertical_variation = np.std(vertical_profile)
        if vertical_variation > 20:  # Suficiente variación vertical
            score += 0.3
        
        # 2. Análisis de simetría aproximada (torso humano es relativamente simétrico)
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        if left_half.shape == right_half.shape:
            right_flipped = np.fliplr(right_half)
            symmetry = np.corrcoef(left_half.flatten(), right_flipped.flatten())[0,1]
            if not np.isnan(symmetry) and symmetry > 0.3:
                score += 0.2
        
        # 3. Análisis de bordes (personas tienen bordes bien definidos)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        if 0.05 < edge_density < 0.3:  # Densidad de bordes típica de personas
            score += 0.3
        
        # 4. Análisis de regiones (cabeza, torso, piernas)
        if self.has_person_structure(gray):
            score += 0.2
        
        return min(1.0, score)
    
    def has_person_structure(self, gray):
        """
        Verificar si tiene estructura básica de persona (cabeza arriba, cuerpo abajo)
        """
        h, w = gray.shape
        
        # Dividir en tercios verticales
        top_third = gray[:h//3, :]
        middle_third = gray[h//3:2*h//3, :]
        
        # La parte superior (cabeza/hombros) suele ser más estrecha que el medio
        top_width_variance = np.var(np.sum(top_third > np.mean(top_third), axis=1))
        middle_width_variance = np.var(np.sum(middle_third > np.mean(middle_third), axis=1))
        
        # Estructura básica de persona: variación en el ancho a lo largo de la altura
        return top_width_variance > 10 and middle_width_variance > 10
    
    def analyze_gradients(self, roi):
        """
        Analizar gradientes para características humanas
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calcular gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitud y dirección de gradientes
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Las personas tienen gradientes con cierta distribución
        grad_mean = np.mean(magnitude)
        grad_std = np.std(magnitude)
        
        # Score basado en distribución de gradientes típica de personas
        if 10 < grad_mean < 80 and grad_std > 15:
            return 1.0
        elif 5 < grad_mean < 100 and grad_std > 10:
            return 0.7
        else:
            return 0.3

    def detect_persons_multiscale(self, image, min_confidence=0.5):
        """
        Detectar personas con validación estricta para eliminar falsos positivos
        
        Args:
            image: imagen de entrada (array de numpy)
            min_confidence: confianza mínima para considerar una detección válida
            
        Returns:
            Lista de rectángulos (x, y, w, h) donde se detectaron personas REALES
        """
        all_detections = []
        all_weights = []
        
        # Configuraciones más conservadoras para evitar falsos positivos
        detection_configs = [
            # Configuración 1: Detección estándar más estricta
            {'winStride': (8, 8), 'padding': (16, 16), 'scale': 1.05, 'hitThreshold': 0.0},
            # Configuración 2: Solo para personas claramente visibles
            {'winStride': (8, 8), 'padding': (24, 24), 'scale': 1.1, 'hitThreshold': 0.2},
        ]
        
        for config in detection_configs:
            try:
                rectangles, weights = self.hog.detectMultiScale(
                    image,
                    winStride=config['winStride'],
                    padding=config['padding'],
                    scale=config['scale'],
                    hitThreshold=config['hitThreshold'],
                    groupThreshold=2
                )
                
                if len(rectangles) > 0:
                    all_detections.extend(rectangles)
                    all_weights.extend(weights)
                    
            except Exception as e:
                print(f"Error en configuración: {e}")
                continue
        
        if len(all_detections) > 0:
            all_detections = np.array(all_detections)
            all_weights = np.array(all_weights)
            
            boxes = []
            scores = []
            for i, (x, y, w, h) in enumerate(all_detections):
                boxes.append([x, y, x + w, y + h])
                scores.append(all_weights[i])
            
            boxes = np.array(boxes, dtype=np.float32)
            scores = np.array(scores, dtype=np.float32)
            
            indices = cv2.dnn.NMSBoxes(boxes, scores, min_confidence, 0.3)
            
            candidate_detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, x2, y2 = boxes[i]
                    candidate_detections.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
            
            validated_detections = []
            for bbox in candidate_detections:
                is_valid, confidence, reasons = self.validate_person_detection(image, bbox)
                
                if is_valid:
                    validated_detections.append(bbox)
            
            return validated_detections
        else:
            return []

    def detect_persons(self, image, min_confidence=0.5):
        """
        Método principal de detección para personas reales
        """
        detections = self.detect_persons_multiscale(image, min_confidence)
        
        if len(detections) > 1:
            final_detections = self.remove_duplicate_detections(detections, overlap_threshold=0.4)
        else:
            final_detections = detections
            
        return final_detections
    
    def preprocess_image(self, image):
        """
        Pre-procesar imagen para mejorar la detección
        """
        processed_images = [image]  # Imagen original
        
        # 1. Imagen con mejor contraste
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        processed_images.append(enhanced)
        
        # 2. Imagen con ecualización de histograma
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        processed_images.append(equalized)
        
        return processed_images
    
    def remove_duplicate_detections(self, detections, overlap_threshold=0.4):
        """
        Eliminar detecciones duplicadas con umbral más estricto
        """
        if len(detections) <= 1:
            return detections
            
        boxes = np.array([[x, y, x+w, y+h] for x, y, w, h in detections], dtype=np.float32)
        scores = np.ones(len(detections), dtype=np.float32)
        
        # NMS más estricto para evitar personas muy cercanas duplicadas
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.1, overlap_threshold)
        
        if len(indices) > 0:
            final_detections = []
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]
                final_detections.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
            return final_detections
        else:
            return detections
    
    def crop_persons_smart(self, image, detections, tight_crop=True):
        """
        Recortar personas exactamente al tamaño del bounding box
        """
        cropped_persons = []
        height, width = image.shape[:2]
        
        for i, (x, y, w, h) in enumerate(detections):
            if tight_crop:
                # Recorte exacto al bounding box con padding mínimo
                padding = 5
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)
            else:
                # Sin padding, exactamente el bounding box
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width, x + w)
                y2 = min(height, y + h)
            
            crop_width = x2 - x1
            crop_height = y2 - y1
            
            if crop_width < 20 or crop_height < 40:
                continue
            
            person_crop = image[y1:y2, x1:x2]
            
            if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                cropped_persons.append({
                    'image': person_crop,
                    'bbox': (x1, y1, crop_width, crop_height),
                    'original_bbox': (x, y, w, h),
                    'person_id': i + 1
                })
        
        return cropped_persons
    
    def validate_crop_quality(self, crop):
        """
        Validar la calidad del recorte de persona
        """
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return False
        
        # Verificar que no sea completamente negro o blanco
        mean_intensity = np.mean(crop)
        if mean_intensity < 10 or mean_intensity > 245:
            return False
        
        # Verificar que tenga suficiente variación (no sea uniforme)
        std_intensity = np.std(crop)
        if std_intensity < 15:
            return False
        
        return True
    
    def calculate_crop_confidence(self, crop):
        """
        Calcular confianza del recorte basado en características visuales
        """
        # Características simples para calcular confianza
        variance = np.var(crop)
        edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_density = np.sum(edges > 0) / (crop.shape[0] * crop.shape[1])
        
        # Combinar métricas (normalizar entre 0-1)
        confidence = min(1.0, (variance / 10000 + edge_density * 10) / 2)
        return confidence
    
    def enhance_person_crop(self, crop):
        """
        Mejorar la calidad visual del recorte de persona
        """
        # 1. Mejorar contraste
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 2. Reducir ruido ligeramente
        denoised = cv2.bilateralFilter(enhanced, 5, 20, 20)
        
        # 3. Aumentar nitidez suavemente
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened

    def crop_persons(self, image, detections, padding=20):
        """
        Método legacy para compatibilidad - usa el nuevo método inteligente
        """
        return self.crop_persons_smart(image, detections, adaptive_padding=True)
    
    def draw_detections(self, image, detections):
        """
        Dibujar rectángulos alrededor de las personas detectadas
        
        Args:
            image: imagen donde dibujar
            detections: lista de detecciones
            
        Returns:
            Imagen con las detecciones dibujadas
        """
        result_image = image.copy()
        
        for i, (x, y, w, h) in enumerate(detections):
            # Dibujar rectángulo verde alrededor de la persona
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Agregar etiqueta con número de persona
            label = f"Persona {i + 1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Fondo para el texto
            cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), (0, 255, 0), -1)
            
            # Texto
            cv2.putText(result_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return result_image
    
    def process_image(self, image_path, output_dir):
        """
        Procesar una imagen individual: detectar, recortar y guardar personas
        
        Args:
            image_path: ruta de la imagen a procesar
            output_dir: directorio donde guardar los recortes
            
        Returns:
            Número de personas detectadas
        """
        print(f"Procesando: {os.path.basename(image_path)}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo cargar la imagen {image_path}")
            return 0
        
        detections = self.detect_persons(image, min_confidence=0.6)
        
        if len(detections) == 0:
            print(f"No se detectaron personas en {os.path.basename(image_path)}")
            return 0
        
        print(f"Detectadas {len(detections)} persona(s)")
        
        cropped_persons = self.crop_persons_smart(image, detections, tight_crop=True)
        
        if len(cropped_persons) == 0:
            print("Ningun recorte validado")
            return 0
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        saved_count = 0
        for person_data in cropped_persons:
            person_image = person_data['image']
            person_id = person_data['person_id']
            bbox = person_data['bbox']
            
            output_filename = f"{base_name}_person_{person_id:02d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            if cv2.imwrite(output_path, person_image):
                crop_w, crop_h = bbox[2], bbox[3]
                print(f"Guardado: {output_filename} ({crop_w}x{crop_h}px)")
                saved_count += 1
            else:
                print(f"Error guardando: {output_filename}")
        
        # Guardar imagen con detecciones dibujadas
        annotated_image = self.draw_detections(image, detections)
        annotated_filename = f"{base_name}_detections.jpg"
        annotated_path = os.path.join(output_dir, annotated_filename)
        cv2.imwrite(annotated_path, annotated_image)
        print(f"Guardada imagen con detecciones: {annotated_filename}")
        
        return saved_count
    
    def process_folder(self, input_folder="person", output_subfolder="pull-person"):
        """
        Procesar todas las imágenes de una carpeta
        
        Args:
            input_folder: carpeta que contiene las imágenes
            output_subfolder: subcarpeta donde guardar los resultados
        """
        print("Iniciando detección y recorte de personas")
        print("=" * 60)
        
        # Verificar que existe la carpeta de entrada
        if not os.path.exists(input_folder):
            print(f"Error: La carpeta '{input_folder}' no existe")
            print(f"Crea la carpeta '{input_folder}' y coloca las imágenes ahí")
            return
        
        # Crear carpeta de salida
        output_dir = os.path.join(input_folder, output_subfolder)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directorio de salida: {output_dir}")
        
        # Buscar archivos de imagen
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        
        for extension in image_extensions:
            pattern = os.path.join(input_folder, extension)
            image_files.extend(glob.glob(pattern))
            # También buscar extensiones en mayúsculas
            pattern_upper = os.path.join(input_folder, extension.upper())
            image_files.extend(glob.glob(pattern_upper))
        
        if not image_files:
            print(f"No se encontraron imágenes en la carpeta '{input_folder}'")
            print("Formatos soportados: JPG, JPEG, PNG, BMP, TIFF")
            return
        
        print(f"Encontradas {len(image_files)} imagen(es) para procesar")
        print("-" * 40)
        
        # Procesar cada imagen
        total_persons = 0
        processed_images = 0
        
        for image_path in image_files:
            try:
                persons_count = self.process_image(image_path, output_dir)
                total_persons += persons_count
                processed_images += 1
                print()  # Línea en blanco para separar
                
            except Exception as e:
                print(f"❌ Error procesando {os.path.basename(image_path)}: {e}")
                continue
        
        # Resumen final
        print("=" * 60)
        print("RESUMEN FINAL:")
        print(f"Imágenes procesadas: {processed_images}/{len(image_files)}")
        print(f"Total personas detectadas: {total_persons}")
        print(f"Recortes guardados en: {output_dir}")
        print("Proceso completado!")
        
        if total_persons > 0:
            print(f"\n💡 Revisa la carpeta '{output_subfolder}' dentro de '{input_folder}' para ver los resultados")

def main():
    """
    Función principal del programa
    """
    print("DETECTOR DE PERSONAS - OpenCV")
    print("Detecta y recorta personas de imágenes automáticamente")
    print("=" * 60)
    
    # Crear instancia del detector
    detector = PersonDetector()
    
    # Procesar todas las imágenes de la carpeta 'person'
    # Los recortes se guardarán en 'person/pull-person'
    detector.process_folder(
        input_folder="person",           
        output_subfolder="pull-person"  
    )

if __name__ == "__main__":
    main()