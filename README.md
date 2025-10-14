# DA-Seamless-Cloning

Un proyecto de clonado sin costuras (seamless cloning) usando OpenCV para combinar imágenes de manera natural y realista.

## Descripción

Este proyecto implementa la técnica de **seamless cloning** utilizando OpenCV para insertar objetos de una imagen en otra de forma que se vean naturalmente integrados. La técnica utiliza algoritmos avanzados de procesamiento de imágenes para hacer que la fusión sea imperceptible.

## Características

- Clonado sin costuras de objetos entre imágenes
- Uso del algoritmo `NORMAL_CLONE` de OpenCV
- Creación automática de máscaras para definir áreas de clonado
- Procesamiento eficiente de imágenes

## Tecnologías Utilizadas

- **Python 3.x**
- **OpenCV** (`cv2`) - Para procesamiento de imágenes
- **NumPy** - Para manipulación de arrays y operaciones matemáticas

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instaladas las siguientes dependencias:

```bash
pip install opencv-python
pip install numpy
```

## Estructura del Proyecto

```
DA-Seamless-Cloning/
├── seamless_cloning.py          # Script principal
├── images/
│   ├── airplane.jpg             # Imagen fuente (objeto a clonar)
│   ├── sky.jpg                  # Imagen destino
│   └── opencv-seamless-cloning-example.jpg  # Resultado generado
└── README.md                    # Este archivo
```

## Uso

1. **Preparar las imágenes:**
   - Coloca tu imagen fuente en `images/airplane.jpg`
   - Coloca tu imagen destino en `images/sky.jpg`

2. **Ejecutar el script:**
   ```bash
   python seamless_cloning.py
   ```

3. **Ver el resultado:**
   - La imagen resultante se guardará como `images/opencv-seamless-cloning-example.jpg`

## Personalización

### Modificar la máscara

Para cambiar el área que se va a clonar, modifica los puntos del polígono en el código:

```python
poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
```

### Cambiar la posición de destino

Para cambiar dónde se colocará el objeto clonado, modifica las coordenadas del centro:

```python
center = (800,100)  # (x, y)
```

### Modos de clonado

OpenCV ofrece diferentes modos de clonado:
- `cv2.NORMAL_CLONE` - Clonado normal (por defecto)
- `cv2.MIXED_CLONE` - Clonado mixto
- `cv2.MONOCHROME_TRANSFER` - Transferencia monocromática

## Ejemplo de Resultado

El proyecto toma un avión de una imagen y lo coloca seamlessly en una imagen de cielo, creando un resultado natural y realista.

## Contribuir

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crea una rama para tu característica (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Autor

**Pedro AM**
- GitHub: [@pedroam](https://github.com/pedroam)
Inspirado en técnicas de computer vision modernas

---

⭐ Si este proyecto te ha sido útil, ¡dale una estrella!