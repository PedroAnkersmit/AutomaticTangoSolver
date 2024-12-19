import flet as ft
import cv2
import numpy as np
from scipy import signal #Para el suavizado gaussiano
import math
import matplotlib.pyplot as plt
import glob
images_path = './src/images/'
import base64

def to_base64(image):
    base64_image = cv2.imencode('.png', image)[1]
    base64_image = base64.b64encode(base64_image).decode('utf-8') 
    return base64_image

def main(page):
    page.title = "Automatic Tango Solver"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.MainAxisAlignment.CENTER


    # Create a blank image for the initial display,
    # image element does not support None for src_base64
    init_image = np.zeros((480, 640, 3), dtype=np.uint8) + 128
    init_base64_image = to_base64(init_image)

    image_src = ft.Image(src_base64=init_base64_image, width=640, height=480)
    image_dst = ft.Image(src_base64=init_base64_image, width=640, height=480)

    images_row = ft.Row([image_src,image_dst], alignment=ft.MainAxisAlignment.CENTER)
    
    image = None
    init_pos =[]
    def gaussian_smoothing(image, sigma, w_kernel):
    # Define 1D kernel
        s=sigma
        w=w_kernel
        kernel_1D = np.array( np.float32([(1/(s*np.sqrt(2*np.pi)))*np.exp(-((z*z)/(2*s*s))) for z in range(-w,w+1)]))
    
    # Apply distributive property of convolution
        vertical_kernel = kernel_1D.reshape(2*w+1,1)
        horizontal_kernel = kernel_1D.reshape(1,2*w+1)   
        gaussian_kernel_2D = signal.convolve2d(vertical_kernel, horizontal_kernel)   
    
    # Blur image
        smoothed_img = cv2.filter2D(image,cv2.CV_8U,gaussian_kernel_2D)
    
    # Normalize to [0 254] values
        smoothed_norm = np.array(image.shape)
        smoothed_norm = cv2.normalize(smoothed_img, None, -255, 255, cv2.NORM_MINMAX) # Leave the second argument as None
    
        return smoothed_norm
    
    def getCuadrados(image):
        height, width = image.shape

        cuadrados = []

        for i in range(0, 6):

            fila = []

            for j in range(0, 6):
                l = int((i * (width / 6)))
                r = int((i + 1) * (width / 6))
                t = int((j * (height / 6)))
                b = int((j + 1) * (height / 6))

                cuadrado = image[t:b, l:r]
                fila.append(cuadrado)

            cuadrados.append(fila)

        return cuadrados

    def eliminarBordes(image):
        # Dibujar un rectangulo en todos los lados menos las localizaciones de las posibles X o =

        inicios = [
            (0, 0),
            (0, 0),
            (0, 85),
            (85, 0),
            (130, 85),
            (85, 130),
            (0, 130),
            (130, 0),
        ]
        finales = [
            (10, 55),
            (55, 10),
            (10, 140),
            (140, 10),
            (140, 140),
            (140, 140),
            (55, 140),
            (140, 55),
        ]
        for i in range(0, 8):
            image = cv2.rectangle(image, inicios[i], finales[i], (255, 255, 255), -1)
    # Ahora hay que construir la matriz resuelta
    def construirMatriz(tipos):
        matriz = np.zeros((6, 6))
        for c in range(0, 6):
            for f in range(0, 6):
                if tipos[c][f][0] != 3:
                    matriz[c, f] = tipos[c][f][0]
                else:
                    matriz[c, f] = 0
        # Matriz del estado original del tablero
        return matriz    
    def getModificador(i):
        izq = der = arr = aba = 0
        if i == 0:
            aba = 1
        elif i == 1:
            izq = 1
            aba = 1
        elif i == 2:
            der = 1
            aba = 1
        elif i == 3:
            aba = 1
            arr = -1
        elif i == 4:
            izq = 1
        elif i == 5:
            izq = 1
            aba = -1
        elif i == 6:
            izq = 1
            arr = -1
        elif i == 7:
            der = 1
        elif i == 8:
            der = 1
            aba = -1
        elif i == 9:
            der = 1
            arr = -1
        elif i == 10:
            arr = 1
        elif i == 11:
            arr = 1
            izq = 1
        elif i == 12:
            arr = 1
            der = 1
        elif i == 13:
            arr = 1
            aba = -1
            # El catorce es el caso "normal"
        elif i == 15:
            aba = -1
        elif i == 16:
            aba = -1
            izq = -1
        elif i == 17:
            aba = -1
            der = -1
        elif i == 18:
            izq = -1
        elif i == 19:
            der = -1
        elif i == 20:
            arr = -1
        elif i == 21:
            arr = -1
            izq = -1
        elif i == 22:
            arr = -1
            der = -1
        else:
            izq = 0
            der = 0
            arr = 0
            aba = 0
        return [izq, der, arr, aba]
    def getNCC(c):
        tipo = 0
        # Primero comprobamos si es Sol, luna o en blanco
        eliminarBordes(c)
        bTemplate = cv2.imread(images_path + "tiles/blank.jpeg")
        bTemplate = cv2.cvtColor(bTemplate, cv2.COLOR_BGR2GRAY)
        mTemplate = cv2.imread(images_path + "tiles/moon.jpeg")
        mTemplate = cv2.cvtColor(mTemplate, cv2.COLOR_BGR2GRAY)
        sTemplate = cv2.imread(images_path + "tiles/sun.jpeg")
        sTemplate = cv2.cvtColor(sTemplate, cv2.COLOR_BGR2GRAY)

        nccB = cv2.matchTemplate(c, bTemplate, cv2.TM_CCORR_NORMED)
        nccM = cv2.matchTemplate(c, mTemplate, cv2.TM_CCORR_NORMED)
        nccS = cv2.matchTemplate(c, sTemplate, cv2.TM_CCORR_NORMED)
        vB = np.amax(nccB)
        vM = np.amax(nccM)
        vS = np.amax(nccS)

        if vB > vM and vB > vS:
            tipo = 3
        elif vM > vB and vM > vS:
            tipo = 2
        elif vS > vB and vS > vM:
            tipo = 1
        return tipo
    def getTipo(c, col, fil):

        # Código numérico de los tipos
        # [relleno][izquierda][derecha][arriba][abajo][valor NCC máximo]
        # 1 = Sol, 2 = Luna, 3 = Desconocido (en blanco)
        # 1 = Igual, -1 = Distinto, 0 = Vacío
        tipo = getNCC(c)

        # Una vez extraído el tipo de la casilla, podemos eliminar el posible sol o luna para poder comprobar sus bordes
        ct = np.copy(c)
        # Si es un Sol o una Luna, vamos a tapar el icono para que funcione mejor la NCC
        if tipo != 3:
            ct = cv2.rectangle(ct, (18, 18), (130, 130), (255, 255, 255), -1)


        path = glob.glob(images_path + "tiles/Blanks/*.jpeg")
        i = 0
        index = 0
        valor = 0.0
        ncc_arr = []
        for img in path:
            t = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            t = gaussian_smoothing(t, 1, 1)
            eliminarBordes(t)
            _, t = cv2.threshold(t, 232, 255, cv2.THRESH_BINARY)  # Binarización
            ncc = cv2.matchTemplate(ct, t, cv2.TM_CCORR_NORMED)
            value = np.sum(ncc)
            ncc_arr.append(value)
            maxV = np.amax(ncc_arr)
            if valor < maxV:
                valor = maxV
                i = index
            index = index + 1

        izq, der, arr, aba = getModificador(i)

        if fil == 0:
            arr = 0
        if col == 0:
            izq = 0
        if fil == 5:
            aba = 0
        if col == 5:
            der = 0
        return [tipo, izq, der, arr, aba, valor]

    def congruenciaTipos(tipos):
        # Comprobamos discordancias
        for f in range(0, 6):
            for c in range(0, 6):
                if c < 5:
                    # Si no tienen el mismo modificador hacia la derecha
                    if tipos[f][c][2] != tipos[f][c + 1][1]:
                        # Comprobamos cuál tiene mayor valor NCC
                        if tipos[f][c][5] > tipos[f][c + 1][5]:
                            tipos[f][c + 1][1] = tipos[f][c][2]
                        elif tipos[f][c + 1][5] > tipos[f][c][5]:
                            tipos[f][c][2] = tipos[f][c + 1][1]
                if c > 0:
                    # Si no tienen el mismo modificador hacia la izquierda
                    if tipos[f][c][1] != tipos[f][c - 1][2]:
                        # Comprobamos cuál tiene mayor valor NCC
                        if tipos[f][c][5] > tipos[f][c - 1][5]:
                            tipos[f][c][2] = tipos[f][c][1]
                        elif tipos[f][c - 1][5] > tipos[f][c][5]:
                            tipos[f][c][1] = tipos[f][c - 1][2]
                if f < 5:
                    # Si no tienen el mismo modificador hacia abajo
                    if tipos[f][c][4] != tipos[f + 1][c][3]:
                        # Comprobamos cuál tiene mayor valor NCC
                        if tipos[f][c][5] > tipos[f + 1][c][5]:
                            tipos[f + 1][c][3] = tipos[f][c][4]
                        elif tipos[f + 1][c][5] > tipos[f][c][5]:
                            tipos[f][c][4] = tipos[f + 1][c][3]
                if f > 0:
                    # Si no tienen el mismo modificador hacia arriba
                    if tipos[f][c][3] != tipos[f - 1][c][4]:
                        # Comprobamos cuál tiene mayor valor NCC
                        if tipos[f][c][5] > tipos[f - 1][c][5]:
                            tipos[f - 1][c][4] = tipos[f][c][3]
                        elif tipos[f - 1][c][5] > tipos[f][c][5]:
                            tipos[f][c][3] = tipos[f - 1][c][4]
        tipos_sin_v = []
        for f in range(0, 6):
            col = []
            for c in range(0, 6):
                col.append(
                    [
                        tipos[f][c][0],
                        tipos[f][c][1],
                        tipos[f][c][2],
                        tipos[f][c][3],
                        tipos[f][c][4],
                    ]
                )
            tipos_sin_v.append(col)
        return tipos_sin_v
    def constructTipos(cuadrados):
        tipos = []

        for f in range(0, 6):
            fila = []
            for c in range(0, 6):
                cuad = cuadrados[c][f]
                tipo = getTipo(cuad, c, f)
                fila.append(tipo)
            tipos.append(fila)
        return tipos
    # Guardamos las posiciones de los símbolos iniciales
    def simbolosIniciales(matriz):
        init_pos = []
        for f in range(0, 6):
            for c in range(0, 6):
                if matriz[f][c] != 0:
                    init_pos.append([f, c])
        return init_pos
    def modificable(f, c):
        if [f, c] in init_pos:
            return False
        else:
            return True
    def generateImage(m):
        sun = cv2.imread(images_path + "tiles/sun_w.jpeg")
        moon = cv2.imread(images_path + "tiles/moon_w.jpeg")
        imagen_final = []
        for f in range(0, 6):
            col = []
            for c in range(0, 6):
                if m[f, c] == 1:
                    col.append(sun)
                elif m[f, c] == 2:
                    col.append(moon)
            col_img = np.hstack(col)
            imagen_final.append(col_img)

        imagen = np.vstack(imagen_final)
        return imagen

    def solve(e):
        nonlocal image
        if image is None:
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = gaussian_smoothing(image, 1, 1)  # Suavizado gaussiano
        _, thresh1 = cv2.threshold(smooth, 240, 255, cv2.THRESH_BINARY)  # Binarización
        contours, _ = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        marked = np.copy(image)
        cropped = np.copy(image)
        if len(contours) != 0:
            # Encontrar el mayor contorno (c) por el area
            c = max(contours, key=cv2.contourArea)
            # Extraer ese contorno, no es el que queremos
            contours = filter(lambda contour: contour.shape != c.shape, contours)
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            #Dibujamos un cuadrado sobre el contorno deseado
            cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = cropped[y + 3 : (y + h - 3), x + 3 : (x + w - 3)]
        _, threshold = cv2.threshold(cropped, 230, 255, cv2.THRESH_BINARY)
        smoothed = gaussian_smoothing(threshold, 1, 1)
        cuadrados = getCuadrados(smoothed)
        tipos = constructTipos(cuadrados)
        tipos = congruenciaTipos(tipos)
        #Construimos la matriz de valores
        m = construirMatriz(tipos)
        # Y resolvemos
        suma_fila = np.sum(m, axis = 1)
        suma_fila0 = suma_fila[0]
        suma_fila1 = suma_fila[1]
        suma_fila2 = suma_fila[2]
        suma_fila3 = suma_fila[3]
        suma_fila4 = suma_fila[4]
        suma_fila5 = suma_fila[5]
        suma_col = np.sum(m, axis = 0)
        suma_col0 = suma_col[0]
        suma_col1 = suma_col[1]
        suma_col2 = suma_col[2]
        suma_col3 = suma_col[3]
        suma_col4 = suma_col[4]
        suma_col5 = suma_col[5]
        while suma_fila0 != 9 or suma_fila1 != 9 or suma_fila2 != 9 or suma_fila3 != 9 or suma_fila4 != 9 or suma_fila5 != 9 or suma_col0 != 9 or suma_col1 != 9 or suma_col2 != 9 or suma_col3 != 9 or suma_col4 != 9 or suma_col5 != 9:
            #Paso 1: Comprobar los modificadores de las casillas con símbolos iniciales
            init_pos = simbolosIniciales(m)
            n_cas = len(init_pos)
            n_cas_aux = 0
            while n_cas != n_cas_aux:
                for i in range(0,n_cas):
                    f,c = init_pos[i]
                    if m[f][c] != 0:
                        if tipos[f][c][1] == 1 and modificable(f,c-1):
                            m[f][c-1] = m[f][c]
                        elif tipos[f][c][1] == -1 and modificable(f,c-1):
                            m[f][c-1] = 3-m[f][c]
                        if tipos[f][c][2] == 1 and modificable(f,c+1):
                            m[f][c+1] = m[f][c]
                        elif tipos[f][c][2] == -1 and modificable(f,c+1):
                            m[f][c+1] = 3-m[f][c]
    
                        if tipos[f][c][3] == 1 and modificable(f-1,c):
                            m[f-1][c] = m[f][c]
                        elif tipos[f][c][3] == -1 and modificable(f-1,c):
                            m[f-1][c] = 3-m[f][c]
                        if tipos[f][c][4] == 1 and modificable(f+1,c):
                            m[f+1][c] = m[f][c]
                        elif tipos[f][c][4] == -1 and modificable(f+1,c):
                            m[f+1][c] = 3-m[f][c]
                n_cas_aux=n_cas
                init_pos = simbolosIniciales(m)
                n_cas = len(init_pos)
                #Paso 2: Comrpobar los = a una casilla de distancia de los símbolos iniciales
            init_pos = simbolosIniciales(m)
            n_cas = len(init_pos)
            n_cas_aux = 0
            while n_cas != n_cas_aux:
                for i in range(0,n_cas):
                    f,c = init_pos[i]
                    if m[f][c] != 0:
                        if c < 4 and tipos[f][c+1][2] == 1 and modificable(f,c+1) and modificable(f,c+2):
                            m[f][c+1] = m[f][c+2] = 3-m[f][c]
                        elif c > 1 and tipos[f][c-1][1] == 1 and modificable(f,c-1) and modificable(f,c-2):
                            m[f][c-1] = m[f][c-2] = 3-m[f][c]
                        if f < 4 and tipos[f+1][c][4] == 1 and modificable(f+1,c) and modificable(f+2,c):
                            m[f+1][c] = m[f+2][c] == 3-m[f][c]
                        elif f > 1 and tipos[f-1][c][3] == 1 and modificable(f-1,c) and modificable(f-2,c):
                            m[f-1][c] = m[f-2][c] == 3-m[f][c]   
                n_cas_aux=n_cas
                init_pos = simbolosIniciales(m)
                n_cas = len(init_pos)
                #Paso 3: No puede haber tres símbolos iguales seguidos
            init_pos = simbolosIniciales(m)
            n_cas = len(init_pos)
            n_cas_aux = 0
            init_pos = simbolosIniciales(m)
            n_cas = len(init_pos)
            n_cas_aux = 0
            while n_cas != n_cas_aux:
                for i in range(0,n_cas):
                    f,c = init_pos[i]
                    if m[f][c] != 0:
                        #Si hay dos iguales hacia la derecha
                        if c < 4 and m[f][c] == m[f][c+1] and modificable(f,c+2) :
                            m[f][c+2] = 3-m[f][c]
                        #Si hay dos iguales hacia la izquierda
                        if c > 1 and m[f][c] == m[f][c-1] and modificable(f,c-2):
                            m[f][c-2] = 3-m[f][c]
                        #Si hay dos iguales hacia abajo
                        if f < 4 and m[f][c] == m[f+1][c] and modificable(f+2,c):
                            m[f+2][c] = 3-m[f][c]
                        #Si hay dos iguales hacia la izquierda
                        if f > 1 and m[f][c] == m[f-1][c] and modificable(f-2,c):
                            m[f-2][c] = 3-m[f][c]
                n_cas_aux=n_cas
                init_pos = simbolosIniciales(m)
                n_cas = len(init_pos)
                #Continuación del paso 3
            init_pos = simbolosIniciales(m)
            n_cas = len(init_pos)
            n_cas_aux = 0
            while n_cas != n_cas_aux:
                for i in range(0,n_cas):
                    f,c = init_pos[i]
                    if m[f][c] != 0:
                        #Si hay dos iguales con un espacio blanco hacia la derecha
                        if c < 3 and m[f][c] == m[f][c+2] and modificable(f,c+1) :
                            m[f][c+1] = 3-m[f][c]
                        #Si hay dos iguales con un espacio blanco hacia la izquierda
                        if c > 2 and m[f][c] == m[f][c-2] and modificable(f,c-1):
                            m[f][c-1] = 3-m[f][c]
                        #Si hay dos iguales con un espacio blanco hacia abajo
                        if f < 3 and m[f][c] == m[f+2][c] and modificable(f+1,c):
                            m[f+1][c] = 3-m[f][c]
                        #Si hay dos iguales con un espacio blanco hacia la izquierda
                        if f > 2 and m[f][c] == m[f-2][c] and modificable(f-1,c):
                            m[f-1][c] = 3-m[f][c]
                n_cas_aux=n_cas
                init_pos = simbolosIniciales(m)
                n_cas = len(init_pos)
                #Paso 4: Rellenar las filas o columnas en las que ya sólo queda una casilla
            init_pos = simbolosIniciales(m)
            n_cas = len(init_pos)
            n_cas_aux = 0
            while n_cas != n_cas_aux:
                for f in range(0,6):
                    for c in range(0,6):
                        if m[f][c] == 0:
                            _, f_cnt = np.unique(m[0:,c], return_counts=True)
                            _, c_cnt = np.unique(m[f,0:], return_counts=True)
                            if(len(f_cnt) == 3):
                            #si hay ya tres soles en la fila
                                if f_cnt[1] == 3:
                                    m[f][c] = 2
                                elif f_cnt[2] == 3:
                                    m[f][c] = 1
                            if(len(c_cnt) == 3):
                                #si hay ya tres soles en la fila
                                if c_cnt[1] == 3:
                                    m[f][c] = 2
                                elif c_cnt[2] == 3:
                                    m[f][c] = 1       
                n_cas_aux=n_cas
                init_pos = simbolosIniciales(m)
                n_cas = len(init_pos)
            suma_fila = np.sum(m, axis = 1)
            suma_fila0 = suma_fila[0]
            suma_fila1 = suma_fila[1]
            suma_fila2 = suma_fila[2]
            suma_fila3 = suma_fila[3]
            suma_fila4 = suma_fila[4]
            suma_fila5 = suma_fila[5]
            print("resolviendo")
            suma_col = np.sum(m, axis = 0)
            suma_col0 = suma_col[0]
            suma_col1 = suma_col[1]
            suma_col2 = suma_col[2]
            suma_col3 = suma_col[3]
            suma_col4 = suma_col[4]
            suma_col5 = suma_col[5]
        
        img = generateImage(m)
        
        
        base64_image = to_base64(img)
        image_dst.src_base64 = base64_image
        image_dst.update()

    def on_file_selected(e):
        nonlocal image
        file_path = e.files[0].path
        print("file selected :", file_path)
        image = cv2.imread(file_path)
        base64_image = to_base64(image)
        image_src.src_base64 = base64_image
        image_src.update()


    file_picker = ft.FilePicker(on_result=on_file_selected)
    page.overlay.append(file_picker)


    def on_click(e):
        file_picker.pick_files(allow_multiple=False, 
                               file_type=ft.FilePickerFileType.IMAGE)

    button = ft.ElevatedButton("Seleccionar Captura", on_click=on_click)
    button_solve = ft.ElevatedButton("Resolver Tablero", on_click=solve)
    r_button = ft.Row([button],alignment=ft.MainAxisAlignment.CENTER )
    r_button_solve = ft.Row([button_solve], alignment=ft.MainAxisAlignment.CENTER)
    column = ft.Column([r_button,images_row,r_button_solve], alignment=ft.MainAxisAlignment.CENTER)
    page.add(column)
    


ft.app(target=main)