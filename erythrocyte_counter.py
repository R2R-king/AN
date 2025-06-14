import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, feature


def count_erythrocytes(image_path, min_radius=None, max_radius=None, visualize=False, 
                       sensitivity=20, two_stage=True, precise_contours=True,
                       force_contours_for_all=True):
    """
    Подсчет эритроцитов на изображении мазка крови с использованием преобразования Хафа.
    
    Параметры:
    ----------
    image_path : str
        Путь к изображению мазка крови
    min_radius : int, optional
        Минимальный радиус клетки в пикселях
    max_radius : int, optional
        Максимальный радиус клетки в пикселях
    visualize : bool, optional
        Флаг отображения промежуточных результатов
    sensitivity : int, optional
        Чувствительность обнаружения (param2 для HoughCircles), меньшие значения = больше кругов
    two_stage : bool, optional
        Использовать двухэтапное обнаружение для повышения точности
    precise_contours : bool, optional
        Использовать точную подстройку контуров
    force_contours_for_all : bool, optional
        Гарантировать наличие контуров для всех обнаруженных центров
    
    Возвращает:
    ----------
    count : int
        Количество обнаруженных эритроцитов
    centers : ndarray
        Координаты центров обнаруженных клеток
    """
    # 1. Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение по пути {image_path}")
    
    # Преобразование в цветовое пространство RGB для визуализации
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Преобразование в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Оценка радиусов клеток, если не заданы
    if min_radius is None or max_radius is None:
        # Предполагаем, что диаметр эритроцита ~ 1/15 - 1/10 ширины изображения
        est_diameter = min(gray.shape) / 12
        min_radius = int(est_diameter * 0.35) if min_radius is None else min_radius
        max_radius = int(est_diameter * 0.75) if max_radius is None else max_radius
        
    # 3. Предобработка для улучшения контрастности и выделения особенностей
    # Улучшение контраста с использованием CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # 4. Медианная фильтрация для устранения шума
    filter_size = max(3, min(9, min_radius // 2))
    if filter_size % 2 == 0:  # Должен быть нечетным
        filter_size += 1
    filtered = cv2.medianBlur(equalized, filter_size)
    
    # Применение билатерального фильтра для сохранения краев при сглаживании шума
    bilateral = cv2.bilateralFilter(filtered, 9, 75, 75)
    
    # Выделяем края для последующей точной подстройки
    edges = None
    if precise_contours:
        # Многоуровневое выделение краев для повышения шансов обнаружения
        edges_canny = cv2.Canny(bilateral, 30, 100)
        
        # Дополнительно используем лапласиан для выделения краев
        laplacian = cv2.Laplacian(bilateral, cv2.CV_8U)
        _, edges_laplacian = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)
        
        # Комбинируем результаты
        edges = cv2.bitwise_or(edges_canny, edges_laplacian)
        
        # Дилатация для соединения прерывистых краев
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 5. Адаптивная бинаризация для выделения клеток
    block_size = min(99, max(11, int(min_radius * 2) | 1))  # должно быть нечетным
    binary = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        2
    )
    
    # 6. Морфологические операции для улучшения бинаризации
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 7. Применяем преобразование Хафа для обнаружения кругов (эритроцитов)
    all_circles = []
    
    if two_stage:
        # Первый проход: обнаружение с высокой точностью
        circles1 = detect_circles(bilateral, min_radius, max_radius, sensitivity)
        
        # Создаем маску уже найденных кругов
        mask = np.zeros_like(gray)
        if circles1 is not None:
            for circle in circles1[0]:
                # Преобразуем координаты в целые числа
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                cv2.circle(mask, center, radius, 255, -1)
        
        # Второй проход: пониженная чувствительность для обнаружения пропущенных кругов
        # Используем изображение с "удаленными" уже обнаруженными клетками
        filtered_for_second_pass = bilateral.copy()
        filtered_for_second_pass[mask > 0] = 0
        
        # Обнаружение с меньшей чувствительностью
        circles2 = detect_circles(
            filtered_for_second_pass, 
            min_radius, 
            max_radius, 
            sensitivity * 0.7  # Понижаем порог для обнаружения менее четких кругов
        )
        
        # Объединяем результаты
        if circles1 is not None:
            all_circles.append(circles1)
        if circles2 is not None:
            all_circles.append(circles2)
            
        # Объединяем результаты в один массив
        if all_circles:
            circles = np.concatenate(all_circles, axis=1) if len(all_circles) > 1 else all_circles[0]
        else:
            circles = None
    else:
        # Обычное обнаружение кругов
        circles = detect_circles(bilateral, min_radius, max_radius, sensitivity)
    
    # 8. Обработка результатов
    centers = []
    radii = []
    contours = []
    all_centers_with_contours = []  # Центры, для которых нашли контуры
    
    if circles is not None:
        # Преобразуем координаты в целые числа
        circles_int = np.uint16(np.around(circles))
        
        # Убираем дубликаты центров, находящиеся слишком близко друг к другу
        filtered_circles = filter_duplicate_circles(circles_int[0], min_radius)
        
        for x, y, r in filtered_circles:
            x, y, r = int(x), int(y), int(r)
            centers.append((x, y))
            
            # Если требуется точная подстройка контуров
            contour_found = False
            if precise_contours and edges is not None:
                # Уточняем радиус с помощью анализа градиента
                refined_r = refine_radius(bilateral, x, y, r, min_radius, max_radius)
                
                # Находим точный контур для эритроцита
                contour = find_precise_contour(edges, gray, x, y, refined_r)
                if contour is not None and len(contour) >= 5:  # Минимум 5 точек для эллипса
                    contours.append(contour)
                    radii.append(refined_r)  # Сохраняем уточненный радиус
                    all_centers_with_contours.append((x, y))
                    contour_found = True
            
            # Если не нашли контур, но нужно создать для всех центров
            if not contour_found:
                if precise_contours:
                    # Используем уточненный радиус, если выполнялось уточнение
                    refined_r = refine_radius(bilateral, x, y, r, min_radius, max_radius)
                    radii.append(refined_r)
                    
                    # Создаем искусственный контур в форме круга
                    if force_contours_for_all:
                        num_points = 36  # Количество точек в контуре
                        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
                        circle_points = []
                        for angle in angles:
                            px = int(x + refined_r * np.cos(angle))
                            py = int(y + refined_r * np.sin(angle))
                            circle_points.append([[px, py]])
                        artificial_contour = np.array(circle_points, dtype=np.int32)
                        contours.append(artificial_contour)
                        all_centers_with_contours.append((x, y))
                else:
                    # Если не используется точная подстройка, просто сохраняем исходный радиус
                    radii.append(r)
    
    # 9. Визуализация результатов
    if visualize:
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # Исходное изображение
        axs[0, 0].imshow(img_rgb)
        axs[0, 0].set_title('Исходное изображение')
        axs[0, 0].axis('off')
        
        # После предобработки
        axs[0, 1].imshow(bilateral, cmap='gray')
        axs[0, 1].set_title('После предобработки')
        axs[0, 1].axis('off')
        
        # Найденные контуры/края
        if edges is not None:
            axs[1, 0].imshow(edges, cmap='gray')
            axs[1, 0].set_title('Выделенные края')
        else:
            axs[1, 0].imshow(binary, cmap='gray')
            axs[1, 0].set_title('Бинаризованное изображение')
        axs[1, 0].axis('off')
        
        # Результаты с отмеченными эритроцитами
        result_img = img_rgb.copy()
        
        # Рисуем контуры, если они найдены
        if contours:
            cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
        else:
            # Рисуем круги, если контуры не найдены
            for (x, y), r in zip(centers, radii):
                cv2.circle(result_img, (x, y), r, (0, 255, 0), 2)
        
        # Всегда отмечаем центры
        for x, y in centers:
            cv2.circle(result_img, (x, y), 2, (255, 0, 0), 2)
            
        axs[1, 1].imshow(result_img)
        axs[1, 1].set_title(f'Обнаружено эритроцитов: {len(centers)}')
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return len(centers), centers


def refine_radius(image, x, y, initial_radius, min_radius, max_radius, num_profiles=36):
    """
    Уточнение радиуса клетки с помощью анализа профилей интенсивности.
    
    Параметры:
    ----------
    image : ndarray
        Изображение в оттенках серого
    x, y : int
        Координаты центра клетки
    initial_radius : int
        Начальная оценка радиуса
    min_radius, max_radius : int
        Минимальный и максимальный допустимые радиусы
    num_profiles : int, optional
        Количество радиальных профилей
        
    Возвращает:
    -----------
    radius : int
        Уточненный радиус
    """
    height, width = image.shape[:2]
    radii = []
    
    # Генерируем набор углов для радиальных профилей
    angles = np.linspace(0, 2*np.pi, num_profiles, endpoint=False)
    
    for angle in angles:
        # Создаем радиальный профиль
        max_len = max(initial_radius * 2, max_radius)
        profile_x = np.round(x + np.cos(angle) * np.arange(max_len)).astype(int)
        profile_y = np.round(y + np.sin(angle) * np.arange(max_len)).astype(int)
        
        # Фильтруем точки, выходящие за границы изображения
        valid_points = (profile_x >= 0) & (profile_x < width) & (profile_y >= 0) & (profile_y < height)
        profile_x = profile_x[valid_points]
        profile_y = profile_y[valid_points]
        
        if len(profile_x) > 0:
            # Получаем профиль интенсивности вдоль радиального направления
            profile = image[profile_y, profile_x]
            
            # Вычисляем градиент профиля
            gradient = np.abs(np.diff(profile.astype(np.float32)))
            
            if len(gradient) > 0:
                # Находим положение максимального градиента (предполагаемая граница)
                edge_idx = np.argmax(gradient)
                if edge_idx > 0:  # Убедимся, что мы не находимся в начале профиля
                    # Расстояние от центра до точки с максимальным градиентом
                    dist = np.sqrt((profile_x[edge_idx] - x) ** 2 + (profile_y[edge_idx] - y) ** 2)
                    
                    # Проверяем, находится ли радиус в допустимом диапазоне
                    if min_radius <= dist <= max_radius:
                        radii.append(dist)
    
    # Если нашли достаточное количество точек, используем медиану
    if len(radii) >= num_profiles * 0.3:  # Снижаем требование до 30% успешных профилей
        return int(np.median(radii))
    else:
        return initial_radius


def find_precise_contour(edges, gray_img, center_x, center_y, radius):
    """
    Находит точный контур эритроцита в окрестности обнаруженного центра.
    
    Параметры:
    ----------
    edges : ndarray
        Изображение с выделенными краями
    gray_img : ndarray
        Исходное изображение в оттенках серого
    center_x, center_y : int
        Координаты центра клетки
    radius : int
        Радиус клетки
        
    Возвращает:
    -----------
    contour : ndarray или None
        Точный контур клетки или None, если контур не найден
    """
    # Создаем маску в окрестности клетки
    height, width = edges.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Ограничиваем область поиска контура
    search_radius = int(radius * 1.7)  # Увеличиваем радиус поиска
    cv2.circle(mask, (center_x, center_y), search_radius, 255, -1)
    
    # Применяем маску к изображению с краями
    roi_edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    # Находим контуры в ROI
    contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
        
    # Выбираем наиболее подходящий контур
    best_contour = None
    best_score = float('-inf')
    
    for contour in contours:
        # Если контур слишком маленький, пропускаем
        if len(contour) < 5:
            continue
            
        # Вычисляем характеристики контура
        area = cv2.contourArea(contour)
        expected_area = np.pi * radius**2
        
        # Если площадь контура слишком маленькая или большая, пропускаем
        if area < expected_area * 0.3 or area > expected_area * 3.0:
            continue
            
        # Находим центр масс контура
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Расстояние от обнаруженного центра до центра масс контура
        dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        
        # Если центр контура слишком далеко от предполагаемого центра эритроцита, пропускаем
        if dist_to_center > radius * 0.8:
            continue
            
        # Оценка схожести контура с окружностью
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Оценка контура по нескольким критериям:
        # - близость к ожидаемой площади
        # - близость центра масс к центру круга
        # - округлость
        area_score = 1 - min(abs(area - expected_area) / expected_area, 1)
        center_score = 1 - min(dist_to_center / radius, 1)
        
        score = area_score * 0.3 + center_score * 0.5 + circularity * 0.2
        
        if score > best_score:
            best_score = score
            best_contour = contour
    
    # Снижаем порог принятия контура
    if best_score > 0.4:  # Было 0.6, снижаем до 0.4
        # Сглаживаем контур для получения более плавных границ
        epsilon = 0.01 * cv2.arcLength(best_contour, True)
        approx_contour = cv2.approxPolyDP(best_contour, epsilon, True)
        return approx_contour
    
    return None


def detect_circles(img, min_radius, max_radius, sensitivity=20):
    """
    Обнаружение кругов с использованием преобразования Хафа.
    
    Параметры:
    ----------
    img : ndarray
        Изображение для обнаружения кругов
    min_radius : int
        Минимальный радиус круга
    max_radius : int
        Максимальный радиус круга
    sensitivity : int or float
        Чувствительность алгоритма (меньше = больше кругов)
        
    Возвращает:
    -----------
    circles : ndarray
        Массив обнаруженных кругов в формате (x, y, r)
    """
    return cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1,                       # Отношение разрешения
        minDist=min_radius * 0.8,   # Минимальное расстояние между центрами
        param1=50,                  # Верхний порог для детектора края Канни
        param2=sensitivity,         # Порог для обнаружения центров
        minRadius=min_radius,
        maxRadius=max_radius
    )


def filter_duplicate_circles(circles, min_radius):
    """
    Фильтрация дубликатов кругов, находящихся слишком близко друг к другу.
    
    Параметры:
    ----------
    circles : ndarray
        Массив кругов в формате (x, y, r)
    min_radius : int
        Минимальный радиус круга
        
    Возвращает:
    -----------
    filtered_circles : list
        Отфильтрованный список кругов без дубликатов
    """
    if len(circles) == 0:
        return []
    
    # Сортируем круги по радиусу (от большего к меньшему)
    sorted_circles = sorted(circles, key=lambda x: -x[2])
    filtered_circles = [sorted_circles[0]]
    
    # Минимальное расстояние между центрами
    min_dist = min_radius * 0.8
    
    for circle in sorted_circles[1:]:
        x, y, r = circle
        
        # Проверяем, не слишком ли близко к уже принятым кругам
        is_duplicate = False
        for accepted_circle in filtered_circles:
            ax, ay, _ = accepted_circle
            distance = np.sqrt((x - ax)**2 + (y - ay)**2)
            if distance < min_dist:
                is_duplicate = True
                break
        
        # Если не дубликат, добавляем в отфильтрованный список
        if not is_duplicate:
            filtered_circles.append(circle)
    
    return filtered_circles


def optimize_parameters(image_path, expected_count=None, visualize=False):
    """
    Оптимизация параметров для более точного обнаружения эритроцитов.
    
    Параметры:
    ----------
    image_path : str
        Путь к изображению мазка крови
    expected_count : int, optional
        Ожидаемое количество эритроцитов (если известно)
    visualize : bool, optional
        Визуализировать результат с лучшими параметрами
        
    Возвращает:
    ----------
    best_params : dict
        Оптимальные параметры для обнаружения эритроцитов
    """
    # Загружаем изображение для определения его размеров
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение по пути {image_path}")
    
    height, width = img.shape[:2]
    estimated_cell_diameter = min(height, width) / 12
    
    # Диапазоны параметров для перебора
    min_radius_range = [
        int(estimated_cell_diameter * 0.3),
        int(estimated_cell_diameter * 0.35),
        int(estimated_cell_diameter * 0.4)
    ]
    
    max_radius_range = [
        int(estimated_cell_diameter * 0.65),
        int(estimated_cell_diameter * 0.7),
        int(estimated_cell_diameter * 0.75)
    ]
    
    sensitivity_range = [15, 20, 25]
    
    best_count = 0
    best_params = None
    best_diff = float('inf')
    
    # Перебор параметров
    for min_radius in min_radius_range:
        for max_radius in max_radius_range:
            if min_radius >= max_radius:
                continue
                
            for sensitivity in sensitivity_range:
                count, _ = count_erythrocytes(
                    image_path, 
                    min_radius, 
                    max_radius, 
                    visualize=False, 
                    sensitivity=sensitivity
                )
                
                if expected_count is not None:
                    diff = abs(count - expected_count)
                    if diff < best_diff:
                        best_diff = diff
                        best_count = count
                        best_params = {
                            'min_radius': min_radius,
                            'max_radius': max_radius,
                            'sensitivity': sensitivity
                        }
                elif count > best_count:
                    best_count = count
                    best_params = {
                        'min_radius': min_radius,
                        'max_radius': max_radius,
                        'sensitivity': sensitivity
                    }
    
    print(f"Наилучшие параметры: {best_params}")
    print(f"Количество обнаруженных эритроцитов: {best_count}")
    
    # Визуализируем результат с лучшими параметрами
    if visualize and best_params:
        count_erythrocytes(
            image_path, 
            best_params['min_radius'], 
            best_params['max_radius'], 
            visualize=True,
            sensitivity=best_params['sensitivity'],
            precise_contours=True,
            force_contours_for_all=True
        )
    
    return best_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Подсчет эритроцитов на изображении мазка крови')
    parser.add_argument('image_path', help='Путь к изображению мазка крови')
    parser.add_argument('--min_radius', type=int, help='Минимальный радиус клетки в пикселях')
    parser.add_argument('--max_radius', type=int, help='Максимальный радиус клетки в пикселях')
    parser.add_argument('--sensitivity', type=float, default=20, 
                        help='Чувствительность обнаружения (меньше = больше кругов)')
    parser.add_argument('--optimize', action='store_true', help='Оптимизировать параметры обнаружения')
    parser.add_argument('--expected_count', type=int, 
                        help='Ожидаемое количество эритроцитов (для оптимизации)')
    parser.add_argument('--visualize', action='store_true', help='Отображать результаты')
    parser.add_argument('--no_two_stage', action='store_true', help='Отключить двухэтапное обнаружение')
    parser.add_argument('--no_precise_contours', action='store_true', help='Отключить точную подстройку контуров')
    parser.add_argument('--no_force_contours', action='store_true', help='Не создавать контуры для всех центров')
    
    args = parser.parse_args()
    
    if args.optimize:
        optimize_parameters(args.image_path, args.expected_count, args.visualize)
    else:
        count, _ = count_erythrocytes(
            args.image_path, 
            args.min_radius, 
            args.max_radius, 
            args.visualize,
            args.sensitivity,
            not args.no_two_stage,
            not args.no_precise_contours,
            not args.no_force_contours
        )
        print(f"Обнаружено эритроцитов: {count}") 