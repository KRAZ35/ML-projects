import cv2
import numpy as np
import matplotlib.pyplot as plt

# Класс для моделирования ганглиозной клетки
class GanglionicCell():
    def __init__(self, position, central_radius=5, peripherial_radius=11, isoff=False):
        self.pos = position
        self.s1 = central_radius
        self.s2 = peripherial_radius
        self.isoff = isoff

    def get_response(self, image):
        gauss_d1 = cv2.GaussianBlur(image, (self.s1, self.s1), sigmaX=0)
        gauss_d2 = cv2.GaussianBlur(image, (self.s2, self.s2), sigmaX=0)
        if self.isoff:
            laplace_response = gauss_d2 - gauss_d1
        else:
            laplace_response = gauss_d1 - gauss_d2
        v = laplace_response[self.pos[1], self.pos[0]]
        return v

# Класс для моделирования простой клетки зрительной коры
class SimpleCell():
    def __init__(self, position, orientation='vertical', size=5):
        self.ganglionic_cells = []
        self.size = size
        d = 3  # Шаг между клетками
        for i in range(-(size//2), size//2 + 1):
            for j in range(-(size//2), size//2 + 1):
                isoff = True
                if orientation == 'vertical' and i == 0:
                    isoff = False
                elif orientation == 'horizontal' and j == 0:
                    isoff = False
                elif orientation == 'diagonal' and i == j:
                    isoff = False
                pos = (position[0] + i * d, position[1] + j * d)
                self.ganglionic_cells.append(GanglionicCell(pos, isoff=isoff))

    def get_response(self, image):
        response = 0.0
        for cell in self.ganglionic_cells:
            response += cell.get_response(image)
        return response

# Класс для моделирования сложной клетки зрительной коры
class ComplexCell():
    def __init__(self, positions, orientation='vertical', size=5):
        self.simple_cells = []
        for pos in positions:
            self.simple_cells.append(SimpleCell(pos, orientation, size))

    def get_response(self, image):
        response = 0.0
        for cell in self.simple_cells:
            response += abs(cell.get_response(image))
        return response

# Функции для экспериментов
def check_point_stimulus(cell):
    response_map = np.zeros((13, 13), dtype=np.float32)
    for i in range(13):
        for j in range(13):
            image = np.zeros((256, 256), dtype=np.float32)
            x = 128 + i - 6
            y = 128 + j - 6
            cv2.circle(image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)
            v = cell.get_response(image)
            response_map[j, i] = v
    return response_map

def check_circle_stimulus(cell):
    responses = []
    radii = range(0, 30)
    for r in radii:
        image = np.zeros((256, 256), dtype=np.float32)
        cv2.circle(image, (128, 128), radius=r, color=(255, 255, 255), thickness=-1)
        v = cell.get_response(image)
        responses.append(v)
    return responses, radii

def rotate_line(cell):
    responses = []
    angles = []
    for i in range(0, 360, 10):
        angle = np.deg2rad(i)
        image = np.zeros((256, 256), dtype=np.float32)
        x0 = int(128 + 150 * np.cos(angle))
        y0 = int(128 + 150 * np.sin(angle))
        x1 = int(128 - 150 * np.cos(angle))
        y1 = int(128 - 150 * np.sin(angle))
        cv2.line(image, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=3)
        v = cell.get_response(image)
        responses.append(v)
        angles.append(i)
    return responses, angles

# Функции для построения графиков
def plot_response_map(response_map, title=''):
    plt.imshow(response_map, cmap='jet', interpolation='nearest', extent=[-6, 6, -6, 6])
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Смещение по X')
    plt.ylabel('Смещение по Y')
    plt.show()

def plot_responses_vs_radii(responses, radii, title=''):
    plt.plot(radii, responses)
    plt.title(title)
    plt.xlabel('Радиус пятна')
    plt.ylabel('Отклик клетки')
    plt.show()

def plot_responses_vs_angles(responses, angles, title=''):
    plt.plot(angles, responses)
    plt.title(title)
    plt.xlabel('Угол (градусы)')
    plt.ylabel('Отклик клетки')
    plt.show()

def main():
    # Инициализация ганглиозной клетки в центре изображения
    cell_pos = (128, 128)
    ganglion_cell = GanglionicCell(cell_pos)

    # Эксперимент 1: Ганглиозная клетка, отклик на точечный стимул в разных позициях
    response_map = check_point_stimulus(ganglion_cell)
    plot_response_map(response_map, 'Ганглиозная клетка: отклик на точечный стимул')

    # Эксперимент 2: Ганглиозная клетка, отклик на световое пятно разного размера
    responses, radii = check_circle_stimulus(ganglion_cell)
    plot_responses_vs_radii(responses, radii, 'Ганглиозная клетка: отклик на пятно разного размера')

    # Эксперимент 3: Ганглиозная клетка, отклик на линию разной ориентации
    responses, angles = rotate_line(ganglion_cell)
    plot_responses_vs_angles(responses, angles, 'Ганглиозная клетка: отклик на линию разной ориентации')

    # Инициализация простой клетки с вертикальной ориентацией
    simple_cell = SimpleCell(cell_pos, orientation='vertical', size=5)

    # Эксперимент 4: Простая клетка, отклик на точечный стимул в разных позициях
    response_map = check_point_stimulus(simple_cell)
    plot_response_map(response_map, 'Простая клетка: отклик на точечный стимул')

    # Эксперимент 5: Простая клетка, отклик на световое пятно разного размера
    responses, radii = check_circle_stimulus(simple_cell)
    plot_responses_vs_radii(responses, radii, 'Простая клетка: отклик на пятно разного размера')

    # Эксперимент 6: Простая клетка, отклик на линию разной ориентации
    responses, angles = rotate_line(simple_cell)
    plot_responses_vs_angles(responses, angles, 'Простая клетка: отклик на линию разной ориентации')

    # Инициализация сложной клетки на основе нескольких простых клеток
    positions = [(128 + dx, 128 + dy) for dx in range(-15, 16, 5) for dy in range(-15, 16, 5)]
    complex_cell = ComplexCell(positions, orientation='vertical', size=5)

    # Эксперимент 7: Сложная клетка, отклик на точечный стимул в разных позициях
    response_map = check_point_stimulus(complex_cell)
    plot_response_map(response_map, 'Сложная клетка: отклик на точечный стимул')

    # Эксперимент 8: Сложная клетка, отклик на световое пятно разного размера
    responses, radii = check_circle_stimulus(complex_cell)
    plot_responses_vs_radii(responses, radii, 'Сложная клетка: отклик на пятно разного размера')

    # Эксперимент 9: Сложная клетка, отклик на линию разной ориентации
    responses, angles = rotate_line(complex_cell)
    plot_responses_vs_angles(responses, angles, 'Сложная клетка: отклик на линию разной ориентации')

if __name__ == '__main__':
    main()
