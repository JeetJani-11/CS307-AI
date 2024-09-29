import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animateTSP(history, points):
    frame_skip = max(1, len(history) // 1500)
    
    figure, axes = plt.subplots()
    path_line, = plt.plot([], [], linewidth=2)
    
    def setup_plot():
        initial_x = [points[i][0] for i in history[0]]
        initial_y = [points[i][1] for i in history[0]]
        plt.plot(initial_x, initial_y, 'co')
        
        margin = 0.05
        x_padding = (max(initial_x) - min(initial_x)) * margin
        y_padding = (max(initial_y) - min(initial_y)) * margin
        axes.set_xlim(min(initial_x) - x_padding, max(initial_x) + x_padding)
        axes.set_ylim(min(initial_y) - y_padding, max(initial_y) + y_padding)
        
        path_line.set_data([], [])
        return path_line,
    
    def render_frame(frame):
        current_path = history[frame] + [history[frame][0]]
        x_coords = [points[i, 0] for i in current_path]
        y_coords = [points[i, 1] for i in current_path]
        path_line.set_data(x_coords, y_coords)
        return path_line

    animation = FuncAnimation(figure, render_frame, frames=range(0, len(history), frame_skip),
                              init_func=setup_plot, interval=3, repeat=False)
    
    plt.show()
