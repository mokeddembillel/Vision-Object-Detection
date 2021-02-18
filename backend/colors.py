import numpy as np

rgb2color = {(255, 0, 0) : 'RED',
             (165,42,42) : 'BROWN',
             (128,0,128) : 'PURPLE',
             (0,255,255) : 'CYAN',
             (0, 255, 0) : 'GREEN',
             (0, 0, 255) : 'BLUE',
             (255,255,0) : 'YELLOW',
             (169,169,169) : 'GRAY',
             (255,20,147) : 'PINK',
             (255,165,0) : 'ORANGE'}

colors = np.array(list(rgb2color.keys()))