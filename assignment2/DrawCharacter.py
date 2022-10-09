from tkinter import *
import numpy as np
import time 

class DrawCharacter:
    def __init__(self, canvas_dim = 200, thickness = 5):

        self.x_hist = []
        self.y_hist = []
        self.b_hist = []

        self.thickness = thickness
        self.canvas_dim = canvas_dim

        self.root = Tk()
        self.w = Canvas(self.root, width=canvas_dim, height=canvas_dim)
        
    
    def track(self,event):
        x, y =  event.x , event.y
        translated_y = self.canvas_dim -y
        self.x_hist.append(x)
        self.y_hist.append(translated_y)
        self.b_hist.append(0)
    
    def paint(self, event):
        x1, y1 = (event.x - self.thickness), (event.y -self.thickness)
        x2, y2 = (event.x + self.thickness), (event.y + self.thickness)
        self.w.create_oval( x1, y1, x2, y2)

        self.x_hist.append(event.x)
        translated_y = self.canvas_dim -event.y
        self.y_hist.append(translated_y)
        self.b_hist.append(1)
    
    def run(self):
        self.root.title('Draw a Character')
        self.root.resizable(False, False)
        self.w.pack(expand = NO, fill = BOTH)

        self.w.bind("<Motion>", self.track)
        self.w.bind("<B1-Motion>", self.paint)
        self.root.mainloop()
    
    def get_xybpoints(self):
        return np.array([self.x_hist, self.y_hist, self.b_hist])