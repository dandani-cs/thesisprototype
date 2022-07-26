import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename 
from PIL import ImageTk, Image

photo_lbl = None
class_lbl = None
size = (300, 300)

def classify(path):
    # run model here

    img = Image.open(path) # output image put here)
    size = img.size
    img = img.resize(size, Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)

    photo_lbl.configure(
        text = "", 
        image = tk_img,
        width = size[0],
        height = size[1]
    )
    photo_lbl.image = tk_img
    photo_lbl.text = ""
    class_lbl.configure(text = "CLASSIFIED")


def uploadFile():
    file_path = askopenfilename(filetypes=[('Image Files', "*.jpg *.png *jpeg")])
    if file_path is not None:
        classify(file_path)

window = tk.Tk()
window.title("DFS - CapsNet Mask and Usage Classification Prototype")

header_txt = "Enhancing CNN-based Face Mask Detection  \n and Usage Classification using Capsule Networks"
names_txt  = "Dela Cruz, Federez, Samonte"
year_txt   = "BSCS - Thesis 3 4Q2122"

header_lbl = tk.Label(
    text = header_txt, 
    font = ('Arial', 18)
    ).pack()
names_lbl  = tk.Label(
    text = names_txt, 
    font = ('Arial', 14)
    ).pack()
year_lbl   = tk.Label(text = year_txt).pack()

photo_lbl = tk.Label(
    text = "Image appears here",
    width = 25,
    height = 25
    )
class_lbl = tk.Label(
    text = "CLASSIFICATION",
    fg = "blue",
    font = ("Arial", 18)
    )

upload_btn = tk.Button(
    text = "Upload and Run",
    width = 15,
    command = uploadFile
    ).pack()


photo_lbl.pack()
class_lbl.pack()



window.mainloop()