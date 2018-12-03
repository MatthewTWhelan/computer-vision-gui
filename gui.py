#!/usr/bin/python3

import tkinter as tk
import tkinter.messagebox as messagebox
from PIL import ImageTk, Image
import cv2 as cv
import numpy as np

class CVWindow:

    def __init__(self):
        self.image_dir = "images/"

        self.root = tk.Tk()
        self.root.title("Computer Vision Examples")
        self.root.geometry("700x750")
        self.root.configure(background="white")

        self.gui_title = tk.Message(self.root, text="Examples of some Basic Computer Vision Processes", background="light blue",
                                  font=("Courier", 15), width=300)
        self.gui_title.place(x=10, y=10)

        self.img_no = 0

        # Uni of Sheff image
        self.uos_image = ImageTk.PhotoImage(Image.open(self.image_dir + "UoS.jpg"))
        self.uos_label = tk.Label(self.root, image=self.uos_image, borderwidth=0).place(x=10, y=680)

        # 5 sample images to choose from
        self.root.imgs = list()
        self.root.imgs.append(ImageTk.PhotoImage(Image.open(self.image_dir + "image1.jpg")))
        self.root.imgs.append(ImageTk.PhotoImage(Image.open(self.image_dir + "image2.jpg")))
        self.root.imgs.append(ImageTk.PhotoImage(Image.open(self.image_dir + "image3.jpg")))
        self.root.imgs.append(ImageTk.PhotoImage(Image.open(self.image_dir + "image4.jpg")))
        self.root.imgs.append(ImageTk.PhotoImage(Image.open(self.image_dir + "image5.jpg")))

        # The help windows (uploaded as images for simplicity)
        self.img_colour_segmentation_help = ImageTk.PhotoImage(Image.open(self.image_dir + "colour_segmentation_help.jpg"))
        self.img_edge_detection_help = ImageTk.PhotoImage(Image.open(self.image_dir + "edge_detection_help.jpg"))
        self.img_smoothing_blurring_help = ImageTk.PhotoImage(Image.open(self.image_dir + "smoothing_blurring_help.jpg"))
        self.img_difference_gaussian_help = ImageTk.PhotoImage(Image.open(self.image_dir + "DoG_help.jpg"))

        # To avoid the processed image being garbaged, it is saved upon each processing operation
        self.root.img_processed = ImageTk.PhotoImage(Image.open(self.image_dir + "image_processed_init.jpg"))

        # They can be selected with radio buttons
        self.Label1 = tk.Label(self.root, text="Select an image:", background="white")
        self.Label1.place(x=10, y=80)

        self.var = tk.IntVar()
        self.R1 = tk.Radiobutton(self.root, text="1", variable=self.var, value=0, background="white",
                                 highlightthickness=0, command=lambda: self.loadImage(0))
        self.R2 = tk.Radiobutton(self.root, text="2", variable=self.var, value=1, background="white",
                                 highlightthickness=0, command=lambda: self.loadImage(1))
        self.R3 = tk.Radiobutton(self.root, text="3", variable=self.var, value=2, background="white",
                                 highlightthickness=0, command=lambda: self.loadImage(2))
        self.R4 = tk.Radiobutton(self.root, text="4", variable=self.var, value=3, background="white",
                                 highlightthickness=0, command=lambda: self.loadImage(3))
        self.R5 = tk.Radiobutton(self.root, text="5", variable=self.var, value=4, background="white",
                                 highlightthickness=0, command=lambda: self.loadImage(4))
        self.R1.place(x=120, y=80)
        self.R2.place(x=160, y=80)
        self.R3.place(x=200, y=80)
        self.R4.place(x=240, y=80)
        self.R5.place(x=280, y=80)


        # display the first image initially
        self.img_display = tk.Label(self.root, image=self.root.imgs[0])
        self.img_display.place(x=10, y=110)

        # insert an arrow between the two images
        self.root.img_arrow = ImageTk.PhotoImage(Image.open(self.image_dir + "arrow.jpg"))
        self.img_arrow_display = tk.Label(self.root, image=self.root.img_arrow, background="white")
        self.img_arrow_display.place(x=185, y=365)

        # display the processed image
        self.img_processed = tk.Label(self.root, image=self.root.img_processed)
        self.img_processed.place(x=10, y=400)

        # The buttons handle the CV algorithm employed on the image
        self.buttons = list()
        self.buttons.append(tk.Button(self.root, text="Colour Segmentation", borderwidth=1, background="light grey",
                                      command=lambda: self.colour_segment()))
        self.buttons.append(tk.Button(self.root, text="Edge Detection", borderwidth=1, background="light grey",
                                      command=lambda: self.edge_detection()))
        self.buttons.append(tk.Button(self.root, text="Blurring and Smoothing", borderwidth=1, background="light grey",
                                      command=lambda: self.smoothing_blurring()))
        self.buttons.append(tk.Button(self.root, text="Difference of Gaussians", borderwidth=1, background="light grey",
                                      command=lambda: self.difference_gaussians()))
        self.buttons[0].place(x=420, y=20)
        self.buttons[1].place(x=420, y=170)
        self.buttons[2].place(x=420, y=345)
        self.buttons[3].place(x=420, y=600)

        # Buttons that open a help window and that describe the algorithms
        self.BH1 = tk.Button(self.root, text="?", command=lambda: self.colour_segment_help())
        self.BH2 = tk.Button(self.root, text="?", command=lambda: self.edge_detection_help())
        self.BH3 = tk.Button(self.root, text="?", command=lambda: self.smoothing_blurring_help())
        self.BH4 = tk.Button(self.root, text="?", command=lambda: self.difference_gaussian_help())
        self.BH1.place(x=620, y=20)
        self.BH2.place(x=620, y=170)
        self.BH3.place(x=620, y=345)
        self.BH4.place(x=620, y=600)

        # Lines to section off each image process
        self.grey_line = ImageTk.PhotoImage(Image.open(self.image_dir + "grey_line.jpg"))
        self.line1 = tk.Label(self.root, image=self.grey_line, borderwidth=0)
        self.line2 = tk.Label(self.root, image=self.grey_line, borderwidth=0)
        self.line3 = tk.Label(self.root, image=self.grey_line, borderwidth=0)
        self.line1.place(x=420, y=140)
        self.line2.place(x=420, y=310)
        self.line3.place(x=420, y=560)

        # The first algorithm has 6 values that can be changed: the lower and upper limits of colour segmentation for RGB
        self.label1 = tk.Label(self.root, text="Lower limit: ", background="white")
        self.label2 = tk.Label(self.root, text="Upper limit: ", background="white")
        self.label1.place(x=420, y=80)
        self.label2.place(x=420, y=105)

        self.text_box1 = tk.Text(height=1, width=4)
        self.text_box2 = tk.Text(height=1, width=4)
        self.text_box3 = tk.Text(height=1, width=4)
        self.text_box4 = tk.Text(height=1, width=4)
        self.text_box5 = tk.Text(height=1, width=4)
        self.text_box6 = tk.Text(height=1, width=4)
        self.text_box1.place(x=510, y=80)
        self.text_box2.place(x=545, y=80)
        self.text_box3.place(x=580, y=80)
        self.text_box4.place(x=510, y=105)
        self.text_box5.place(x=545, y=105)
        self.text_box6.place(x=580, y=105)

        # We provide example values from the start
        self.text_box1.insert("end", "0")
        self.text_box2.insert("end", "80")
        self.text_box3.insert("end", "0")
        self.text_box4.insert("end", "255")
        self.text_box5.insert("end", "150")
        self.text_box6.insert("end", "255")

        tk.Label(text="R", background="white").place(x=520, y=60)
        tk.Label(text="G", background="white").place(x=555, y=60)
        tk.Label(text="B", background="white").place(x=590, y=60)

        # The edge detector has two limits that can be altered using a slider
        self.label1 = tk.Label(self.root, text="Lower limit: ", background="white")
        self.label2 = tk.Label(self.root, text="Upper limit: ", background="white")
        self.label1.place(x=420, y=233)
        self.label2.place(x=420, y=273)

        self.S1 = tk.Scale(self.root, from_=0, to=500, orient="horizontal", background="white", border=0,
                           highlightthickness=0, command=lambda x: self.edge_detection())
        self.S1.place(x=510, y=215)
        self.S2 = tk.Scale(self.root, from_=0, to=500, orient="horizontal", background="white", border=0,
                           highlightthickness = 0, command=lambda x: self.edge_detection())
        self.S2.place(x=510, y=255)

        # Three radio buttons to select between average and Gaussian smoothing/blurring
        self.root.var_smooth = tk.IntVar()
        self.R6 = tk.Radiobutton(self.root, text="Average smoothing", variable=self.root.var_smooth, value=0,
                                 background="white", highlightthickness=0)
        self.R7 = tk.Radiobutton(self.root, text="Gaussian blurring", variable=self.root.var_smooth, value=1,
                                 background="white", highlightthickness=0)
        self.R8 = tk.Radiobutton(self.root, text="Difference of Gaussian;", variable=self.root.var_smooth,
                                 value=2, background="white", highlightthickness=0)
        self.R6.place(x=420, y=400)
        self.R7.place(x=420, y=480)


        # Scale to set the blurring kernel size
        self.S3 = tk.Scale(self.root, from_=3, to=15, orient="horizontal", background="white", highlightthickness=0,
                           command=self.fix_slider, length=80)
        self.S3.place(x=520, y=420)

        # Scale to set the Gaussian standard deviation
        self.S4 = tk.Scale(self.root, from_=0.1, to=5, resolution=0.2, orient="horizontal", background="white",
                           highlightthickness=0, length=80, command=lambda x: self.smoothing_blurring())
        self.S4.place(x=570, y=500)
        tk.Label(self.root, text="Kernel Size =", background="white").place(x=420, y=440)
        tk.Label(self.root, text="Standard Deviation =", background="white").place(x=420, y=520)

        # Scales to set the Difference of Gaussian standard deviations
        self.S5 = tk.Scale(self.root, from_=1, to=1.4, resolution=0.02, orient="horizontal", background="white",
                           highlightthickness=0, length=80, command=lambda x: self.difference_gaussians())
        self.S6 = tk.Scale(self.root, from_=1, to=1.4, resolution=0.02, orient="horizontal", background="white",
                           highlightthickness=0, length=80, command=lambda x: self.difference_gaussians())
        self.S5.place(x=580, y=640)
        self.S6.place(x=580, y=690)
        tk.Label(self.root, text="Standard Deviation 1 =", background="white").place(x=420, y=660)
        tk.Label(self.root, text="Standard Deviation 2 =", background="white").place(x=420, y=710)

        self.root.mainloop()

    # There is a bug in tk.Scale which doesn't allow odd numbers in steps of two to be shown. This is a quick fix for that.
    def fix_slider(self, n):
        val = int(n)
        if val % 2 == 0:
            self.S3.set(str(val+1))
        self.smoothing_blurring()

    def loadImage(self, img_no):
        self.img_display.configure(image=self.root.imgs[img_no])

    # Changes the colour of the button selected
    def buttonColour(self, button_no):
        for i in range(4):
            if not button_no == i:
                self.buttons[i].configure(background="light grey")
            else:
                pass
                self.buttons[i].configure(background="light green")

    ####################################################################################################################
    # Separate windows that open upon pressing the help buttons

    def colour_segment_help(self):
        self.window1 = tk.Toplevel(self.root, background="white")
        self.window1.title("Colour Segmentation")
        self.window1.geometry("600x600")

        scrollbar = tk.Scrollbar(self.window1)
        scrollbar.pack(side="right", fill="both")

        canvas = tk.Canvas(self.window1, width=600, height=700, bg="white", highlightthickness=0,
                           yscrollcommand=scrollbar.set)
        canvas.pack(expand="yes", fill="both", padx=10)
        canvas.create_image((0, 0), image=self.img_colour_segmentation_help, anchor="nw")
        scrollbar.config(command=canvas.yview)
        canvas.configure(scrollregion=canvas.bbox('all'))

    def edge_detection_help(self):
        self.window2 = tk.Toplevel(self.root, background="white")
        self.window2.title("Edge Detection")
        self.window2.geometry("600x600")

        scrollbar = tk.Scrollbar(self.window2)
        scrollbar.pack(side="right", fill="both")

        canvas = tk.Canvas(self.window2, width=600, height=700, bg="white", highlightthickness=0,
                           yscrollcommand=scrollbar.set)
        canvas.pack(expand="yes", fill="both")
        canvas.create_image((0, 0), image=self.img_edge_detection_help, anchor="nw")
        scrollbar.config(command=canvas.yview)
        canvas.configure(scrollregion=canvas.bbox('all'))


    def smoothing_blurring_help(self):
        self.window3 = tk.Toplevel(self.root, background="white")
        self.window3.title("Smoothing and Blurring")
        self.window3.geometry("600x600")

        scrollbar = tk.Scrollbar(self.window3)
        scrollbar.pack(side="right", fill="both")

        canvas = tk.Canvas(self.window3, width=600, height=700, bg="white", highlightthickness=0,
                           yscrollcommand=scrollbar.set)
        canvas.pack(expand="yes", fill="both")
        canvas.create_image((0, 0), image=self.img_smoothing_blurring_help, anchor="nw")
        scrollbar.config(command=canvas.yview)
        canvas.configure(scrollregion=canvas.bbox('all'))

    def difference_gaussian_help(self):
        self.window4 = tk.Toplevel(self.root, background="white")
        self.window4.title("Difference of Gaussians")
        self.window4.geometry("600x600")

        scrollbar = tk.Scrollbar(self.window4)
        scrollbar.pack(side="right", fill="both")

        canvas = tk.Canvas(self.window4, width=600, height=700, bg="white", highlightthickness=0,
                           yscrollcommand=scrollbar.set)
        canvas.pack(expand="yes", fill="both")
        canvas.create_image((0, 0), image=self.img_difference_gaussian_help, anchor="nw")
        scrollbar.config(command=canvas.yview)
        canvas.configure(scrollregion=canvas.bbox('all'))

    ####################################################################################################################
    # Here are the computer vision algorithms, deployed using the open source OpenCV package

    def colour_segment(self):
        self.img_no = 0
        self.buttonColour(0)

        lower_limit = ( int(self.text_box3.get('1.0', 'end-1c')),
                        int(self.text_box2.get('1.0', 'end-1c')),
                        int(self.text_box1.get('1.0', 'end-1c'))
                        )
        upper_limit = ( int(self.text_box6.get('1.0', 'end-1c')),
                        int(self.text_box5.get('1.0', 'end-1c')),
                        int(self.text_box4.get('1.0', 'end-1c'))
                        )

        img_no = self.var.get()
        image_name = self.image_dir + "image" + str(img_no + 1) + ".jpg"
        img = cv.imread(image_name)
        img_mask = cv.inRange(img, lower_limit, upper_limit)
        img_processed = cv.bitwise_and(img, img, mask=img_mask)
        cv.imwrite(self.image_dir + "image_processed.jpg", img_processed)
        self.root.img_processed = ImageTk.PhotoImage(Image.open(self.image_dir + "image_processed.jpg"))
        self.img_processed.configure(image=self.root.img_processed)

    def edge_detection(self):
        self.img_no = 1
        self.buttonColour(1)

        lower_limit = self.S1.get()
        upper_limit = self.S2.get()

        img_no = self.var.get()
        image_name = self.image_dir + "image" + str(img_no+1) + ".jpg"
        img = cv.imread(image_name)
        img_processed = cv.Canny(img, lower_limit, upper_limit)
        cv.imwrite(self.image_dir + "image_processed.jpg", img_processed)
        self.root.img_processed = ImageTk.PhotoImage(Image.open(self.image_dir + "image_processed.jpg"))
        self.img_processed.configure(image=self.root.img_processed)

    def smoothing_blurring(self):
        self.img_no = 2
        self.buttonColour(2)

        img_no = self.var.get()
        image_name = self.image_dir + "image" + str(img_no + 1) + ".jpg"
        img = cv.imread(image_name)

        kernel_size = (int(self.S3.get()), int(self.S3.get()))

        if not self.root.var_smooth.get():
            # average smoothing
            kernel = np.ones(kernel_size, np.float32) / kernel_size[0]**2
            img_processed = cv.filter2D(img, -1, kernel)

        else:
            # Gaussian blurring
            sigma = self.S4.get()
            img_processed = cv.GaussianBlur(img, (0,0), sigma) # kernel size computed as: [(sigma - 0.8)/0.3 + 1] / 0.5 + 1
                                                               # see opencv documentation

        cv.imwrite(self.image_dir + "image_processed.jpg", img_processed)
        self.root.img_processed = ImageTk.PhotoImage(Image.open(self.image_dir + "image_processed.jpg"))
        self.img_processed.configure(image=self.root.img_processed)

    def difference_gaussians(self):
        self.img_no = 3
        self.buttonColour(3)

        img_no = self.var.get()
        image_name = self.image_dir + "image" + str(img_no + 1) + ".jpg"
        img = cv.imread(image_name)

        sigma1 = self.S5.get()
        sigma2 = self.S6.get()
        img_gauss1 = cv.GaussianBlur(img, (0, 0), sigma1)
        img_gauss2 = cv.GaussianBlur(img, (0, 0), sigma2)
        img_processed = img_gauss1 - img_gauss2

        cv.imwrite(self.image_dir + "image_processed.jpg", img_processed)
        self.root.img_processed = ImageTk.PhotoImage(Image.open(self.image_dir + "image_processed.jpg"))
        self.img_processed.configure(image=self.root.img_processed)


if __name__ == '__main__':
    app = CVWindow()
