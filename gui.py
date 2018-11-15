#!/usr/bin/python3

import tkinter as tk
import tkinter.messagebox as messagebox
from PIL import ImageTk, Image
import cv2 as cv


class CVWindow:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Computer Vision Examples")
        self.root.geometry("650x580")
        self.root.configure(background="white")

        # 5 sample images to choose from
        self.root.imgs = list()
        self.root.imgs.append(ImageTk.PhotoImage(Image.open("image1.jpg")))
        self.root.imgs.append(ImageTk.PhotoImage(Image.open("image2.jpg")))
        self.root.imgs.append(ImageTk.PhotoImage(Image.open("image3.jpg")))
        self.root.imgs.append(ImageTk.PhotoImage(Image.open("image4.jpg")))
        self.root.imgs.append(ImageTk.PhotoImage(Image.open("image5.jpg")))

        # To avoid the processed image being garbaged, it is saved upon each processing operation and opened here
        self.root.img_processed = ImageTk.PhotoImage(Image.open("image_processed.jpg"))

        # They can be selected with radio buttons
        self.Label1 = tk.Label(self.root, text="Select an image:", background="white")
        self.Label1.place(x=10, y=10)

        self.var = tk.IntVar()
        self.R1 = tk.Radiobutton(self.root, text="1", variable=self.var, value=0, background="white", command=lambda: self.loadImage(0))
        self.R2 = tk.Radiobutton(self.root, text="2", variable=self.var, value=1, background="white", command=lambda: self.loadImage(1))
        self.R3 = tk.Radiobutton(self.root, text="3", variable=self.var, value=2, background="white", command=lambda: self.loadImage(2))
        self.R4 = tk.Radiobutton(self.root, text="4", variable=self.var, value=3, background="white", command=lambda: self.loadImage(3))
        self.R5 = tk.Radiobutton(self.root, text="5", variable=self.var, value=4, background="white", command=lambda: self.loadImage(4))
        self.R1.place(x=120, y=10)
        self.R2.place(x=160, y=10)
        self.R3.place(x=200, y=10)
        self.R4.place(x=240, y=10)
        self.R5.place(x=280, y=10)


        # display the first image initially
        self.img_display = tk.Label(self.root, image=self.root.imgs[0])
        self.img_display.place(x=10, y=40)

        # insert an arrow between the two images
        self.root.img_arrow = ImageTk.PhotoImage(Image.open("arrow.jpg"))
        self.img_arrow_display = tk.Label(self.root, image=self.root.img_arrow, background="white")
        self.img_arrow_display.place(x=185, y=290)

        # display the processed image
        self.img_processed = tk.Label(self.root, image=self.root.img_processed)
        self.img_processed.place(x=10, y=320)

        # The buttons handle the CV algorithm employed on the image
        self.buttons = list()
        self.buttons.append(tk.Button(self.root, text="Colour Segmentation", borderwidth=1, background="light grey",
                                      command=lambda: self.colour_segment()))
        self.buttons.append(tk.Button(self.root, text="Edge Detection", borderwidth=1, background="light grey",
                                      command=lambda: self.edge_detection()))
        self.buttons.append(tk.Button(self.root, text="Blurring and Smoothing", borderwidth=1, background="light grey",
                                      command=lambda: self.smoothing_blurring()))
        self.buttons[0].place(x=420, y=40)
        self.buttons[1].place(x=420, y=170)
        self.buttons[2].place(x=420, y=300)

        # Buttons that open a help window and that describe the algorithms
        self.BH1 = tk.Button(self.root, text="?")
        self.BH2 = tk.Button(self.root, text="?")
        self.BH3 = tk.Button(self.root, text="?")
        self.BH1.place(x=620, y=40)
        self.BH2.place(x=620, y=170)
        self.BH3.place(x=620, y=300)

        # The first algorithm has 6 values that can be changed: the lower and upper limits of colour segmentation for RGB
        self.label1 = tk.Label(self.root, text="Lower limit: ", background="white")
        self.label2 = tk.Label(self.root, text="Upper limit: ", background="white")
        self.label1.place(x=420, y=100)
        self.label2.place(x=420, y=125)

        self.text_box1 = tk.Text(height=1, width=4)
        self.text_box2 = tk.Text(height=1, width=4)
        self.text_box3 = tk.Text(height=1, width=4)
        self.text_box4 = tk.Text(height=1, width=4)
        self.text_box5 = tk.Text(height=1, width=4)
        self.text_box6 = tk.Text(height=1, width=4)
        self.text_box1.place(x=500, y=100)
        self.text_box2.place(x=535, y=100)
        self.text_box3.place(x=570, y=100)
        self.text_box4.place(x=500, y=125)
        self.text_box5.place(x=535, y=125)
        self.text_box6.place(x=570, y=125)

        # We provide example values from the start
        self.text_box1.insert("end", "0")
        self.text_box2.insert("end", "80")
        self.text_box3.insert("end", "0")
        self.text_box4.insert("end", "255")
        self.text_box5.insert("end", "150")
        self.text_box6.insert("end", "255")

        tk.Label(text="R", background="white").place(x=510, y=80)
        tk.Label(text="G", background="white").place(x=545, y=80)
        tk.Label(text="B", background="white").place(x=580, y=80)

        # The edge detector has two limits that can be altered using a slider
        self.label1 = tk.Label(self.root, text="Lower limit: ", background="white")
        self.label2 = tk.Label(self.root, text="Upper limit: ", background="white")
        self.label1.place(x=420, y=217)
        self.label2.place(x=420, y=267)

        self.S1 = tk.Scale(self.root, from_=0, to=500, orient="horizontal")
        self.S1.place(x=500, y=200)
        self.S2 = tk.Scale(self.root, from_=0, to=500, orient="horizontal")
        self.S2.place(x=500, y=250)


        # self.S3 = tk.Scale(self.root, from_=0, to=255, orient="horizontal")
        # self.S3.grid(row=4, column=8)
        # self.S4 = tk.Scale(self.root, from_=1, to=15, resolution=2, orient="horizontal")
        # self.S4.grid(row=5, column=8)

        self.root.mainloop()

    # There is a bug in tk.Scale which doesn't allow odd numbers in steps of two to be shown. This is a quick fix for that.
    def fix(self):
        val = self.S4.get()
        if val % 2 == 0:
            self.S4.set(val+1)


    def loadImage(self, img_no):
        self.img_display.configure(image=self.root.imgs[img_no])

    # Changes the colour of the button selected
    def buttonColour(self, button_no):
        for i in range(3):
            if not button_no == i:
                self.buttons[i].configure(background="light grey")
            else:
                pass
                self.buttons[i].configure(background="light green")

    def colour_segment(self):
        self.buttonColour(0)

        lower_limit = ( int(self.text_box1.get('1.0', 'end-1c')),
                        int(self.text_box2.get('1.0', 'end-1c')),
                        int(self.text_box3.get('1.0', 'end-1c'))
                        )
        upper_limit = ( int(self.text_box4.get('1.0', 'end-1c')),
                        int(self.text_box5.get('1.0', 'end-1c')),
                        int(self.text_box6.get('1.0', 'end-1c'))
                        )

        img_no = self.var.get()
        image_name = "image" + str(img_no + 1) + ".jpg"
        img = cv.imread(image_name)
        img_mask = cv.inRange(img, lower_limit, upper_limit)
        img_processed = cv.bitwise_and(img, img, mask=img_mask)
        cv.imwrite("image_processed.jpg", img_processed)
        self.root.img_processed = ImageTk.PhotoImage(Image.open("image_processed.jpg"))
        self.img_processed.configure(image=self.root.img_processed)

    def edge_detection(self):
        self.buttonColour(1)

        lower_limit = self.S1.get()
        upper_limit = self.S2.get()

        img_no = self.var.get()
        image_name = "image" + str(img_no+1) + ".jpg"
        img = cv.imread(image_name)
        img_processed = cv.Canny(img, lower_limit, upper_limit)
        cv.imwrite("image_processed.jpg", img_processed)
        self.root.img_processed = ImageTk.PhotoImage(Image.open("image_processed.jpg"))
        self.img_processed.configure(image=self.root.img_processed)

    def smoothing_blurring(self):
        #self.msg = messagebox.showinfo("!")
        self.buttonColour(2)

        # img_no = self.var.get()
        # image_name = "image" + str(img_no + 1) + ".jpg"
        # img = cv.imread(image_name)
        # kernel_size = (self.S4.get() + 1, self.S4.get() + 1)
        # img_processed = cv.GaussianBlur(img, kernel_size, 0)
        # cv.imwrite("image_processed.jpg", img_processed)
        # self.root.img_processed = ImageTk.PhotoImage(Image.open("image_processed.jpg"))
        # self.img_processed.configure(image=self.root.img_processed)

app = CVWindow()
#app.main()
#
#