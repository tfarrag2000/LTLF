import tkinter as tk
import MySQLdb

root = tk.Tk()
# geometry is width x height + x_offset + y_offset:
root.geometry("250x150+30+30")
db = MySQLdb.connect("localhost", "testuser", "test123", "TESTDB")
cursor = db.cursor()


def insFN():


def selFN():


L1 = tk.Label(root, text="Name")
L1.place(x=10, y=30, width=120, height=25)
L2 = tk.Label(root, text="Address")
L2.place(x=10, y=60, width=120, height=25)

T1 = tk.Entry(root)
T1.place(x=100, y=30, width=120, height=25)
T2 = tk.Entry(root)
T2.place(x=100, y=60, width=120, height=25)

B1 = tk.Button(root, text="insert", command=insFN)
B1.place(x=130, y=100, width=100, height=25)
B2 = tk.Button(root, text="select", command=selFN)
B2.place(x=20, y=100, width=100, height=25)
root.mainloop()

root.mainloop()
