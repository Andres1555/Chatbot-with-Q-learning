import tkinter as tk
from gui.app import ChatbotApp


def main():
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
