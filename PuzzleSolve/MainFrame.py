import wx


class MainFrame(wx.Frame):

    def __init__(self, parent=None, title="Jigsaw Puzzle Solver", *args, **kw):
        super(MainFrame, self).__init__(parent=parent, title=title, *args, **kw)

        pan = wx.Panel(self)

        wx.StaticText(pan, label="Hello World", pos=(25, 25))
