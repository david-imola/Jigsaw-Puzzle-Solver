import wx
import core.puzzlize as pliz


class MainFrame(wx.Frame):

    def __init__(self, parent=None, title="Jigsaw Puzzle Solver", *args, **kw):
        super(MainFrame, self).__init__(parent=parent,
                                        title=title, *args, **kw)

        pan = wx.Panel(self)

        #wx.StaticText(pan, label="Hello World", pos=(25, 25))

        openButton = wx.Button(pan, label="Open File")
        openButton.Bind(wx.EVT_BUTTON, self.OnOpen)

    def OnOpen(self, event):
        with wx.FileDialog(self, "Open Picture File",
                           wildcard="Pictures (*.jpeg,*.jpg,*.png,*.ico,*.bmp)"
                           "|*.jpeg;*.jpg;*.png;*.ico;*.bmp") as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = fileDialog.GetPath()
            try:
                with open(pathname, 'r') as file:
                    pass
            except IOError:
                pass
