if __name__ == '__main__':
    import wx
    import MainFrame

    app = wx.App()
    frame = MainFrame.MainFrame()
    frame.Show()
    app.MainLoop()
