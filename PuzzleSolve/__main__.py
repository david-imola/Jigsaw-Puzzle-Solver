if __name__ == '__main__':
    import wx
    import mainframe

    app = wx.App()
    frame = mainframe.MainFrame()
    frame.Show()
    app.MainLoop()
