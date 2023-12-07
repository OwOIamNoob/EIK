import cv2
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.clock import Clock
import virtual
import super_virtual as svirtual

Window.size = (960, 720)
Window.top = 50
Window.left = 250


class EIK(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.screen = Builder.load_file('app.kv')

        self.enable = False
        self.mode = 'Normal'  # 'Super'
        Clock.schedule_interval(self.running, 1.0/300.0)

        self.virtual = virtual.Virtual()
        self.super_virtual = svirtual.SuperVirtual()

    def build(self):
        return self.screen

    def switch(self, box):
        other_box = (
            self.root.ids.normal
            if box == self.root.ids.super
            else self.root.ids.super
        )
        other_box.active = not box.active

    def active_keyboard(self):
        if self.enable:
            self.quit()

        else:
            self.enable = True
            self.screen.ids.button.text = 'Tắt'

            if self.screen.ids.normal.active:
                self.mode = 'Normal'
                self.virtual.open()
            else:
                self.mode = 'Super'
                self.super_virtual.open()

    def quit(self):
        cv2.destroyAllWindows()
        self.screen.ids.button.text = 'Bật ngay và luôn'
        self.enable = False

    def running(self, *args):
        if self.enable:
            if self.mode == 'Normal':
                self.virtual.run(self)
            else:
                self.super_virtual.run(self)


if __name__ == '__main__':
    EIK().run()
