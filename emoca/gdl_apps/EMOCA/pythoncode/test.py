import torch
import os
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager
from kivymd.uix.screen import MDScreen
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.button import MDFlatButton
from kivy.animation import Animation
from kivymd.toast import toast
from kivy.metrics import dp
from kivy.core.window import Window
from kivymd.uix.menu import MDDropdownMenu
from data_preprocessing import load_data, expression_labels
from chat_bot import EmotionChatBot
from model_prediction import load_saved_models, predict_emotion
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.core.text import LabelBase
from emoji_combiner import get_combined_emoji_image
from io import BytesIO
from kivy.core.image import Image as CoreImage

LabelBase.register(name="pangram", fn_regular=r'./fonts/Pangram-Regular.otf')
Window.size = (360, 640)

class SplashScreen(MDScreen):
    pass

class MainScreen(MDScreen):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = self.ids.get('camera', None)
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,  # Khi nhấn nút thoát hoặc back
            select_path=self.select_path,    # Khi một tệp được chọn
            preview=True,                    # Đặt True nếu muốn xem trước tệp
        )

    def file_manager_open(self):
        # camera = self.ids.get('camera', None)
        self.camera.play = False  # Tạm dừng camera
        self.file_manager.show('/')  # Hiển thị file manager từ thư mục gốc của filesystem
        self.manager_open = True


    def select_path(self, path):
        # Đây là callback khi một tệp được chọn
        self.exit_manager()
        #toast(path)  # Hiển thị đường dẫn tệp đã chọn
        # Now handle the selected image and perform emotion prediction
        if os.path.isfile(path):   
            self.manager.current = 'detector'
            emotion_screen = self.manager.get_screen('detector')
            emotion_screen.display_image_and_predict_emotion(path)


    def exit_manager(self, *args):
        # Đóng file manager
        self.manager_open = False
        # camera = self.ids.get('camera', None)
        self.camera.play = True  # Kích hoạt lại camera
        self.file_manager.close()

    def capture(self):
        '''
        Hàm này sẽ chụp ảnh từ Camera và xử lý ảnh đó.
        '''
        # camera = self.ids.get('camera', None)
        if self.camera:
            self.camera.play = False  # Tạm dừng camera
            filename = "capture.png"
            current_path = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_path, filename)
            self.camera.export_to_png(full_path)  # Lưu ảnh chụp vào tệp
            
            # Chuyển sang màn hình dự đoán
            self.manager.current = 'detector'
            emotion_screen = self.manager.get_screen('detector')
            emotion_screen.display_image_and_predict_emotion(full_path)

    def on_enter(self):
        '''Kích hoạt lại camera khi màn hình hiển thị.'''
        # camera = self.ids.get('camera', None)
        if self.camera:
            self.camera.play = True

    def on_pre_leave(self):
        '''Tắt camera khi chuẩn bị rời khỏi màn hình này.'''
        # camera = self.ids.get('camera', None)
        if self.camera:
            self.camera.play = False

class EmotionDetectorScreen(MDScreen):
    emotion_menu = None
    current_image_path = None
    def display_image_and_predict_emotion(self, image_path):
        # Đảm bảo ảnh đã được tải lên trước khi dự đoán
        #self.ids.image_display.source = image_path
        #self.ids.image_display.reload()  # Tải lại ảnh để hiển thị
        
        # Thực hiện dự đoán cảm xúc ở đây
        data = load_data(image_path)
        prediction = predict_emotion(data.reshape(1, -1), MDApp.get_running_app().models)
        predicted_emotion = expression_labels[prediction[0]]
        self.ids.label.text = predicted_emotion  
        MDApp.get_running_app().update_emoji_display(predicted_emotion, self)
        # Sau khi dự đoán xong, xóa ảnh
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Get a response from the chat bot
        chat_bot = EmotionChatBot()
        bot_response = chat_bot.get_response(predicted_emotion)
        self.ids.bot_response.text = bot_response
    
    def display_combined_emoji(self, base_emoji):
        current_emoji = self.ids.emoji_label_display.text
        combined_image_data = get_combined_emoji_image(current_emoji, base_emoji)
        
        if combined_image_data:
            # Load the combined image directly from the data
            combined_image_texture = CoreImage(BytesIO(combined_image_data), ext='png').texture
            self.ids.emoji_image_display.texture = combined_image_texture
            self.ids.emoji_image_display.opacity = 1
            self.ids.emoji_label_display.opacity = 0
        else:
            print("Failed to retrieve the combined emoji image")

    def menu_item_release(self, text_item):
        # Handle the menu item release event
        self.ids.bot_response.text = text_item  # For example, updating the bot response label
        self.emotion_menu.dismiss()

        

    # def on_pre_leave(self):
    #     '''Xử lý thêm khi chuẩn bị rời khỏi EmotionDetectorScreen.'''
    #     # Đảm bảo rằng ảnh chụp được xóa nếu tồn tại
    #     image_path = self.ids.image_display.source
    #     if os.path.exists(image_path):
    #         os.remove(image_path)
    
    

class EmotionDetectorApp(MDApp):
    models = None

    def build(self):
        self.title = "SentiMate"
        self.models = load_saved_models("./models", model_version="91")
        for key in self.models:
            if isinstance(self.models[key], torch.nn.Module):
                self.models[key] = self.models[key].to('cpu')
        
        sm = ScreenManager()
        sm.add_widget(SplashScreen(name='splash'))
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(EmotionDetectorScreen(name='detector'))

        # Schedule the main screen to replace splash screen after 2 seconds
        # Clock.schedule_once(lambda dt: sm.switch_to(sm.get_screen('main')), 10)
        Clock.schedule_once(self.show_main_screen, 20)
        Window.bind(on_drop_file=self.on_drop_file)
        return sm
    
    def show_main_screen(self, *args):
        self.root.current = 'main'

    def on_drop_file(self, window, file_path, x, y):
        if file_path:
            self.root.get_screen('main').choose_image([file_path.decode('utf-8')])

    def update_emoji_display(self, emotion, screen):
        emoji_mapping = {
            "Anger": "😠",
            "Disgust": "🤢",
            "Fear": "😨",
            "Happy": "😊",
            "Neutral": "😐",
            "Sad": "😢",
            "Surprise": "😮",
        }
        # Here 'screen' should be the instance of EmotionDetectorScreen which has the 'ids'
        screen.ids.emoji_label_display.text = emoji_mapping.get(emotion, "")
        screen.ids.emoji_label_display.opacity = 1
        screen.ids.emoji_image_display.opacity = 0


if __name__ == '__main__':
    EmotionDetectorApp().run()