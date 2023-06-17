import numpy as np
import cv2
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import sys
import numpy as np
import threading
import time
import redis
import kafka
import sounddevice as sd
import soundfile as sf

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Naloga 4")
        self.View = View()
        # self.setCentralWidget(self.View)
        self.controler = Controler(self.View)

        layout = QVBoxLayout()

        button_file = QPushButton("Select File")
        button_file.clicked.connect(self.controler.choose_file_path)
        #button_file.setStyleSheet("QPushButton::checked { background-color: green; }")
        button_stop = QPushButton("Stop")
        button_stop.clicked.connect(self.controler.end_video)
        button_start = QPushButton("Start")
        button_start.clicked.connect(self.controler.start_video)
        group_box_alg = QGroupBox("Video server")
        group_box_alg_layout = QVBoxLayout()


        gumb_group_selekcija_algoritma = QButtonGroup()
        gumb_group_selekcija_algoritma.setExclusive(False)
        
        radio_button1 = QPushButton("Noise on")
        radio_button2 = QPushButton("Zoom on")
        radio_button1.clicked.connect(self.controler.on_noise_clicked)
        radio_button2.clicked.connect(self.controler.on_rotate_clicked)
        gumb_group_selekcija_algoritma.addButton(radio_button1)
        gumb_group_selekcija_algoritma.addButton(radio_button2)

        group_box_alg_layout.addWidget(radio_button1)
        group_box_alg_layout.addWidget(radio_button2)
        group_box_alg_layout.addWidget(button_start)
        group_box_alg_layout.addWidget(button_stop)
        group_box_alg_layout.addWidget(button_file)
        group_box_alg.setLayout(group_box_alg_layout)

        spin_box = QSpinBox()
        spin_box.setMinimum(1)
        spin_box.setMaximum(100)
        spin_box.valueChanged.connect(lambda: self.controler.on_generiraj_clicked(spin_box.value()))
        group_box_alg_layout.addWidget(spin_box)
        layout.addWidget(group_box_alg)

        group_box_logika = QGroupBox("Logika server")
        group_box_logika_layout = QVBoxLayout()

        radio_button3 = QPushButton("Start")
        radio_button4 = QPushButton("Stop")
        radio_button3.clicked.connect(self.controler.start_logika)
        radio_button4.clicked.connect(self.controler.end_logika)
        group_box_logika_layout.addWidget(radio_button3)
        group_box_logika_layout.addWidget(radio_button4)
        group_box_logika.setLayout(group_box_logika_layout)

        layout.addWidget(group_box_logika)
        
        # pbutton1 = QPushButton("Set")
        # pbutton1.clicked.connect(lambda: self.controler.on_generiraj_clicked(spin_box.value()))
        # group_box_tocke = QGroupBox("Vzorcevalna frekvenca")
        # group_box_tocke_layout = QVBoxLayout()
        # group_box_tocke_layout.addWidget(spin_box)
        # # group_box_tocke_layout.addWidget(pbutton1)
        # group_box_tocke.setLayout(group_box_tocke_layout)

        # layout.addWidget(group_box_tocke)

        widget = QWidget()
        widget.setLayout(layout)
        dock_widget = QDockWidget(self)
        dock_widget.setWidget(widget)
        
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_widget)

    def prnt(self):
        radioButton = self.sender()
        print(radioButton.text())


class View(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.video_size = QSize(620, 480)
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.video_size)
        # self.setScene(QGraphicsScene(self))
        # self.setSceneRect(QRect(self.viewport().rect()))

    def display_video_stream(self, frame):
        image = QImage(frame, frame.shape[1], frame.shape[0], 
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image)) #image



class Model:
    def __init__(self, view=None):
        self.red = redis.Redis()
        self.topic = 'frame_noticifation'
        self.producer = kafka.KafkaProducer(bootstrap_servers='localhost:9092')

        self.path = 1
        self.fvz = 30
        self.view = view
        self.noise_is_true = False
        self.rotate_is_true = False
        

    def thread_video(self):
        cap = cv2.VideoCapture(self.path)

        print("Starting frame generation")
        while True:
            t_start = time.perf_counter()
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if self.noise_is_true:
                noise = np.random.normal(0, 0.5, frame.shape).astype(np.uint8)
                frame = cv2.add(frame, noise)

            if self.rotate_is_true:
                h, w = frame.shape[:2]
                cut_size = (h//4, w//4, 3*h//4, 3*w//4)
                crop = frame[cut_size[0]:cut_size[2], cut_size[1]:cut_size[3]]
                frame = cv2.resize(crop, (w, h))

            self.red.set("frame:latest", np.array(frame).tobytes())
            future = self.producer.send(self.topic, b"new_frame", timestamp_ms=round(time.time()*1000))

            try:
                rm = future.get(timeout=10)
            except kafka.KafkaError:
                pass

            t_stop = time.perf_counter()
            t_elapsed = t_stop - t_start
            t_frame = 1000 / self.fvz / 1000
            t_sleep = t_frame - t_elapsed
            if t_sleep > 0:
                time.sleep(t_sleep)

            if self.event.is_set():
                print("Stopped frame generation")
                self.noise_is_true = False
                self.rotate_is_true = False
                self.path = 1
                cap.release()
                break

class Model_logika:
    def __init__(self, view=None):
        self.topic = 'frame_rezultati'
        self.consumer = kafka.KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='latest', consumer_timeout_ms=2000)
        self.consumer.subscribe([self.topic])
        self.fvz = 10
        self.data, self.samplerate = sf.read("start-up/warning-sound-6686.mp3")

    def thread_logika(self):
        print("Starting logika")
        eye_closed_counter = 0
        eye_open_counter = 0
        face_utrujen_counter = 0
        face_ne_utrujen_counter = 0
        
        sound = False
        sound_thread = None

        #faktor = 4

        while True:
            for message in self.consumer:
                decoded_message = message.value.decode("utf-8")
                split_message = decoded_message.split(",")
                eye = split_message[1]
                face = split_message[0]
                #print(eye, face)

                if eye == "zaprte":
                    eye_closed_counter += 1
                    eye_open_counter = 0

                if eye == "odprte":
                    eye_open_counter += 1
                    eye_closed_counter = 0

                if face == "utrujen":
                    face_utrujen_counter += 1
                    face_ne_utrujen_counter = 0

                if face == "ne_utrujen":
                    face_ne_utrujen_counter += 1
                    face_utrujen_counter = 0


                if ((eye_closed_counter > 4 ) or (face_utrujen_counter > 4)) and sound == False:
                    print("START sound")
                    self.sound_start()
                    sound = True
                    
                if (eye_open_counter > 2) and (face_ne_utrujen_counter > 2) and sound == True:
                    print("STOP sound")
                    self.sound_stop()
                    sound = False
                    
                
                if self.event.is_set():
                    break   

            # Stop loop
            if self.event.is_set():
                print("Stopping logika")
                self.sound_stop()
                break

    def sound_start(self):
        self.thread_sound = threading.Thread(target=lambda: self.warning_sound())
        self.event_sound = threading.Event()
        self.thread_sound.start()

    def sound_stop(self):
        self.event_sound.set()
        self.thread_sound.join()

    def warning_sound(self):
        while not self.event_sound.is_set():
            sd.play(self.data, self.samplerate)
            sd.wait()          
        

class Controler(QObject):
    def __init__(self, view):
        super().__init__()
        self.view = view
        self.model = Model(view=self.view)
        self.model_logika = Model_logika(view=self.view)

    #LOGIKA SERVER
    def start_logika(self):
        self.model_logika.thread = threading.Thread(target=lambda: self.model_logika.thread_logika())
        self.model_logika.event = threading.Event()
        self.model_logika.thread.start()

    def end_logika(self):
        self.model_logika.event.set()
        self.model_logika.thread.join()

    #VIDEO SERVER
    def start_video(self):
        self.model.thread = threading.Thread(target=lambda: self.model.thread_video())
        self.model.event = threading.Event()
        self.model.thread.start()

    def end_video(self):
        self.model.event.set()
        self.model.thread.join()
        
    def choose_file_path(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.exec()
        file_paths = file_dialog.selectedFiles()
        if file_paths:
            chosen_file_path = file_paths[0]
            self.model.path = chosen_file_path

    def on_generiraj_clicked(self, value):
        self.model.fvz = value
        self.model_logika.fvz = value

    def on_noise_clicked(self):
        self.model.noise_is_true = True

    def on_rotate_clicked(self):
        self.model.rotate_is_true = True

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())

