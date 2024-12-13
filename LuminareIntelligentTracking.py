import src.LITGuiWithClasses
import src.LITSubsystemInterface
import src.ObjectDetectionModel
import multiprocessing
import argparse
from src.ObjectDetectionModel import ObjectDetectionModel
from src.LITSubsystemInterface import LITSubsystemData
from src.LITGuiWithClasses import LITGUI
import warnings
warnings.filterwarnings("ignore")

def run_gui_process(camera_idx: int, number_of_leds: int = 256, numbmer_of_sections: int = 8, host:str = '', port: str = '', 
                    object_detect_status: bool=False, model_path: str='', label_path: str='',use_tpu: bool = False):
    model_path = r'C:\Users\brand\Documents\seniordesign\OldLITTest\ModelFiles\detect.tflite'
    label_path = r'C:\Users\brand\Documents\seniordesign\OldLITTest\ModelFiles\labelmap.txt'
    if object_detect_status:
        from tensorflow.lite.python.interpreter import Interpreter 
        from tensorflow.lite.python.interpreter import load_delegate
        object_detection_model = ObjectDetectionModel(model_path=model_path, use_edge_tpu=use_tpu, camera_index=camera_idx, label_path=label_path,resolution=(720, 405))
        lit_subsystem_data = LITSubsystemData(camera_idx, object_detection_model, number_of_leds=number_of_leds, number_of_sections=numbmer_of_sections, host=host, port=port)
        gui = LITGUI(lit_subsystem_data)
    else:
        lit_subsystem_data = LITSubsystemData(camera_idx, number_of_leds=256, number_of_sections=8, host=host, port=port)
        gui = LITGUI(lit_subsystem_data)
    gui.start_event_loop()
    return

def start_gui(camera_idx: int, number_of_leds: int = 256, numbmer_of_sections: int = 8, host:str = '', port: str = '', 
              object_detect_status: bool = False, model_path: str='', label_path: str='', use_tpu: bool = False)->multiprocessing.Process:
    p = multiprocessing.Process(target=run_gui_process, args=(camera_idx,number_of_leds, numbmer_of_sections, host, port, object_detect_status, model_path, label_path, use_tpu))
    p.start()
    return p

def set_command_line_arguments()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance_mode", help="(Optional) Run subsystems in parallel", action="store_true", default=False)
    parser.add_argument("--host", help='(Optional) Local IP address of the server for sending data', action='store', default=None)
    parser.add_argument('--camera_one_index', help='(Optional) The index to camera one of the lighting system.', action='store', default=0)
    parser.add_argument('--camera_two_index', help='(Optional) The index to camera two of the lighting system.', action='store', default=1)
    parser.add_argument("--camera_one_port", help='(Optional) The port camera one will be sending data to.', action='store', default=5000)
    parser.add_argument("--camera_two_port", help='(Optional) The port camera two will be sending data to.', action='store', default=5001)    
    return parser

if __name__ == '__main__':
    parser = set_command_line_arguments()
    args = parser.parse_args()
    performance_status = args.performance_mode
    host = args.host
    camera_one_index = args.camera_one_index
    camera_two_index = args.camera_two_index
    camera_one_port = args.camera_one_port
    camera_two_port = args.camera_two_port
    object_detection_tflite_path = r'ModelFiles\detect.tflite'
    object_detection_label_path = r'ModelFiles\labelmap.txt'
    # wifi_host='192.168.1.71'
    # ethernet_host = '169.254.31.162'
    ports = [5000, 5001]
    if performance_status:
        process1 = start_gui(camera_idx=camera_one_index, number_of_leds=256, numbmer_of_sections=8, host=host, 
                             port=ports[0], object_detect_status=True, model_path=object_detection_tflite_path, label_path=object_detection_label_path, use_tpu=False)
        process2 = start_gui(camera_idx=camera_two_index, number_of_leds=256, numbmer_of_sections=8, host=host, 
                             port=ports[1], object_detect_status=True, model_path=object_detection_tflite_path, label_path=object_detection_label_path, use_tpu=False)
        process1.join()
        process2.join()
    else:
        object_detection_model_one = ObjectDetectionModel(model_path=object_detection_tflite_path, use_edge_tpu=False, camera_index=camera_one_index, label_path=object_detection_label_path, resolution=(720, 405))
        object_detection_model_two = ObjectDetectionModel(model_path=object_detection_tflite_path, use_edge_tpu=False, camera_index=camera_two_index, label_path=object_detection_label_path, resolution=(720, 405))
        subsystem_one = LITSubsystemData(0, object_detection_model_one, number_of_leds=256, number_of_sections=8, host=host, port=camera_one_port)
        subsystem_two = LITSubsystemData(1, object_detection_model_two, number_of_leds=256, number_of_sections=8, host=host, port=camera_two_port)
        subsystem_list = [subsystem_two, subsystem_one]
        lit_gui = LITGUI(subsystem_list, background_images_dir=r'BackgroundImages')
        lit_gui.start_event_loop()
