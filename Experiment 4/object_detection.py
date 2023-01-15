import torch
import numpy as np
import cv2
from time import time
import argparse
import threading
import pandas as pd
import os

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, url, processing_unit="cpu",out_file="output.webm"):
  
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = processing_unit
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.backends.cudnn.benchmark = True
            self.model.to(torch.device('cuda'))
        else:
            self.model.to(torch.device('cpu'))
            
        print(f"{self.device} available")

    def get_video_from_url(self):

        return cv2.VideoCapture(self._URL)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        
        frame = [frame]
        results = self.model(frame)
        if self.device  == "cpu":
            labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        else:
            labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        fps_list = []
        cap = self.get_video_from_url()
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'VP90')
        except Exception as e:
            print(e)
            return
        out = cv2.VideoWriter(self.out_file, fourcc, 20, (1920, 1080))
        while True:
            start_time = time()
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            lab = {}
            for label in results[0]:
                name =  self.class_to_label(label)
                if name not in lab:
                    lab[name]=0
                lab[name]+=1
            # print(lab)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            fps_list.append({
                                "frame_per_sec":fps,
                                "label_count":len(lab),
                            })
            # print(fps)
            out.write(frame)
        cap.release()
        df = pd.DataFrame(fps_list)
        return df



def parallel_execution(i,args,results,thread_n,processing_unit):
    start_time = time()
    a = ObjectDetection(args.input_stream,processing_unit)
    fps = a()
    describe = fps.describe()
    end_time = time()
    elapsed_time = end_time - start_time
    # print(f"Time taken by the function ({i}):{elapsed_time} ")
    results[i] = {
        "time": elapsed_time,
        "thread": i,
        "thread_n": thread_n,
        "processing_unit":processing_unit,
        "total_frames": fps.shape[0], 
        "fps_mean": describe["frame_per_sec"]["mean"],
        "fps_max": describe["frame_per_sec"]["max"],
        "fps_min": describe["frame_per_sec"]["min"],
        "label_count_mean": describe["label_count"]["mean"],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_stream', type=str, help="The destination IP address to use")
    parser.add_argument('max_thread_number', type=int, help="The number of threads to use")
    # parser.add_argument('processing_unit', type=str, help="Choose gpu or cpu")
    # parser.add_argument('thread_number', type=str, help="The number of threads to use")
    args = parser.parse_args()
    processing_units = ["cpu","cuda"]

    all_results = []
    for processing_unit in processing_units:
        for thread_n in (args.max_thread_number,0,-2):
            print(f"processing unit: {processing_unit}, thread count:{thread_n}")
            overall_start_time = time()
            threads = []

            results = [None] * thread_n
            for i in range(thread_n):
                t = threading.Thread(target=parallel_execution, args=(i,args,results,thread_n,processing_unit))
                threads.append(t)
                t.start()

            # Wait for all threads to complete
            for t in threads:
                t.join()

            overall_end_time = time()
            overall_elapsed_time = overall_end_time - overall_start_time
            print(f"Overall time taken: {overall_elapsed_time}")

            df = pd.DataFrame(results)
            df["overall_time"] = overall_elapsed_time
            df["processing_unit"] = processing_unit
            df["thread_n"] = thread_n

            all_results.append(df)

    # concatenate all results into one dataframe
    all_results_df = pd.concat(all_results)
    # Write the dataframe to a CSV file
    all_results_df.to_csv("results.csv", header=True)

            




if __name__ == '__main__':
    main()
